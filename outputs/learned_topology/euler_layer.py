"""End-to-end cubical Euler-curve layer for images.

For superlevel sets of face (pixel) heights z, every edge/vertex appears when
ANY incident face appears, hence its height is the maximum incident face height.
Replacing each inclusion indicator by a logistic CDF gives the exact expectation
of Euler characteristic under ONE shared logistic perturbation of the threshold.
It is not the Euler characteristic of an independent Bernoulli pixel field.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def cell_heights(z):
    """z: [B,C,H,W]; absent exterior faces have height -infinity."""
    padded=F.pad(z,(1,1,1,1),value=float('-inf'))
    horizontal=torch.maximum(padded[:,:,:-1,1:-1],padded[:,:,1:,1:-1])
    vertical=torch.maximum(padded[:,:,1:-1,:-1],padded[:,:,1:-1,1:])
    vertices=torch.maximum(torch.maximum(padded[:,:,:-1,:-1],padded[:,:,1:,:-1]),
                           torch.maximum(padded[:,:,:-1,1:],padded[:,:,1:,1:]))
    return vertices,horizontal,vertical,z


def smooth_cell_counts(z,thresholds,temperature=.1):
    """Return V,E,F with shape [B,C,Q]. Differentiable almost everywhere in z."""
    vertices,horizontal,vertical,faces=cell_heights(z)
    def count(v):return torch.sigmoid((v[:,:,None]-thresholds[None,:,:,None,None])/temperature).sum((-1,-2))
    return count(vertices),count(horizontal)+count(vertical),count(faces)


def smooth_euler(z,thresholds,temperature=.1):
    v,e,f=smooth_cell_counts(z,thresholds,temperature)
    return v-e+f


def initialize_filters(conv):
    """Common initialization for topological, geometric and ordinary CNNs."""
    with torch.no_grad():
        conv.weight.zero_();conv.bias.zero_()
        for c in range(conv.out_channels):
            if c%3==0:conv.weight[c,0,1,1]=1.
            elif c%3==1:conv.weight[c,0].fill_(1/9)
            else:conv.weight[c,0,1,1]=2.;conv.weight[c,0].sub_(1/9)
        conv.weight.add_(.005*torch.randn_like(conv.weight))


class EulerCurveLayer(nn.Module):
    """Learn filters and filtration thresholds through a cubical Euler sum."""
    def __init__(self,mode='euler',frozen=False,temperature=.1):
        super().__init__();self.mode=mode;self.temperature=temperature;self.frozen=frozen
        self.conv=nn.Conv2d(1,2,3,padding=1);initialize_filters(self.conv)
        q=2 if mode=='geometry' else 4
        self.thresholds=nn.Parameter(torch.linspace(.15,.75,q).repeat(2,1))
        if frozen:
            for p in self.conv.parameters():p.requires_grad_(False)
            self.thresholds.requires_grad_(False)
    def forward(self,x):
        z=self.conv(x)
        v,e,f=smooth_cell_counts(z,self.thresholds,self.temperature)
        if self.mode=='euler':features=v-e+f
        elif self.mode=='area':features=f
        elif self.mode=='perimeter':features=2*e-4*f
        elif self.mode=='geometry':features=torch.cat([f,2*e-4*f],2)
        else:raise ValueError(self.mode)
        return features.flatten(1)


class SmallClassifier(nn.Module):
    def __init__(self,kind,mean,std,keep,budget=512):
        super().__init__();self.kind=kind
        self.register_buffer('pixel_mean',mean.flatten().float());self.register_buffer('pixel_std',std.flatten().float());self.register_buffer('pixel_keep',keep.flatten().bool())
        self.front=None;self.norm=None;self.pool=None
        if kind=='mlp':dim=64;overhead=0
        elif kind.startswith('cnn_'):
            parts=kind.split('_');self.skip='skip' in parts;c=int(parts[-1][1:])
            self.front=nn.Conv2d(1,c,3,padding=1);initialize_filters(self.front);self.norm=nn.BatchNorm2d(c,momentum=.05);self.pool=nn.AdaptiveAvgPool2d(4)
            dim=16*c+(64 if self.skip else 0);overhead=sum(p.numel() for p in self.front.parameters())+sum(p.numel() for p in self.norm.parameters())
        else:
            mode='euler' if kind.startswith('euler_') else kind.removesuffix('_learned')
            self.front=EulerCurveLayer(mode=mode,frozen=kind=='euler_frozen');self.norm=nn.BatchNorm1d(8,momentum=.05);dim=72
            # Frozen ablation retains the same head width as the learned Euler net.
            overhead=sum(p.numel() for p in self.front.parameters())+sum(p.numel() for p in self.norm.parameters())
        self.hidden=(budget-overhead-10)//(dim+11);assert self.hidden>=1
        self.body=nn.Sequential(nn.Linear(dim,self.hidden),nn.ReLU(),nn.Linear(self.hidden,10))
        self.trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.stored_parameters=sum(p.numel() for p in self.parameters());assert self.stored_parameters<=budget
    def forward(self,x):
        raw=x.flatten(1);pixels=torch.where(self.pixel_keep,(raw-self.pixel_mean)/self.pixel_std,0.)
        if self.kind=='mlp':features=pixels
        elif self.kind.startswith('cnn_'):
            side=self.pool(F.relu(self.norm(self.front(x)))).flatten(1)
            features=torch.cat([pixels,.3*side],1) if self.skip else side
        else:features=torch.cat([pixels,.3*self.norm(self.front(x))],1)
        return self.body(features)
