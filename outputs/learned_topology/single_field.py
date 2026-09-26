"""A smaller Euler front end leaves room for six hidden units under512weights.

Uses the exact audited topological operator from euler_layer.py; changes only
channels and the matching readout dimensions. Same variants for geometry.
"""
import torch
from torch import nn
from euler_layer import SmallClassifier,initialize_filters

class SingleFieldClassifier(SmallClassifier):
    def __init__(self,kind,mean,std,keep,budget=512):
        super().__init__(kind,mean,std,keep,budget)
        assert kind in ['euler_learned','euler_frozen','area_learned','perimeter_learned','geometry_learned']
        self.front.conv=nn.Conv2d(1,1,3,padding=1);initialize_filters(self.front.conv)
        q=2 if self.front.mode=='geometry' else 4
        self.front.thresholds=nn.Parameter(torch.linspace(.15,.75,q)[None])
        if kind=='euler_frozen':
            for p in self.front.parameters():p.requires_grad_(False)
        self.norm=nn.BatchNorm1d(4,momentum=.05)
        overhead=sum(p.numel() for p in self.front.parameters())+sum(p.numel() for p in self.norm.parameters())
        dim=68;self.hidden=(budget-overhead-10)//(dim+11)
        self.body=nn.Sequential(nn.Linear(dim,self.hidden),nn.ReLU(),nn.Linear(self.hidden,10))
        self.trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.stored_parameters=sum(p.numel() for p in self.parameters());assert self.stored_parameters<=budget
