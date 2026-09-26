"""Static scientific figures from frozen numerical records."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/tiny_topology'
os.environ['MPLCONFIGDIR']=str(ROOT/'work/matplotlib')
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#2364AA';ORANGE='#D97720';GREY='#687684'


def training():
    sys.path.insert(0,str(ROOT/'outputs/real_topology'))
    from optdigits_pilot import read_original
    from optdigits_ablation import scale_features
    images,y=read_original('tra');top,_=scale_features(images)
    fig=plt.figure(figsize=(12.0,5.6));gs=fig.add_gridspec(2,6,height_ratios=[1.65,1],hspace=.65,wspace=.42)
    ax=fig.add_subplot(gs[0,:4]);means=np.array([[(top[y==k,2*r+1]>0).mean() for r in range(4)] for k in range(10)]).T
    im=ax.imshow(means,aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(10));ax.set_xlabel('Метка цифры');ax.set_yticks(range(4),['0','1','2','3']);ax.set_ylabel('Радиус замыкания');ax.set_title('Доля изображений с отверстиями по классам',loc='left',fontweight='bold')
    for r in range(4):
        for k in range(10):ax.text(k,r,f'{means[r,k]*100:.0f}%',ha='center',va='center',fontsize=8,color='white' if means[r,k]>.55 else '#243447')
    ax2=fig.add_subplot(gs[0,4:]);ax2.bar([0,1,2,3],[(top[:,1]==k).mean()*100 if k<3 else (top[:,1]>=3).mean()*100 for k in range(4)],color=BLUE);ax2.set_xticks([0,1,2,3],['0','1','2','3+']);ax2.set_xlabel('Число отверстий, β₁');ax2.set_ylabel('% изображений');ax2.set_title('Без замыкания',loc='left',fontweight='bold');ax2.grid(axis='y',alpha=.18);ax2.set_axisbelow(True)
    for col,k in enumerate([0,1,4,6,8,9]):
        # Deterministic first training image of each label, not cherry-picked by quality.
        i=np.flatnonzero(y==k)[0];a=fig.add_subplot(gs[1,col]);a.imshow(images[i],cmap='gray_r',interpolation='nearest');a.set_axis_off();a.set_title(f'Цифра {k}\nβ₀={int(top[i,0])}, β₁={int(top[i,1])}',fontsize=10)
    fig.suptitle('Топология реальных рукописных цифр OptDigits',x=.08,ha='left',fontweight='bold',fontsize=16,y=.99)
    fig.text(.08,.01,'Только обучающая часть TRA: 1934 изображения. Примеры — первое изображение каждого показанного класса.',fontsize=9,color=GREY)
    fig.subplots_adjust(top=.86,bottom=.11,left=.08,right=.98)
    fig.savefig(OUT/'tiny_train_topology.png',dpi=170);plt.close(fig)


def final():
    r=json.loads((OUT/'tiny_confirmation_test_results.json').read_text());p=r['primary'];curve=r['secondary_budget_curve']
    fig,axs=plt.subplots(1,3,figsize=(15,4.9),gridspec_kw={'width_ratios':[1.4,1,1.1]});ax=axs[0]
    budgets=np.array([v['budget'] for v in curve]);x=np.arange(len(curve))
    for label,name,color in [('baseline','Без топологических признаков',BLUE),('topology','С топологическими признаками',ORANGE)]:
        ys=np.array([v[label+'_mean_accuracy'] for v in curve])*100
        seeds=np.array([v[label+'_seed_accuracy'] for v in curve])*100
        ax.plot(x,ys,'o-',color=color,label=name,lw=2)
        ax.fill_between(x,seeds.min(1),seeds.max(1),color=color,alpha=.09)
    ax.axvspan(.8,1.2,color='#aab7c4',alpha=.15,zorder=-1);ax.set_xticks(x,[str(b) for b in budgets]);ax.set_xlabel('Предел числа обучаемых весов');ax.set_ylabel('Средняя точность, %');ax.set_title('Зависимость от бюджета',loc='left',fontweight='bold');ax.legend(fontsize=8,loc='lower right',frameon=False);ax.grid(axis='y',alpha=.2)
    ax=axs[1];b=np.array(r['seeds']['baseline_accuracy'])*100;t=np.array(r['seeds']['topology_accuracy'])*100
    for s in range(len(b)):ax.plot([0,1],[b[s],t[s]],color=ORANGE if t[s]>b[s] else GREY,alpha=.4,lw=1,marker='o',ms=3)
    ax.plot([0,1],[b.mean(),t.mean()],color='#152C42',lw=3,marker='D',ms=6,label='Среднее 10 обучений');ax.set_xticks([0,1],['Контроль','Топология']);ax.set_xlim(-.25,1.25);ax.set_ylabel('Точность, %');ax.set_title('Основной бюджет: 512',loc='left',fontweight='bold');ax.legend(fontsize=8,frameon=False,loc='lower right');ax.grid(axis='y',alpha=.2)
    ax=axs[2];ci=np.array(p['crossed_seed_image_bootstrap95'])*100;delta=p['mean_accuracy_difference']*100
    ax.axvline(0,color=GREY,lw=1);ax.errorbar([delta],[0],xerr=np.array([[delta-ci[0]],[ci[1]-delta]]),fmt='o',color=ORANGE,elinewidth=3,capsize=7,ms=9)
    ax.set_yticks([]);ax.set_ylim(-1,1);ax.set_xlabel('Изменение точности, п.п.');ax.set_title('Разность и интервал 95%',loc='left',fontweight='bold');ax.text(.03,.86,f'Δ = {delta:+.3f} п.п.\n95% ДИ [{ci[0]:+.3f}; {ci[1]:+.3f}]\np × 6 = {p["bonferroni6_p"]:.4g}',transform=ax.transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',edgecolor='none',pad=2))
    passed=r['success_gate_passed'];ax.text(.03,.07,'Заданный критерий пройден' if passed else 'Заданный критерий не пройден',transform=ax.transAxes,fontsize=9,color='#277047' if passed else '#9B4D24');ax.grid(axis='x',alpha=.2)
    fig.suptitle('OptDigits WDEP: качество отдельных маленьких сетей',x=.065,ha='left',fontsize=16,fontweight='bold')
    fig.text(.065,.055,'943 ранее не использованных изображения знакомых авторов; 10 фиксированных обучений, без ансамблирования.\n512 — основной тест среди MLP/CNN; остальные бюджеты — описательная кривая MLP. Заливка слева — диапазон по обучениям.',fontsize=9,color=GREY)
    fig.subplots_adjust(top=.82,bottom=.22,left=.065,right=.98,wspace=.4)
    fig.savefig(OUT/'tiny_test_results.png',dpi=180);fig.savefig(OUT/'tiny_test_results.pdf');plt.close(fig)

if __name__=='__main__':
    if '--final' in sys.argv:final()
    else:training()
