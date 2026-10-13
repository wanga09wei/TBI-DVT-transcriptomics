# -*- coding: utf-8 -*-
"""生成 Fig 1 队列筛选流程图 与 Fig 2 三时点纵向轨迹图（英文标注，投稿可用）"""
import pandas as pd, numpy as np, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,
                     'savefig.dpi':300,'savefig.bbox':'tight','figure.facecolor':'white'})

D=pd.read_pickle('merged.pkl'); E=D[D['血栓0_1'].notna()].copy()
loc=E['血栓位置'].astype(str)
PROX=r'腘|股总|股浅|股深|股静脉|髂'; CALF=r'肌间|胫后|胫前|腓'; SUP=r'大隐|小隐'
E['g']=np.where(loc.str.contains(PROX),'P',np.where(loc.str.contains(CALF),'D',np.where(loc.str.contains(SUP),'S','U')))
def num(x):
    s=pd.Series(x).astype(str).str.replace('——','',regex=False).str.replace('不详','',regex=False)
    return pd.to_numeric(s.str.replace(r'[^0-9.\-]','',regex=True),errors='coerce')
for t in ['T0','T1','T2']:
    for k in ['血小板','FIB','D-二聚体','WBC']:
        c=f'{t}_{k}'; E['v_'+c]=num(E[c]) if c in E.columns else np.nan

# ================= FIG 1 =================
fig,ax=plt.subplots(figsize=(7.2,5.4)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
def box(x,y,w,h,txt,fc,ec,fs=9,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12,rounding_size=0.18",
                 fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',linespacing=1.45)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.1,color='#5a6672'))
box(2.4,8.7,5.2,1.0,'Trauma registry, 2014–2018\nTBI with concomitant lower-limb fracture\n338 records screened',
    '#e8f1fb','#2f6fb0',8.6,True)
arrow(5.0,8.7,5.0,8.15)
box(2.9,7.1,4.2,0.95,'Excluded: 53 records\nno venous ultrasound assessment',"#fdeeee",'#c98a8a',8.4)
arrow(5.0,7.1,5.0,6.6)
box(2.2,5.55,5.6,0.95,'Analysis population: 285 patients with VTE assessment\n284/285 (99.6%) DVT-positive',
    '#eaf7ef','#3f9163',8.6,True)
arrow(5.0,5.55,5.0,5.0)
box(2.0,3.75,6.0,1.1,'Anatomic distribution of DVT\nProximal/trunk 33 (11.6%)  |  Distal calf only 240 (84.2%)\n'
    'Superficial only 3  |  Unspecified 9\nBilateral involvement 66 (23.2%)',"#eef2f7",'#7b8794',8.3)
arrow(5.0,3.75,5.0,3.2)
box(1.7,1.85,6.6,1.15,'Three-timepoint longitudinal laboratory course\nT0 admission  ->  T1 VTE detection (median 12 d post-injury)  ->  T2 2 wk after anticoagulation',
    '#fff6e5','#c99a34',8.3)
arrow(5.0,1.85,5.0,1.35)
box(2.4,0.25,5.2,1.0,'Median detection time 12 d (IQR 7–18)  |  IVC filter 57 (20.0%)\nBleeding after anticoagulation 11 (3.9%)  |  Thrombotic complications 9 (3.2%)',
    '#f2eefa','#8a72b8',8.1)
ax.set_title('Fig 1  Study cohort and DVT phenotype',fontsize=11,fontweight='bold',pad=6)
plt.savefig('../figures/Fig1_cohort_flow.png'); plt.close()

# ================= FIG 2 =================
panels=[('v_T0_血小板','v_T1_血小板','v_T2_血小板','A   Platelet (x10^9/L)','#c0392b',None),
        ('v_T0_FIB','v_T1_FIB','v_T2_FIB','B   Fibrinogen (g/L)','#d35400',None),
        ('v_T0_D-二聚体','v_T1_D-二聚体','v_T2_D-二聚体','C   D-dimer (mg/L)','#2471a3',True),
        ('v_T0_WBC','v_T1_WBC','v_T2_WBC','D   WBC (x10^9/L)','#1e8449',None)]
fig,axes=plt.subplots(2,2,figsize=(8.6,6.4))
for ax,(c0,c1,c2,title,col,ylog) in zip(axes.ravel(),panels):
    M=E[[c0,c1,c2]].apply(pd.to_numeric,errors='coerce').values
    for r in M:
        if np.sum(~np.isnan(r))>=2:
            ax.plot([0,1,2],r,color=col,alpha=0.07,lw=0.7,zorder=1)
    med=[np.nanmedian(M[:,i]) for i in range(3)]
    q1=[np.nanpercentile(M[:,i],25) for i in range(3)]; q3=[np.nanpercentile(M[:,i],75) for i in range(3)]
    ax.fill_between([0,1,2],q1,q3,color=col,alpha=0.16,zorder=2)
    ax.plot([0,1,2],med,'o-',color=col,lw=2.4,ms=6,zorder=4,label='Median (IQR)')
    for i,m in enumerate(med): ax.annotate(f'{m:.2f}',(i,m),textcoords='offset points',
        xytext=(0,9),ha='center',fontsize=8,color=col,fontweight='bold')
    ns=[int(np.sum(~np.isnan(M[:,i]))) for i in range(3)]
    ax.set_xticks([0,1,2]); ax.set_xticklabels([f'T0\n(n={ns[0]})',f'T1\n(n={ns[1]})',f'T2\n(n={ns[2]})'],fontsize=8)
    ax.set_title(title,fontsize=9.5,fontweight='bold',loc='left')
    ax.grid(alpha=0.22,ls=':'); ax.set_axisbelow(True)
    if ylog: ax.set_yscale('log')
    for s in ['top','right']: ax.spines[s].set_visible(False)
axes[0,0].legend(fontsize=7.5,frameon=False,loc='upper left')
fig.suptitle('Fig 2  Three-timepoint longitudinal laboratory trajectories in TBI + lower-limb fracture with DVT',
             fontsize=10,fontweight='bold',y=0.99)
fig.text(0.5,0.005,'T0, admission; T1, DVT detection (median 12 days post-injury); T2, 2 weeks after anticoagulation. '
         'Grey lines, individual patients.',ha='center',fontsize=7.6,style='italic')
plt.tight_layout(rect=[0,0.028,1,0.965]); plt.savefig('../figures/Fig2_trajectories.png'); plt.close()
print("[saved] figures/Fig1_cohort_flow.png  figures/Fig2_trajectories.png")
