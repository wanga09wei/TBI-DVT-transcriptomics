# -*- coding: utf-8 -*-
"""仅重绘 Fig 1（纯文本流程图，不依赖 merged.pkl），修复文字超框问题"""
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,
                     'savefig.dpi':300,'savefig.bbox':'tight','figure.facecolor':'white'})

fig,ax=plt.subplots(figsize=(7.2,5.8)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
BX,LW=1.0,8.0   # 统一框左缘与宽度，保证长行不溢出
def box(x,y,w,h,txt,fc,ec,fs=9,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12,rounding_size=0.18",
                 fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,
            fontweight='bold' if bold else 'normal',linespacing=1.45)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1.1,color='#5a6672'))
box(BX,8.75,LW,1.0,'Trauma registry, 2014–2018\nTBI with concomitant lower-limb fracture\n338 records screened',
    '#e8f1fb','#2f6fb0',8.6,True)
arrow(5.0,8.75,5.0,8.2)
box(BX,7.15,LW,0.95,'Excluded: 53 records\nno venous ultrasound assessment',"#fdeeee",'#c98a8a',8.4)
arrow(5.0,7.15,5.0,6.6)
box(BX,5.55,LW,0.95,'Analysis population: 285 patients with VTE assessment\n284/285 (99.6%) DVT-positive',
    '#eaf7ef','#3f9163',8.5,True)
arrow(5.0,5.55,5.0,5.0)
box(BX,3.8,LW,1.1,'Anatomic distribution of DVT\nProximal/trunk 33 (11.6%)  |  Distal calf only 240 (84.2%)\n'
    'Superficial only 3  |  Unspecified 9  |  Bilateral involvement 66 (23.2%)',"#eef2f7",'#7b8794',8.2)
arrow(5.0,3.8,5.0,3.25)
box(BX,1.95,LW,1.2,'Three-timepoint longitudinal laboratory course\nT0 admission  ->  T1 VTE detection (median 12 d post-injury)\n->  T2, 2 wk after anticoagulation',
    '#fff6e5','#c99a34',8.2)
arrow(5.0,1.95,5.0,1.4)
box(BX,0.25,LW,1.05,'Median detection time 12 d (IQR 7–18)  |  IVC filter 57 (20.0%)\nBleeding after anticoagulation 11 (3.9%)  |  Thrombotic complications 9 (3.2%)',
    '#f2eefa','#8a72b8',8.0)
ax.set_title('Fig 1  Study cohort and DVT phenotype',fontsize=11,fontweight='bold',pad=6)
plt.savefig('../figures/Fig1_cohort_flow.png'); plt.close()
print('[saved] figures/Fig1_cohort_flow.png')
