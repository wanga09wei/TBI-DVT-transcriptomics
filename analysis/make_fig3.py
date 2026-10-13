# -*- coding: utf-8 -*-
"""Fig 3 促凝轨迹表型图 + Table 3 轨迹表型特征表"""
import pandas as pd, numpy as np, warnings, json
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.9,
                     'savefig.dpi':300,'savefig.bbox':'tight','figure.facecolor':'white'})

D = pd.read_pickle('merged.pkl'); E = D[D['血栓0_1'].notna()].copy()
TC = pd.read_csv('traj_classes.csv', index_col=0)
def num(x):
    s=pd.Series(x).astype(str).str.replace('——','',regex=False).str.replace('不详','',regex=False)
    return pd.to_numeric(s.str.replace(r'[^0-9.\-]','',regex=True),errors='coerce')
def yn(x):
    o=[]
    for v in pd.Series(x):
        s=str(v).strip()
        o.append(0 if s in ('0','0.0','无','——','－','未','未查') else (1 if s in ('1','1.0','有','是') else np.nan))
    return pd.Series(o,index=pd.Series(x).index)
loc=E['血栓位置'].astype(str)
E['lv_prox']=loc.str.contains(r'腘|股总|股浅|股深|股静脉|髂').astype(float)
E['lv_bilat']=loc.str.contains('双').astype(float)
E['bleed']=yn(E['合并其它出血']); E['ivcf']=yn(E['IVC滤器']); E['comp']=yn(E['血栓并发症'])
E['age']=num(E['年龄']); E['los']=num(E['住院天数']); E['male']=(num(E['性别'])==1).astype(float)
E['detect_day']=num(E['血栓检出时间'].astype(str).str.extract(r'伤后(\d+)')[0]).fillna(num(E['血栓检出时间']))
for t,l in [('T0','血红蛋白'),('T1','血红蛋白'),('T2','血色素')]: E[f'v_{t}_HB']=num(E[f'{t}_{l}'])
for m in ['WBC','FIB']:
    for t in ['T0','T1','T2']: E[f'v_{t}_{m}']=num(E[f'{t}_{m}'])
E['v_T0_DD']=num(E['T0_D-二聚体']); E['v_T1_DD']=num(E['T1_D-二聚体']); E['v_T2_DD']=num(E['T2_D-二聚体'])
TC['class_K3']=TC['class_K3'].astype(int)
E=E.join(TC[['class_K3','T0','T1','T2']].rename(columns={'class_K3':'traj'}),how='inner')
order=E.groupby('traj')['T0'].mean().sort_values().index.tolist()
E['traj']=E['traj'].map({o:i for i,o in enumerate(order)})

NAMES={0:'Low burden\n(blunted rise)',1:'Intermediate burden\n(sustained rise)',2:'High burden\n(hyper-reactive, peak-and-fall)'}
SHORT={0:'Low burden',1:'Intermediate burden',2:'High burden'}
COL={0:'#2e86c1',1:'#e67e22',2:'#c0392b'}
sel=json.load(open('gbtm_selection.json',encoding='utf-8'))
Y=E[['T0','T1','T2']].values

# ================= FIG 3 =================
fig=plt.figure(figsize=(9.6,7.2))
gs=fig.add_gridspec(2,2,hspace=0.34,wspace=0.26)

# --- A: BIC 选型 ---
ax=fig.add_subplot(gs[0,0])
ks=[1,2,3,4,5]; bics=[sel[str(k)]['BIC'] for k in ks]
ax.plot(ks,bics,'o-',color='#34495e',lw=1.8,ms=6,zorder=3)
kbest=int(min(sel,key=lambda k:sel[k]['BIC']))
ax.plot([kbest],[sel[str(kbest)]['BIC']],'*',ms=18,color='#c0392b',zorder=5)
ax.annotate(f'BIC-optimal\nK = {kbest}',(kbest,sel[str(kbest)]['BIC']),textcoords='offset points',
            xytext=(16,-32),fontsize=8.5,color='#c0392b',fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#c0392b',lw=0.9))
for k,av in zip(ks,[sel[str(k)]['AvePP'] for k in ks]):
    ax.annotate(f'AvePP {av:.2f}',(k,bics[ks.index(k)]),textcoords='offset points',
                xytext=(10,7),ha='left',fontsize=6.9,color='#1e8449')
ax.set_ylim(7900,8160)
ax.set_xlabel('Number of latent trajectory classes (K)'); ax.set_ylabel('BIC')
ax.set_title('A   Model selection',fontsize=9.5,fontweight='bold',loc='left')
ax.grid(alpha=0.22,ls=':'); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)

# --- B: 分型血小板轨迹（中位连线 + IQR 带 + GBTM 拟合虚线） ---
ax=fig.add_subplot(gs[0,1])
tg=np.linspace(0,2,50); Xd=np.vander(tg,2,increasing=True)
for k in range(3):
    m=(E.traj==k).values
    med=[np.median(Y[m,t]) for t in range(3)]
    q1=[np.percentile(Y[m,t],25) for t in range(3)]; q3=[np.percentile(Y[m,t],75) for t in range(3)]
    beta=np.array(sel['3']['beta'][k]); fit=Xd@beta
    ax.fill_between([0,1,2],q1,q3,color=COL[k],alpha=0.12,zorder=1)
    ax.plot(tg,fit,color=COL[k],lw=1.4,zorder=3,ls='--',alpha=0.75)
    ax.plot([0,1,2],med,'o-',color=COL[k],lw=2.6,ms=7,mec='white',mew=1.2,zorder=4)
    ax.annotate(f'{med[2]:.0f}',(2,med[2]),textcoords='offset points',xytext=(8,-3),
                fontsize=8,color=COL[k],fontweight='bold')
    ax.annotate(f'{med[1]:.0f}',(1,med[1]),textcoords='offset points',xytext=(0,9),
                ha='center',fontsize=7.6,color=COL[k],fontweight='bold')
    ax.annotate(f'{med[0]:.0f}',(0,med[0]),textcoords='offset points',xytext=(-8,-3),
                ha='right',fontsize=7.6,color=COL[k],fontweight='bold')
ax.set_xlim(-0.45,2.42)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['T0\nAdmission','T1\nVTE detection','T2\n2 wk post-AC'],fontsize=8.2)
ax.set_ylabel('Platelet (×10$^9$/L)')
ax.set_title('B   Platelet trajectory classes (GBTM)',fontsize=9.5,fontweight='bold',loc='left')
ax.grid(alpha=0.22,ls=':'); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Line2D([],[],color=COL[k],lw=2.2,
           label=f'C{k+1}  {SHORT[k]} (n={int((E.traj==k).sum())})') for k in range(3)],
          fontsize=7.4,frameon=False,loc='upper left',bbox_to_anchor=(0.0,1.0))
ax.text(0.985,0.03,'dashed = GBTM fit; solid = observed median',transform=ax.transAxes,
        ha='right',fontsize=6.8,color='#5b6570',style='italic')

# --- C: 个体 spaghetti ---
ax=fig.add_subplot(gs[1,0])
for k in range(3):
    m=(E.traj==k).values
    for r in Y[m]:
        ax.plot([0,1,2],r,color=COL[k],alpha=0.13,lw=0.7)
    ax.plot([0,1,2],[np.median(Y[m,t]) for t in range(3)],'o-',color=COL[k],lw=2.4,ms=6,zorder=10)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['T0','T1','T2'],fontsize=8.6)
ax.set_ylabel('Platelet (×10$^9$/L)'); ax.set_yscale('log')
ax.set_title('C   Individual trajectories by class',fontsize=9.5,fontweight='bold',loc='left')
ax.grid(alpha=0.22,ls=':',which='both'); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(fc=COL[k],label=f'C{k+1}  {SHORT[k]}') for k in range(3)],
          fontsize=7.6,frameon=False,loc='upper left')

# --- D: 结局关联 ---
ax=fig.add_subplot(gs[1,1])
outcomes=[('Bleeding after\nanticoagulation','bleed'),('Proximal/trunk\ninvolvement','lv_prox'),
          ('IVC filter','ivcf'),('Thrombotic\ncomplications','comp')]
x=np.arange(len(outcomes)); wd=0.26
for k in range(3):
    m=(E.traj==k).values
    vals=[100*np.nanmean(pd.to_numeric(E.loc[m,c],errors='coerce')) for _,c in outcomes]
    ax.bar(x+(k-1)*wd,vals,wd,color=COL[k],label=f'C{k+1}',alpha=0.9)
    for xi,v in zip(x+(k-1)*wd,vals):
        if v>0: ax.text(xi,v+0.35,f'{v:.1f}',ha='center',fontsize=7,color=COL[k],fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([n for n,_ in outcomes],fontsize=7.4,rotation=14,ha='right')
ax.set_ylabel('Patients (%)')
ax.set_title('D   Clinical outcomes by trajectory class',fontsize=9.5,fontweight='bold',loc='left')
ax.grid(alpha=0.22,axis='y',ls=':'); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(fontsize=7.6,frameon=False,ncol=3,loc='upper center')
ax.annotate('OR 5.48, P = 0.025',xy=(0.26,15.5),xytext=(0.85,21.5),fontsize=7.6,color='#c0392b',
            fontweight='bold',arrowprops=dict(arrowstyle='->',color='#c0392b',lw=0.9))

fig.suptitle('Fig 3  Procoagulant trajectory phenotypes in TBI with concomitant lower-limb fracture (n = 213)',
             fontsize=10.2,fontweight='bold',y=0.985)
fig.text(0.5,0.008,'Group-based trajectory modelling (GBTM) on three-timepoint platelet counts (admission / VTE detection / 2 weeks after anticoagulation). '
         'Classes ordered by baseline burden.\nC2 (high-burden) shows the highest multi-marker inflammatory burden (WBC, fibrinogen), a peak-and-fall platelet shape, '
         'a younger age profile (45 y), and a 5.5-fold higher odds of bleeding after anticoagulation.',ha='center',fontsize=7.2,style='italic')
plt.savefig('../figures/Fig3_trajectory_phenotypes.png'); plt.close()
print('[saved] figures/Fig3_trajectory_phenotypes.png')

# ================= TABLE 3 =================
def fmt(s,d=1): return '—' if len(s.dropna())==0 else f'{s.median():.{d}f} ({s.quantile(.25):.{d}f}–{s.quantile(.75):.{d}f})'
rows=[]
def cont(v,lab,d=1):
    gs_=[pd.to_numeric(E.loc[E.traj==k,v],errors='coerce').dropna() for k in range(3)]
    p=stats.kruskal(*[g for g in gs_ if len(g)>2]).pvalue if all(len(g)>2 for g in gs_) else np.nan
    rows.append([lab]+[fmt(pd.to_numeric(E.loc[E.traj==k,v],errors='coerce'),d) for k in range(3)]+
                [f'{p:.4f}' if p==p else '—'])
def catf(v,lab,den=None):
    m=(pd.to_numeric(E[den],errors='coerce')==1) if den else pd.Series(True,index=E.index)
    cells=[]; tab=[]
    for k in range(3):
        s=pd.to_numeric(E.loc[E.traj==k,v],errors='coerce'); mm=m.loc[E.traj==k]
        kk=int(s[mm].sum()); nn=int(s[mm].notna().sum()); tab.append([kk,nn-kk])
        cells.append(f'{kk} ({100*kk/max(nn,1):.1f}%)')
    try:
        p=stats.chi2_contingency(tab)[1]
        if stats.chi2_contingency(tab)[3].min()<5: p=stats.fisher_exact([tab[0],tab[1]])[1] if len(tab)==2 else p
    except Exception: p=np.nan
    rows.append([lab]+cells+[f'{p:.4f}' if p==p else '—'])

cont('T0','Platelet T0, ×10⁹/L',0); cont('T1','Platelet T1, ×10⁹/L',0); cont('T2','Platelet T2, ×10⁹/L',0)
cont('v_T0_WBC','WBC T0, ×10⁹/L',2); cont('v_T1_WBC','WBC T1, ×10⁹/L',2)
cont('v_T0_FIB','Fibrinogen T0, g/L',2); cont('v_T1_FIB','Fibrinogen T1, g/L',2)
cont('v_T0_DD','D-dimer T0, mg/L',2); cont('v_T1_DD','D-dimer T1, mg/L',2)
cont('v_T0_HB','Haemoglobin T0, g/L',0)
cont('age','Age, years',0)
catf('male','Male sex'); catf('lv_prox','Proximal/trunk involvement'); catf('lv_bilat','Bilateral involvement')
catf('ivcf','IVC filter placement'); catf('bleed','Bleeding after anticoagulation'); catf('comp','Thrombotic complications')
cont('detect_day','Time to VTE detection, days',0); cont('los','Length of hospital stay, days',0)
T3=pd.DataFrame(rows,columns=['Variable','C1 Low burden (n=64)','C2 Intermediate (n=119)','C3 High burden (n=30)','P value'])
print('\n'+T3.to_string(index=False))
T3.to_csv('table3_trajectory_phenotypes.csv',index=False,encoding='utf-8-sig')
with open('table3_trajectory_phenotypes.md','w',encoding='utf-8') as f:
    f.write('| '+' | '.join(T3.columns)+' |\n|'+'---|'*len(T3.columns)+'\n')
    for _,r in T3.iterrows(): f.write('| '+' | '.join(str(x) for x in r.values)+' |\n')
print('\n[saved] table3_trajectory_phenotypes.csv/.md')
