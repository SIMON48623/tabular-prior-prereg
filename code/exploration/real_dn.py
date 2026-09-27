import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
P='/home/claude/p0b/phase0b_outputs/'
d=pd.read_csv(P+'datasets_stratifiers.csv',float_precision='round_trip'); d=d[d.excluded_reason.isna()].copy()
m=json.load(open(P+'marginal_aucs.json'))
def neff(l):
    l=np.asarray(l); s=(l**2).sum(); return l.sum()**2/s if s>0 else np.nan
out=[]
for _,r in d.iterrows():
    a=np.array(list(m[str(r.openml_id)].values()))
    n1=round(r.n*r.minority_rate); n0=r.n-n1; se=np.sqrt((n0+n1+1)/(12*n0*n1))
    rec={'openml_id':r.openml_id,'se0':se}
    for z in (0,1.0,1.645,1.96): rec[f'z{z}']=neff(np.maximum(a-0.5-z*se,0))
    rec['n_sig']=int((a-0.5>1.645*se).sum())
    out.append(rec)
o=pd.DataFrame(out); d=d.merge(o,on='openml_id')
t=d[(d.auc_lr>=0.6)&(d.auc_lr<=0.85)].copy()
print('target',len(t),'undefined z1.645:',t['z1.645'].isna().sum(),' n_sig==1:',(t.n_sig==1).sum(),' n_sig min',t.n_sig.min())
for z in ('z0','z1.0','z1.645','z1.96'):
    x=np.log(t[z])
    print(z, 'rho logp %.3f  logn %.3f  auc_lr %.3f  minority %.3f  | range %.2f-%.2f'%(spearmanr(x,np.log(t.p_used)).statistic,spearmanr(x,np.log(t.n)).statistic,spearmanr(x,t.auc_lr).statistic,spearmanr(x,t.minority_rate).statistic,x.min(),x.max()),
          ' corr with raw %.3f'%spearmanr(x,np.log(t.z0)).statistic)
x=np.log(t['z1.645']); q=np.quantile(x,[1/3,2/3],method='linear'); print('tertile cut (linear quantile)',q)
T=np.where(x<q[0],'T1',np.where(x<q[1],'T2','T3')); t['Tdn']=T
print(t.groupby('Tdn').agg(n=('name','size'),fam=('family','nunique')))
t['Traw']=np.where(np.log(t.N_eff)<1.7073919,'T1',np.where(np.log(t.N_eff)<2.2692851,'T2','T3'))
print(pd.crosstab(t.Traw,t.Tdn))
# residual SD of log N_dn given logp, logn and collinearity
X=np.column_stack([np.ones(len(t)),np.log(t.p_used),np.log(t.n)]); y=x.values
res=y-X@np.linalg.lstsq(X,y,rcond=None)[0]; print('SD',y.std().round(3),'resid SD | logp,logn',res.std().round(3),'R2',1-res.var()/y.var())
t[['openml_id','name','family','n','p_used','minority_rate','auc_lr','N_eff','se0','z1.645','n_sig']].to_csv('/tmp/claude-0/-home-claude/f3770dde-fb94-5ea9-b187-c34b6606f20b/scratchpad/target_dn.csv',index=False)
print(t.sort_values('z1.645')[['name','p_used','n','N_eff','z1.645','n_sig']].head(8).to_string())
