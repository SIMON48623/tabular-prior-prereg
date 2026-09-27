# run at 2026-09-26T14:09:49.162Z (UTC); command recovered verbatim from the session log
cd /tmp/claude-0/-home-claude/f3770dde-fb94-5ea9-b187-c34b6606f20b/scratchpad && python3 - <<'EOF'
import numpy as np, pandas as pd
from scipy.stats import spearmanr
t=pd.read_csv('target_dn.csv')
print(t[t['z1.645'].isna()].to_string())
t=t.dropna(subset=['z1.645']).copy()
P='/home/claude/p0b/phase0b_outputs/'
import json
m=json.load(open(P+'marginal_aucs.json'))
def neff(l):
    l=np.asarray(l); s=(l**2).sum(); return l.sum()**2/s if s>0 else np.nan
for z in (1.0,1.645,1.96):
    x=np.log([neff(np.maximum(np.array(list(m[str(i)].values()))-0.5-z*se,0)) for i,se in zip(t.openml_id,t.se0)])
    ok=np.isfinite(x)
    print(z,'rho logp %.3f logn %.3f auc_lr %.3f minority %.3f raw %.3f'%tuple(spearmanr(x[ok],v[ok]).statistic for v in [np.log(t.p_used.values),np.log(t.n.values),t.auc_lr.values,t.minority_rate.values,np.log(t.N_eff.values)]))
x=np.log(t['z1.645'].values)
X=np.column_stack([np.ones(len(t)),np.log(t.p_used),np.log(t.n)]); res=x-X@np.linalg.lstsq(X,x,rcond=None)[0]
print('SD %.3f resid SD|logp,logn %.3f'%(x.std(),res.std()))
X2=np.column_stack([X,t.auc_lr]); res2=x-X2@np.linalg.lstsq(X2,x,rcond=None)[0]; print('resid SD|logp,logn,auc_lr %.3f'%res2.std())
q=np.quantile(x,[1/3,2/3]); print('cuts',q, np.exp(q))
T=np.where(x<q[0],'T1',np.where(x<q[1],'T2','T3')); t['T']=T
print(t.groupby('T').agg(n=('name','size'),fam=('family','nunique')))
for T_,g in t.groupby('T'): print(T_, g.family.value_counts().head(6).to_dict())
EOF
