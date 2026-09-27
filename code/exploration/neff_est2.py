import sys, json; sys.path.insert(0,'/home/claude/tabprior_v2/code'); sys.path.insert(0,'/tmp/claude-0/-home-claude/f3770dde-fb94-5ea9-b187-c34b6606f20b/scratchpad')
import numpy as np, generators as G, pandas as pd
from scipy.stats import spearmanr
exec(open('/tmp/claude-0/-home-claude/f3770dde-fb94-5ea9-b187-c34b6606f20b/scratchpad/neff_est.py').read().split('res={}')[0])
def variants(a,y):
    s=se0(y); d=a-0.5
    out={'raw':neff(np.maximum(d,0))}
    for z in (1.0,1.645,1.96,2.576): out[f'soft{z}']=neff(np.maximum(d-z*s,0))
    for z in (1.96,): out[f'hard{z}']=neff(np.where(d>z*s,d,0))
    return out
rows=[]
for p_keep in (32,12):
  for auc in (0.7,0.8):
    for k in G.SYN_K:
      if k>p_keep: continue
      for n in (300,1000,3000):
        for rep in range(4):
          X,y,_,inf=G.synth_draw('linear',k,0.0,auc,n,rep,'train',cal)
          noise=[j for j in range(32) if j not in inf][:p_keep-k]
          cols=sorted(inf+noise)
          v=variants(nested_marg(X[:,cols],y),y)
          rows.append(dict(p=p_keep,auc=auc,k=k,n=n,rep=rep,**{kk:np.log(vv) if vv==vv and vv>0 else np.nan for kk,vv in v.items()}))
df=pd.DataFrame(rows); df['logk']=np.log(df.k)
V=[c for c in df.columns if c.startswith(('raw','soft','hard'))]
print('mean abs error by p'); print(df.groupby('p')[V].apply(lambda g: (g.sub(df.loc[g.index,'logk'],axis=0)).abs().mean()).round(3))
print('share undefined'); print(df[V].isna().mean().round(3))
print('Spearman with log k within (p, n):')
print(df.groupby(['p','n']).apply(lambda g: pd.Series({v: spearmanr(g[v],g.logk,nan_policy='omit').statistic for v in V})).round(3))
# sensitivity to p at fixed k: difference in estimate between p=32 and p=12 for same k,n
m=df.groupby(['p','k','n'])[V].mean(); d=(m.xs(32)-m.xs(12)).dropna(); print('mean inflation p32 vs p12'); print(d.mean().round(3))
