import sys, json; sys.path.insert(0,'/home/claude/tabprior_v2/code')
import numpy as np, generators as G
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
cal=json.loads(G.CALIB_FILE.read_text())
def nested_marg(X,y):
    # emulate Phase 0b: direction on training fold, held-out scores -> within-fold percentiles, pooled AUROC
    n,p=X.shape; S=np.zeros((n,p))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
        for j in range(p):
            a=roc_auc_score(y[tr],X[tr,j]); s=X[te,j]*(1 if a>=0.5 else -1)
            S[te,j]=(rankdata(s)-0.5)/len(te)
    return np.array([roc_auc_score(y,S[:,j]) for j in range(p)])
def se0(y):
    n1=y.sum(); n0=len(y)-n1; return np.sqrt((n0+n1+1)/(12*n0*n1))
def neff(l):
    l=np.asarray(l); return l.sum()**2/(l**2).sum() if (l**2).sum()>0 else np.nan
def est(a,y):
    s=se0(y); d=a-0.5; p=len(a)
    raw=neff(np.maximum(d,0))
    soft1=neff(np.maximum(d-1.0*s,0)); soft2=neff(np.maximum(d-1.96*s,0))
    S1=d.sum(); S2=(d**2-s**2).sum(); num=S1**2-p*s**2
    bc=num/S2 if S2>0 and num>0 else np.nan
    return raw,soft1,soft2,bc
res={}
for auc in (0.7,0.8):
  for k in G.SYN_K:
    for n in (300,1000,3000):
      vals=[]
      for rep in range(4):
        X,y,_,_=G.synth_draw('linear',k,0.0,auc,n,rep,'train',cal)
        vals.append(est(nested_marg(X,y),y))
      v=np.nanmean(np.log(np.array(vals,float)),axis=0)
      res[(auc,k,n)]=v
      print(f"auc={auc} k={k:2d} n={n:4d}  log k={np.log(k):.2f} | raw {v[0]:.2f} soft1 {v[1]:.2f} soft1.96 {v[2]:.2f} bc {v[3]:.2f}",flush=True)
E=np.array([[res[key][i]-np.log(key[1]) for key in res] for i in range(4)])
for i,nm in enumerate(['raw','soft1','soft1.96','bc']): print(nm,'mean abs log error',np.nanmean(np.abs(E[i])).round(3),'mean bias',np.nanmean(E[i]).round(3))
