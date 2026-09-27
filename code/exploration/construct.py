import sys, json; sys.path.insert(0,'/home/claude/tabprior_v2/code')
import numpy as np, generators as G, phase0c_functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
cal=json.loads(G.CALIB_FILE.read_text()); rows=[]
for gen in G.SYN_GEN:
  for rho in G.SYN_RHO:
    for k in (1,4,16):
      for n in (300,3000):
        X,y,_,inf=G.synth_draw(gen,k,rho,0.8,n,0,'train',cal); names=[f'f{j}' for j in range(32)]
        marg={}; per=[]
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
          sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]),y[tr])
          per.append(F.feature_contributions(lr.coef_[0],sc.transform(X[te]),names))
        # simple marginal: direction from full data (approximation of the nested Phase 0b rule)
        m={nm:max(roc_auc_score(y,X[:,j]),1-roc_auc_score(y,X[:,j])) for j,nm in enumerate(names)}
        rows.append((gen,rho,k,n,G.n_eff(m.values()),F.neff_conditional(per,names)))
        print(f"{gen:9s} rho={rho} k={k:2d} n={n:4d}  N_eff marginal {rows[-1][4]:5.1f}   conditional {rows[-1][5]:5.1f}")
