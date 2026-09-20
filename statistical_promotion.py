"""Conservative two-stream block-bootstrap promotion test."""
from __future__ import annotations
import numpy as np, pandas as pd

def clean(records):
    rows=[]
    for r in records:
        if r.get("status") not in {"TP","SL","EXPIRED"}: continue
        try:
            t=float(r["open_time"]); x=float(r["net_r"])
            if np.isfinite(t) and np.isfinite(x): rows.append((t,x))
        except Exception: pass
    return pd.DataFrame(rows,columns=["open_time","net_r"])

def block_bootstrap_mean(df, block_ms=86_400_000, rounds=10_000, seed=7):
    if df.empty:return np.array([])
    d=df.sort_values("open_time").copy(); keys=(d.open_time//block_ms).astype("int64")
    blocks=[g.net_r.to_numpy() for _,g in d.groupby(keys,sort=True)]
    rng=np.random.default_rng(seed); out=np.empty(rounds)
    for i in range(rounds): out[i]=np.mean(np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))]))
    return out

def compare(candidate, production, min_resolved=50, rounds=10_000, alpha=.05):
    c,p=clean(candidate),clean(production)
    if len(c)<min_resolved or len(p)<min_resolved:
        return {"promotion_ready":False,"reason":"INSUFFICIENT_RESOLVED_SAMPLE","candidate_n":len(c),"production_n":len(p)}
    cb=block_bootstrap_mean(c,rounds=rounds); pb=block_bootstrap_mean(p,rounds=rounds,seed=17); db=cb-pb
    q=100*alpha/2
    cci=np.percentile(cb,[q,100-q]); pci=np.percentile(pb,[q,100-q]); dci=np.percentile(db,[q,100-q])
    ready=bool(cci[0]>0 and dci[0]>0)
    return {"promotion_ready":ready,"reason":"PASS" if ready else "CI_HURDLE_FAILED",
            "candidate_n":len(c),"production_n":len(p),"candidate_mean":float(c.net_r.mean()),
            "production_mean":float(p.net_r.mean()),"candidate_ci95":cci.tolist(),
            "production_ci95":pci.tolist(),"difference_ci95":dci.tolist(),"bootstrap_rounds":rounds}
