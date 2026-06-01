import os
os.environ["TORCHDYNAMO_DISABLE"]="1"; os.environ["TORCHINDUCTOR_DISABLE"]="1"
import copy, time, numpy as np, torch, torch.nn as nn
exec(open('baselines_dev.py').read().split('SEEDS=[0,1]')[0])  # reuse defs above the run block

SEEDS=[0,1,2]; EP=150
t0=time.time()
print("EWC lambda sweep (avg, forget) mean over seeds:")
for lam in [1.0,10.0,50.0,100.0,300.0,1000.0,3000.0]:
    r=np.array([run_ewc(s,EP,lam) for s in SEEDS]).mean(0)
    print(f"  lam={lam:7.1f}  avg={r[0]:.3f}  forget={r[1]:.3f}")
print("SI lambda sweep:")
for lam in [0.1,0.5,1.0,5.0,10.0,50.0]:
    r=np.array([run_si(s,EP,lam) for s in SEEDS]).mean(0)
    print(f"  lam={lam:7.1f}  avg={r[0]:.3f}  forget={r[1]:.3f}")
print("Replay buffer sweep:")
for b in [50,100,200,400]:
    r=np.array([run_replay(s,EP,b) for s in SEEDS]).mean(0)
    print(f"  buf/task={b:4d}  avg={r[0]:.3f}  forget={r[1]:.3f}")
print(f"[{time.time()-t0:.0f}s]")
