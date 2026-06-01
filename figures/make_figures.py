"""Generate all paper figures from verified results. Pure reproduction of numbers
already established in Steps 1-6, multi-seed, baselines, and the limitation study."""
import os
os.environ["TORCHDYNAMO_DISABLE"]="1"; os.environ["TORCHINDUCTOR_DISABLE"]="1"
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,
                     "figure.dpi":150,"savefig.dpi":200,"savefig.bbox":"tight"})
FIG="paper/figs"; os.makedirs(FIG,exist_ok=True)
BLUE="#2c6fbb"; RED="#c0392b"; GREEN="#27914f"; ORANGE="#e08a1e"; GREY="#555"

# ---------- Fig 1: schematic of the physics<->ML mapping (drawn) ----------
fig,ax=plt.subplots(figsize=(7.0,3.4)); ax.axis("off")
th=np.linspace(0,2*np.pi,200)
for R,c,lab in [(1.0,GREY,None),(1.7,BLUE,None)]:
    ax.plot(0.0+R*np.cos(th),R*np.sin(th),color=c,lw=2)
ax.fill(0.9*np.cos(th),0.9*np.sin(th),color=BLUE,alpha=0.12)
ax.annotate("",xy=(1.95,0),xytext=(1.05,0),arrowprops=dict(arrowstyle="-|>",color=RED,lw=2))
ax.text(1.5,0.16,r"$V_n$",color=RED,fontsize=13)
ax.text(0,0,"consolidated\n(solid, $\\phi<0$)",ha="center",va="center",fontsize=9)
ax.text(0,2.05,"plastic (liquid, $\\phi>0$)",ha="center",fontsize=9,color=BLUE)
ax.set_xlim(-2.4,2.6); ax.set_ylim(-2.2,2.4); ax.set_aspect("equal")
plt.savefig(f"{FIG}/fig1_schematic.pdf"); plt.close()

# ---------- Fig 2: accuracy matrices naive vs Stefan-CL (verified Step 3/5) ----------
A_naive=np.array([
 [0.993,0.743,0.493,0.247,0.005],
 [0.787,0.993,0.749,0.504,0.256],
 [0.609,0.782,0.993,0.756,0.499],
 [0.429,0.555,0.767,0.992,0.751],
 [0.423,0.313,0.487,0.749,0.997]])
A_stefan=np.array([
 [0.993,0.743,0.496,0.247,0.005],
 [0.944,0.906,0.686,0.450,0.204],
 [0.948,0.925,0.893,0.723,0.499],
 [0.936,0.909,0.914,0.921,0.763],
 [0.938,0.917,0.900,0.904,0.963]])
fig,axs=plt.subplots(1,2,figsize=(8.2,3.6))
for ax,Amat,ttl in [(axs[0],A_naive,"Naive sequential"),(axs[1],A_stefan,"Stefan-CL")]:
    im=ax.imshow(Amat,cmap="viridis",vmin=0,vmax=1,aspect="equal")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"T{j+1}" for j in range(5)]); ax.set_yticklabels([f"after T{i+1}" for i in range(5)])
    ax.set_title(ttl); ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f"{Amat[i,j]:.2f}",ha="center",va="center",
                    color="white" if Amat[i,j]<0.6 else "black",fontsize=7)
fig.colorbar(im,ax=axs,fraction=0.025,pad=0.02,label="test accuracy")
plt.savefig(f"{FIG}/fig2_accmatrix.pdf"); plt.close()

# ---------- Fig 3: frontier tracking vs Frank-sphere law (verified Step 5) ----------
k=np.arange(1,6); law=1.0*np.sqrt(k)
disc=np.array([1.052,1.448,1.755,2.028,2.243])      # Step5 verified
disc_err=np.array([0.008]*5)
fig,ax=plt.subplots(figsize=(5.0,3.6))
kk=np.linspace(1,5,100)
ax.plot(kk,1.0*np.sqrt(kk),color=GREY,lw=2,label=r"Frank-sphere law $R_k=R_0\sqrt{k}$")
ax.errorbar(k,disc,yerr=disc_err,fmt="o",color=RED,capsize=3,label="discovered frontier (data-driven)")
ax.set_xlabel("task $k$"); ax.set_ylabel("frontier radius $R_k$"); ax.legend(fontsize=9)
ax.set_title("Self-advected frontier recovers the growth law")
plt.savefig(f"{FIG}/fig3_frontier.pdf"); plt.close()

# ---------- Fig 4: latent-heat stability-plasticity dial (verified multi-seed) ----------
L=np.array([0.5,1.0,2.0,4.0,8.0])
pf=np.array([1.013,1.003,0.997,0.742,0.533]); pf_s=np.array([0.001,0.0,0.0,0.0,0.0])
fg=np.array([0.019,0.020,0.020,0.086,0.206]); fg_s=np.array([0.003,0.003,0.004,0.005,0.006])
pl=np.array([0.934,0.935,0.939,0.982,0.988]); pl_s=np.array([0.003,0.004,0.004,0.001,0.001])
fig,ax=plt.subplots(figsize=(5.6,3.8))
ax.errorbar(L,fg,yerr=fg_s,fmt="o-",color=RED,capsize=3,label="forgetting")
ax.errorbar(L,1-pl,yerr=pl_s,fmt="s-",color=BLUE,capsize=3,label="rigidity (1$-$plasticity)")
ax.errorbar(L,pf,yerr=pf_s,fmt="^--",color=GREEN,capsize=3,label="protected fraction")
ax.set_xscale("log"); ax.set_xlabel("latent heat $L$ (consolidation cost)")
ax.set_ylabel("metric"); ax.legend(fontsize=9,loc="center left")
ax.set_title("Latent heat as a stability--plasticity dial")
plt.savefig(f"{FIG}/fig4_latentheat.pdf"); plt.close()

# ---------- Fig 5: baseline comparison (verified) ----------
methods=["Naive","SI","EWC","Stefan-CL","Replay"]
avg=np.array([0.516,0.701,0.716,0.923,0.940]); avg_s=np.array([0.007,0.022,0.027,0.004,0.004])
forget=np.array([0.600,0.241,0.287,0.021,0.056]); forget_s=np.array([0.008,0.051,0.045,0.003,0.006])
cols=[GREY,ORANGE,ORANGE,RED,BLUE]
fig,axs=plt.subplots(1,2,figsize=(8.4,3.5))
x=np.arange(len(methods))
axs[0].bar(x,avg,yerr=avg_s,capsize=3,color=cols); axs[0].set_xticks(x); axs[0].set_xticklabels(methods,rotation=20)
axs[0].set_ylabel("avg accuracy"); axs[0].set_ylim(0,1); axs[0].set_title("Average accuracy (higher better)")
axs[1].bar(x,forget,yerr=forget_s,capsize=3,color=cols); axs[1].set_xticks(x); axs[1].set_xticklabels(methods,rotation=20)
axs[1].set_ylabel("forgetting"); axs[1].set_title("Forgetting (lower better)")
plt.savefig(f"{FIG}/fig5_baselines.pdf"); plt.close()

# ---------- Fig 6: non-circular limitation (verified Result A/B) ----------
fig,axs=plt.subplots(1,2,figsize=(8.4,3.5))
rho=[0.6,0.9,1.3,1.6]; sa_repr=[0.989,0.989,0.979,0.991]
axs[0].plot(rho,sa_repr,"o-",color=GREEN,label="region sign accuracy")
axs[0].axvline(1.2,color=GREY,ls=":",label="merge ($\\rho=1.2$)")
axs[0].set_ylim(0.9,1.01); axs[0].set_xlabel(r"union radius $\rho$"); axs[0].set_ylabel("accuracy")
axs[0].set_title("Result A: field REPRESENTS the merge"); axs[0].legend(fontsize=8)
stage_rho=[0.7,1.0,1.4,1.6]; area_pred=[0,0,0,0]; area_true=[0.097,0.198,0.375,0.469]
x=np.arange(4); w=0.38
axs[1].bar(x-w/2,area_true,w,color=GREEN,label="true region area")
axs[1].bar(x+w/2,area_pred,w,color=RED,label="advected (eroded)")
axs[1].set_xticks(x); axs[1].set_xticklabels([f"$\\rho$={r}" for r in stage_rho])
axs[1].set_ylabel("region area fraction"); axs[1].set_title("Result B: advection FAILS to track")
axs[1].legend(fontsize=8)
plt.savefig(f"{FIG}/fig6_limitation.pdf"); plt.close()

print("figures written:")
for f in sorted(os.listdir(FIG)): print("  ",f)
