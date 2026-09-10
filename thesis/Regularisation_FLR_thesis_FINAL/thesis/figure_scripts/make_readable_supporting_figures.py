#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import io
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import LogLocator, MaxNLocator, PercentFormatter
import numpy as np

WIDTH = 6.14
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MODEL_NAMES = ["Model 1", "Model 2", "Model 3"]
INK = "#252525"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2,
        "axes.labelsize": 9.2, "axes.titlesize": 10,
        "xtick.labelsize": 8.8, "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlepad": 7,
        "axes.linewidth": 0.65, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, name, args):
    # Preserve the physical canvas: bbox_inches='tight' can enlarge it and
    # shrink every font when LaTeX subsequently fits it to the text width.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf")
    data = buffer.getvalue()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise OSError(f"Incomplete PDF render: {name}")
    destination = args.output_dir / f"{name}.pdf"
    descriptor, temp_name = tempfile.mkstemp(dir=args.output_dir, suffix=".pdf")
    try:
        with os.fdopen(descriptor,"wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if Path(temp_name).read_bytes() != data:
            raise OSError(f"Incomplete PDF write: {name}")
        os.replace(temp_name,destination)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    if args.preview_dir:
        fig.savefig(args.preview_dir / f"{name}.png", dpi=170)
    plt.close(fig)


def grid(ax, log=False):
    ax.grid(axis="y", which="major", color="#e3e3e3", lw=.55)
    ax.set_axisbelow(True)
    if log:
        ax.yaxis.set_major_locator(LogLocator(numticks=5))
        ax.tick_params(which="minor", labelleft=False)


def make_design(core, args):
    s = np.linspace(0, 1, 200)
    basis = core.create_cosine_basis(s, 100)
    models = core.make_model_specs(100, basis)
    j = np.arange(1, 101)
    coefficients = [4 / j**2.7, np.r_[np.full(5, 4.), 4 / j[5:]**2.7], 4 / j**2.7]

    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.35))
    for i, (model, b) in enumerate(zip(models, coefficients)):
        left, right = axes[i]
        left.plot(s, model.beta, color=COLORS[i])
        left.axhline(0, color=".6", lw=.5)
        left.set(title=f"{model.name}: true slope", xlabel=r"Location $s$", ylabel=r"$\beta(s)$")
        left.set_ylim(-16, 29)
        left.set_xticks([0, .5, 1])
        right.semilogy(j[:20], np.abs(b[:20]), color=COLORS[i], marker="o", ms=2.6)
        right.set(title="Generating coefficients", xlabel=r"Component $j$", ylabel=r"$|b_j|$", xlim=(.5,20.5))
        right.set_xticks([1,5,10,15,20])
        grid(right, True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.07, top=.96, hspace=.82, wspace=.40)
    save(fig, "true_slope_structure", args)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.30))
    for k, label, marker in [(0,"Models 1 and 2","o"),(2,"Model 3","s")]:
        lam = models[k].eigenvalues
        axes[0].semilogy(j[:30], lam[:30], color=COLORS[k], marker=marker, ms=2.5, label=label)
        axes[1].plot(j, np.cumsum(lam)/lam.sum(), color=COLORS[k], marker=marker, markevery=10, ms=3, label=label)
    axes[0].set(xlabel=r"Component $j$", ylabel=r"Score variance $\lambda_j$", title="Score-variance decay")
    axes[1].set(xlabel=r"Components $m$", ylabel="Cumulative variance", title="Variance explained", ylim=(0,1.03), xlim=(1,100))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for ax in axes:
        ax.axvline(5.5, color=".5", ls=":", lw=1)
        grid(ax)
    axes[0].legend(loc="upper right", handlelength=1.4)
    fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.88,wspace=.48)
    save(fig,"covariance_spectrum",args)

    fig, axes = plt.subplots(1,2,figsize=(WIDTH,3.55))
    for i, model in enumerate(models):
        q = model.eigenvalues * (basis @ model.beta / 200)**2
        cumulative = np.cumsum(q)/q.sum()
        axes[0].semilogy(j[:20], q[:20], color=COLORS[i], marker=["o","s","^"][i], ms=3, label=model.name)
        axes[1].plot(j[:20], cumulative[:20], color=COLORS[i], marker=["o","s","^"][i], markevery=3,ms=3)
        m99 = int(np.searchsorted(cumulative,.99)+1)
        axes[1].scatter(m99,cumulative[m99-1],s=30,facecolor="white",edgecolor=COLORS[i],zorder=5)
    axes[0].set(xlabel=r"Component $j$",ylabel=r"Signal contribution $q_j$",title="Componentwise signal",xlim=(.5,20.5))
    axes[0].legend(loc="upper right")
    grid(axes[0],True)
    axes[1].set(xlabel=r"Components $m$",ylabel="Cumulative signal",title="Signal captured",xlim=(.5,20.5),ylim=(0,1.08))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    for share, ls in [(.90,"--"),(.95,"-."),(.99,":")]:
        axes[1].axhline(share,color=".55",ls=ls,lw=.7,zorder=0)
    axes[1].text(.08,.16,"Components for 90 / 95 / 99%\nModel 1: 1 / 1 / 2\nModel 2: 4 / 5 / 5\nModel 3: 1 / 1 / 2",transform=axes[1].transAxes,fontsize=8.5,linespacing=1.5)
    for ax in axes: ax.set_xticks([1,5,10,15,20])
    fig.subplots_adjust(left=.11,right=.98,bottom=.18,top=.89,wspace=.49)
    save(fig,"predictive_signal_allocation",args)

    # Reconstruct precisely the original illustrative draws, not new Monte Carlo data.
    scores=np.random.default_rng(2026).normal(size=(10,100))
    fig,axes=plt.subplots(2,1,figsize=(WIDTH,4.50),sharex=True,sharey=True)
    max_abs=0
    for ax,k,label in zip(axes,[0,2],["Models 1 and 2","Model 3"]):
        lam=models[k].eigenvalues
        curves=(scores*np.sqrt(lam))@basis
        band=1.6448536269514722*np.sqrt(np.sum(lam[:,None]*basis**2,axis=0))
        max_abs=max(max_abs,float(np.abs(curves).max()),float(band.max()))
        ax.fill_between(s,-band,band,color=COLORS[k],alpha=.16,lw=0)
        ax.plot(s,curves.T,color=COLORS[k],alpha=.36,lw=.7)
        ax.axhline(0,color=".5",lw=.6)
        ax.set(title=label,ylabel=r"$X(s)$")
    axes[0].set_ylim(-1.06*max_abs,1.06*max_abs)
    axes[1].set_xlabel(r"Location $s$")
    fig.legend([Patch(color=".7",alpha=.4),Line2D([],[],color=".4")],["Exact pointwise 90% band","Ten example curves"],loc="upper center",ncol=2,bbox_to_anchor=(.55,1.0),frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.82,hspace=.37)
    save(fig,"functional_predictor_examples",args)
    return s,basis,models


def make_duality(args):
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,5.25))
    theta=np.linspace(0,2*np.pi,900)
    diamond=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]])
    square=np.array([[1,1],[-1,1],[-1,-1],[1,-1],[1,1]])
    for ax in axes:
        ax.set(aspect="equal",xlim=(-1.2,1.2),ylim=(-1.2,1.2),xlabel=r"$x_1$",ylabel=r"$x_2$")
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1])
        ax.axhline(0,color=".7",lw=.6);ax.axvline(0,color=".7",lw=.6)
        ax.annotate("",xy=(.8,.6),xytext=(0,0),arrowprops={"arrowstyle":"-|>","color":INK,"lw":1.6})
        ax.scatter(.8,.6,s=20,color=INK,zorder=5)
    axes[0].plot(diamond[:,0],diamond[:,1],color=COLORS[0],label=r"$B_1^2$")
    axes[0].plot(np.cos(theta),np.sin(theta),color=COLORS[1],label=r"$B_2^2$")
    axes[0].plot(square[:,0],square[:,1],color=COLORS[2],label=r"$B_\infty^2$")
    axes[0].set_title("(a) Unit balls and distance")
    axes[0].text(.15,.32,r"$h$",fontsize=10)
    axes[0].legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=3,handlelength=1.1,columnspacing=.8,frameon=False)
    axes[1].fill(np.cos(theta),np.sin(theta),color=COLORS[1],alpha=.10)
    axes[1].plot(np.cos(theta),np.sin(theta),color=COLORS[1])
    xx=np.linspace(-.2,1.2,300);yy=(1-.8*xx)/.6
    axes[1].plot(xx,yy,color=COLORS[0],ls="--")
    axes[1].text(.08,.35,r"$u$",fontsize=10)
    axes[1].text(-.91,-.60,r"$u^\mathsf{T}x\leq1$",color=COLORS[0])
    axes[1].set_title("(b) Duality; self-duality at 2")
    axes[1].text(.5,-.23,r"$(B_1^2)^\circ=B_\infty^2,\quad(B_2^2)^\circ=B_2^2$",transform=axes[1].transAxes,ha="center",fontsize=9.3)
    fig.subplots_adjust(left=.09,right=.985,bottom=.45,top=.95,wspace=.35)
    fig.text(.5,.285,r"$h=u=(4/5,3/5):\quad\|h\|_1=7/5,\quad\|h\|_2=1,\quad\|h\|_\infty=4/5$",ha="center")
    fig.text(.5,.215,r"$p^{-1}+q^{-1}=1:\quad\|y\|_q=\sup_{\|x\|_p\leq1}|y^\mathsf{T}x|$",ha="center",fontsize=10)
    fig.text(.5,.135,r"$f=\sum_j f_j e_j\ \longleftrightarrow\ (f_j)\in\ell^2,\qquad\|f-g\|_{L^2}^2=\sum_j|f_j-g_j|^2$",ha="center",fontsize=10)
    fig.text(.5,.065,r"$\Lambda_\beta(f)=\langle f,\beta\rangle_{L^2}$",ha="center",fontsize=10)
    save(fig,"lp_duality_geometry",args)


def make_geometry(basis,models,args):
    module=importlib.import_module("make_krylov_geometry_figure")
    geom=module.compute_geometry(basis,models[:1],12,.20,1e-10)[0]
    recorded=read_csv(args.code_root/"four_method_study_FINAL_R5000/krylov_geometry/krylov_geometry_summary.csv")[0]
    assert geom.raw_edge_count==int(recorded["raw_edges_shown"])==66
    assert geom.arnoldi_edge_count==int(recorded["arnoldi_edges_shown"])==0
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,3.10),gridspec_kw={"width_ratios":[1,1,1]})
    axes[0].imshow(module._composite_dependence_image(geom),origin="upper",interpolation="nearest")
    axes[0].set_xticks([0,3,7,11],[1,4,8,12]);axes[0].set_yticks([0,3,7,11],[1,4,8,12])
    axes[0].set(xlabel="Vector index",ylabel="Vector index",title="Pairwise dependence")
    axes[0].text(.98,.97,"Arnoldi",transform=axes[0].transAxes,ha="right",va="top",color=COLORS[0],fontsize=8.8)
    axes[0].text(.04,.05,"Raw",transform=axes[0].transAxes,color="white",fontsize=8.8)
    positions=module._network_positions(12)
    for ax,dep,color,name in zip(axes[1:],[geom.raw_dependence,geom.arnoldi_dependence],[COLORS[1],COLORS[0]],["Raw Krylov basis","CGS2-Arnoldi basis"]):
        for a in range(12):
            for b in range(a+1,12):
                if dep[a,b]>=.20:
                    ax.plot(positions[[a,b],0],positions[[a,b],1],color=color,alpha=.4,lw=.65)
        for i,(x,y) in enumerate(positions):
            ax.add_patch(Circle((x,y),.123,facecolor="white",edgecolor=color,lw=1,zorder=4))
            ax.text(x,y,str(i+1),ha="center",va="center",fontsize=8.5,zorder=5)
        ax.set(aspect="equal",xlim=(-1.18,1.18),ylim=(-1.18,1.18),title=name)
        ax.axis("off")
    axes[1].text(.5,-.07,"66 / 66 edges",transform=axes[1].transAxes,ha="center")
    axes[2].text(.5,-.07,"0 / 66 edges",transform=axes[2].transAxes,ha="center")
    fig.text(.5,.16,r"Raw condition number $\approx10^{16}$; Arnoldi orthogonality defect $<10^{-15}$.",ha="center",fontsize=9)
    fig.text(.5,.07,r"Model 1: $J=100$, $T=200$, $m=12$; edges require $|\langle v_i,v_j\rangle|\geq0.20$.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.29,top=.87,wspace=.30)
    save(fig,"krylov_geometry",args)


def make_quadratics(args):
    rows=read_csv(args.code_root/"cg_vs_steepest_descent_results/trajectory_2d.csv")
    paths={name:np.array([[float(r["x1"]),float(r["x2"]),float(r["objective_gap"])] for r in rows if r["method"]==name]) for name in ["Steepest Descent","Conjugate Gradient"]}
    angle=np.deg2rad(32);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    A=rot@np.diag([1.,50.])@rot.T;star=np.array([1.,-.75])
    allpoints=np.vstack([p[:,:2] for p in paths.values()]+[star[None,:]])
    lower=allpoints.min(axis=0);upper=allpoints.max(axis=0);pad=.18*np.maximum(upper-lower,1)
    xx,yy=np.meshgrid(np.linspace(lower[0]-pad[0],upper[0]+pad[0],350),np.linspace(lower[1]-pad[1],upper[1]+pad[1],350))
    dx=xx-star[0];dy=yy-star[1];gap=.5*(A[0,0]*dx**2+2*A[0,1]*dx*dy+A[1,1]*dy**2)
    levels=np.geomspace(max(float(np.quantile(gap[gap>0],.01)),1e-6),float(gap.max()),18)
    fig,ax=plt.subplots(figsize=(3.0,3.0))
    ax.contour(xx,yy,gap,levels=levels,colors="#bbc0c8",linewidths=.6)
    for name,color in zip(paths,[COLORS[1],COLORS[0]]):
        p=paths[name];ax.plot(p[:,0],p[:,1],color=color,marker="o",ms=2.5,markevery=max(1,len(p)//24),label=("SD" if name.startswith("Steepest") else "CG")+f" ({len(p)-1})")
    ax.scatter(*star,marker="*",s=70,color=COLORS[2],zorder=5,label="Minimum")
    ax.set(xlabel=r"$x_1$",ylabel=r"$x_2$",aspect="equal")
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",ncol=1,frameon=False,bbox_to_anchor=(.73,.99),fontsize=8.4,labelspacing=.3)
    fig.subplots_adjust(left=.24,right=.97,bottom=.18,top=.76)
    save(fig,"quadratic_trajectory_2d",args)

    # Lift the archived paths onto the original illustrative quadratic surface.
    ztop=1.08*max(p[:,2].max() for p in paths.values())
    eig,V=np.linalg.eigh(A);rr,tt=np.meshgrid(np.linspace(0,1,70),np.linspace(0,2*np.pi,140),indexing="ij")
    ecoord=np.c_[np.sqrt(2*ztop/eig[0])*rr.ravel()*np.cos(tt.ravel()),np.sqrt(2*ztop/eig[1])*rr.ravel()*np.sin(tt.ravel())]
    points=star+ecoord@V.T
    fig=plt.figure(figsize=(3.,3.));ax=fig.add_subplot(111,projection="3d",computed_zorder=False)
    ax.plot_surface(points[:,0].reshape(rr.shape),points[:,1].reshape(rr.shape),ztop*rr**2,cmap="Blues",alpha=.35,lw=0,rcount=60,ccount=70,zorder=1)
    for name,color in zip(paths,[COLORS[1],COLORS[0]]):
        p=paths[name];ax.plot(p[:,0],p[:,1],p[:,2]+.012*ztop,color=color,lw=1.6,marker="o",ms=2.2,markevery=max(1,len(p)//24),zorder=10)
    ax.scatter(*star,.055*ztop,marker="*",s=70,color=COLORS[2],depthshade=False,zorder=12)
    ax.set_xlabel(r"$x_1$",labelpad=-5);ax.set_ylabel(r"$x_2$",labelpad=-5)
    ax.set_zlim(0,ztop);ax.view_init(elev=30,azim=-55);ax.set_box_aspect((1.2,1,.76))
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.set_major_locator(MaxNLocator(3));axis.set_tick_params(pad=-2,labelsize=8.5);axis.pane.set_alpha(0)
    fig.text(.5,.94,r"Objective gap $f(x)-f(x_*)$",ha="center",fontsize=9.2)
    fig.subplots_adjust(left=.02,right=.90,bottom=.07,top=.87)
    save(fig,"quadratic_surface_3d",args)


def make_inference(s,models,args):
    folder=args.code_root/"babii_inference_results_R5000"
    raw=np.load(folder/"inference_raw_results.npz")
    rows=read_csv(folder/"stopping_summary.csv")
    fig,axes=plt.subplots(3,3,figsize=(WIDTH,6.2))
    for c,name in enumerate(MODEL_NAMES):
        subset=[r for r in rows if r["model"]==name]
        fixed=sorted([r for r in subset if r["stopping_rule"].startswith("m=")],key=lambda r:float(r["component_value"]))
        for ri,(metric,label) in enumerate([("median_sqrt_n_residual",r"Median $\sqrt{n}\|e_m\|$"),("median_abs_stat_difference",r"Median $|\mathcal{T}_{n,m}-\mathcal{S}_n|$"),("rejection_rate","Null rejection")]):
            ax=axes[ri,c]
            ax.plot([float(r["component_value"]) for r in fixed],[float(r[metric]) for r in fixed],color=COLORS[c],marker="o",ms=3)
            for rule,mark,col,size in [("adaptive","*","#8e44ad",60),("max-rank","D","#252525",23)]:
                r=next(r for r in subset if r["stopping_rule"]==rule)
                ax.scatter(float(r["component_value"]),float(r[metric]),marker=mark,color=col,s=size,zorder=5)
            ax.set_xscale("log");ax.set_xlim(.8,130);ax.set_xticks([1,10,100],["1","10","100"])
            ax.set_xlabel(r"Components / rank")
            if ri<2:
                ax.set_yscale("log");grid(ax,True)
            else:
                ax.axhline(.05,color=".4",ls="--",lw=.8)
                ax.set_ylim(-.02,1.03);ax.set_yticks([0,.5,1]);grid(ax)
            if ri==0:
                ax.axhline(float(subset[0]["median_sqrt_n_threshold"]),color=".4",ls=":",lw=1)
                ax.set_title(name)
            if c==0:ax.set_ylabel(label)
    handles=[Line2D([],[],color=".45",marker="o",ms=3,label="Fixed m"),Line2D([],[],color="#8e44ad",marker="*",ls="",ms=8,label="Adaptive variant"),Line2D([],[],color=INK,marker="D",ls="",ms=4,label="Maximum rank")]
    fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False,bbox_to_anchor=(.55,1.0),columnspacing=.9,handlelength=1.3)
    fig.subplots_adjust(left=.12,right=.985,bottom=.07,top=.88,hspace=.63,wspace=.36)
    save(fig,"stopping_diagnostics",args)

    fig,axes=plt.subplots(3,2,figsize=(WIDTH,6.20))
    for ri,name in enumerate(MODEL_NAMES):
        key=name.lower().replace(" ","_")
        empirical=raw[f"null_{key}_late_statistic"];limit=raw[f"oracle_{key}_limit_draws"]
        upper=np.quantile(np.r_[empirical,limit],.995);bins=np.linspace(0,upper,42)
        axes[ri,0].hist(empirical,bins=bins,density=True,color=COLORS[0],alpha=.55)
        axes[ri,0].hist(limit,bins=bins,density=True,histtype="step",color=COLORS[1],lw=1.5)
        axes[ri,0].set(title=name,xlabel="Statistic",ylabel="Density")
        probs=np.linspace(.01,.99,99);eq=np.quantile(empirical,probs);lq=np.quantile(limit,probs)
        axes[ri,1].scatter(eq,lq,s=8,color=COLORS[ri]);lo=min(eq[0],lq[0]);hi=max(eq[-1],lq[-1])
        axes[ri,1].plot([lo,hi],[lo,hi],"--",color=".4",lw=.8)
        axes[ri,1].set(title="Quantile comparison",xlabel="Finite-sample quantiles",ylabel="Reference quantiles")
    fig.legend([Patch(color=COLORS[0],alpha=.55),Line2D([],[],color=COLORS[1])],[r"Finite-sample $\mathcal{T}_{n,70}$","Continuous-design reference"],loc="upper center",ncol=2,bbox_to_anchor=(.55,1),frameon=False,columnspacing=1,handlelength=1.4)
    fig.subplots_adjust(left=.12,right=.98,bottom=.075,top=.88,hspace=.90,wspace=.48)
    save(fig,"null_calibration",args)

    rows=read_csv(folder/"power_summary.csv")
    fig,axes=plt.subplots(3,1,figsize=(WIDTH,5.80),sharex=True,sharey=True)
    for ax,name in zip(axes,MODEL_NAMES):
        for n,col,ls in [(100,COLORS[0],"-"),(200,COLORS[1],"--")]:
            subset=sorted([r for r in rows if r["model"]==name and int(r["n"])==n],key=lambda r:float(r["delta"]))
            ax.plot([float(r["delta"]) for r in subset],[float(r["power"]) for r in subset],ls,color=col,label=f"n = {n}")
        ax.axhline(.05,color=".4",ls=":",lw=1)
        ax.set(title=name,ylabel="Rejection probability",ylim=(0,1.03));ax.set_yticks([0,.5,1]);grid(ax)
    axes[-1].set_xlabel(r"Alternative scale $\delta$")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.55,1))
    fig.subplots_adjust(left=.13,right=.98,bottom=.09,top=.88,hspace=.40)
    save(fig,"power_curves",args)

    fig,axes=plt.subplots(3,1,figsize=(WIDTH,5.80),sharex=True)
    for ax,model in zip(axes,models):
        key=model.name.lower().replace(" ","_")
        ax.fill_between(s,raw[f"confidence_{key}_median_lower"],raw[f"confidence_{key}_median_upper"],color="#A9ADB4",alpha=.50)
        for suffix in ["lower","upper"]:
            ax.plot(s,raw[f"confidence_{key}_representative_{suffix}"],color="#61758b",ls="--",lw=.9)
        ax.plot(s,model.beta,color="black",lw=1.4)
        ax.set(title=model.name,ylabel=r"$\beta(s)$")
    axes[-1].set_xlabel(r"Location $s$")
    fig.legend([Patch(color="#A9ADB4",alpha=.5),Line2D([],[],color="#61758b",ls="--"),Line2D([],[],color="black")],["Pointwise median endpoints","One realised envelope","Complete true slope"],loc="upper center",ncol=1,frameon=False,bbox_to_anchor=(.55,1),labelspacing=.3)
    fig.subplots_adjust(left=.10,right=.98,bottom=.09,top=.82,hspace=.44)
    save(fig,"confidence_sets",args)

    rows=[r for r in read_csv(folder/"size_summary.csv") if float(r["alpha"])==.05 and r["calibration"]=="oracle"]
    fig,ax=plt.subplots(figsize=(WIDTH,3.5))
    for i,name in enumerate(MODEL_NAMES):
        r=next(r for r in rows if r["model"]==name)
        ax.errorbar(i,float(r["rejection_rate"]),yerr=1.96*float(r["mcse"]),fmt="o",color=COLORS[i],capsize=4)
    ax.axhline(.05,color=".4",ls="--",lw=1,label="Nominal 5%")
    ax.set_xticks(range(3),MODEL_NAMES);ax.set(xlim=(-.45,2.45),ylim=(0,.085),ylabel="Null rejection probability")
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.legend(loc="lower right",frameon=False);grid(ax)
    fig.subplots_adjust(left=.13,right=.98,bottom=.16,top=.96)
    save(fig,"calibration_comparison",args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[1]/"figures")
    parser.add_argument("--preview-dir",type=Path)
    args=parser.parse_args()
    args.code_root=args.code_root.resolve();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.preview_dir:args.preview_dir.mkdir(parents=True,exist_ok=True)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(args.code_root))
    core=importlib.import_module("four_method_core")
    style()
    s,basis,models=make_design(core,args)
    make_duality(args);make_geometry(basis,models,args);make_quadratics(args)
    make_inference(s,models,args)
    print(f"Redrew 13 supporting figures in {args.output_dir}")


if __name__=="__main__":
    main()
