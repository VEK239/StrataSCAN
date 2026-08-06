from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

METHODS=['StrataSCAN','DBSCAN','HDBSCAN','OPTICS','SNN-DBSCAN','VDBSCAN-2007','AMD-DBSCAN','kNN-DBSCAN','kNN+Leiden']
ARCH=['Gaussian 4x','Gaussian 16x','Gaussian 64x','Moon arcs','Rings','Anisotropic ellipses']
MARK=dict(zip(METHODS,'os^vDPX<>')); COL=dict(zip(METHODS,['#0072B2','#E69F00','#009E73','#D62728','#7A4FB7','#8C564B','#CC79A7','#7F7F7F','#B3A800']))
AMARK=list('osD^v*'); ACOL=['#0072B2','#E69F00','#009E73','#7A4FB7','#D62728','#8C564B']; ASHORT=['G4','G16','G64','Moon','Rings','Ellipse']

def style():
 mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':7,'axes.labelsize':7.4,'axes.titlesize':8.2,'xtick.labelsize':6.3,'ytick.labelsize':6.5,'legend.fontsize':6,'axes.linewidth':.65,'grid.linewidth':.4,'grid.alpha':.3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,'savefig.bbox':'tight','savefig.pad_inches':.03})
def save(fig,out,name):
 out.mkdir(parents=True,exist_ok=True)
 for e in ('pdf','svg','png'): fig.savefig(out/f'{name}.{e}',dpi=600 if e=='png' else None)
 plt.close(fig)
def label(ax,s): ax.text(-.02,1.035,s,transform=ax.transAxes,ha='right',va='bottom',fontweight='bold',fontsize=8.2)
def noise99(path):
 d=pd.read_csv(path).rename(columns={'target_f1_mean':'target','noise_f1_mean':'noise','pairwise_f1_mean':'pairwise','granularity_ratio_mean':'granularity'})
 d['status']=np.where(d.seeds_completed.fillna(0).gt(0),'ok',np.where(d.status_summary.str.contains('timeout',na=False),'timeout','resource skip'))
 return d
def fig1(d,out):
 fig,ax=plt.subplots(1,3,figsize=(7.16,3.55),sharey=True); fig.subplots_adjust(left=.145,right=.995,top=.88,bottom=.27,wspace=.1); y=np.arange(9)[::-1]; off=np.linspace(-.18,.18,6)
 for p,(a,(m,t)) in enumerate(zip(ax,[('target','Target-wise F1'),('noise','Noise F1'),('pairwise','Pairwise F1')])):
  for r,method in enumerate(METHODS):
   s=d[d.method.eq(method)].set_index('architecture').reindex(ARCH); v=pd.to_numeric(s[m],errors='coerce').to_numpy(float); ok=np.isfinite(v)
   if ok.any():
    for i in np.flatnonzero(ok): a.scatter(v[i],y[r]+off[i],s=16,marker=AMARK[i],color=ACOL[i],edgecolor='white',linewidth=.25,zorder=4)
    mean,lo,hi=np.nanmean(v),np.nanmin(v),np.nanmax(v); a.errorbar(mean,y[r],xerr=[[mean-lo],[hi-mean]],fmt='o',color='black',ms=4.5,capsize=2.2,lw=.9,zorder=5)
   else: a.text(.98,y[r],'TO x6' if s.status.astype(str).eq('timeout').all() else 'RS x6',ha='right',va='center',fontsize=6.3)
  a.set(xlim=(0,1.05),xlabel='F1 (higher is better)',title=t); a.set_xticks(np.linspace(0,1,6)); label(a,f'({chr(97+p)})'); a.grid(axis='x'); a.axhspan(y[0]-.48,y[0]+.48,facecolor='#F3F3F3',edgecolor='black',lw=.65,zorder=-1)
  for yy in y[:-1]-.5:a.axhline(yy,color='#B8B8B8',lw=.45,ls=(0,(2,3)),zorder=0)
 ax[0].set_yticks(y,METHODS); ax[0].get_yticklabels()[0].set_fontweight('bold'); [a.tick_params(labelleft=False) for a in ax[1:]]
 h=[Line2D([0],[0],marker=AMARK[i],ls='none',mfc=ACOL[i],mec='white',ms=5,label=ASHORT[i]) for i in range(6)]+[Line2D([0],[0],marker='o',color='black',ms=4.5,lw=.9,label='Mean (min-max)')]
 fig.legend(handles=h,loc='lower center',bbox_to_anchor=(.56,.105),ncol=7,frameon=True,fancybox=False); fig.text(.57,.035,'G4/G16/G64: compact Gaussian targets with 4x/16x/64x density ranges. TO: timeout; RS: resource skip.',ha='center',fontsize=6); save(fig,out,'fig1_noise99_dotwhisker')
def granularity(d,out):
 q=d[d.status.eq('ok')].copy(); q['abslog']=abs(np.log2(q.granularity.astype(float))); t=q.groupby('method').agg(completed=('granularity','size'),median_ratio=('granularity','median'),min_ratio=('granularity','min'),max_ratio=('granularity','max'),median_abs_log2=('abslog','median')).reindex(METHODS); t['coverage']=t.completed.fillna(0).astype(int).astype(str)+'/6'; t.reset_index().to_csv(out/'table1_noise99_granularity.csv',index=False)
 lines=['\\begin{table}[t]','\\centering','\\caption{Granularity on six 99\\% diffuse-background benchmarks. A ratio of 1 is exact; values above 1 indicate over-segmentation. Summaries use completed runs only.}','\\label{tab:noise99-granularity}','\\scriptsize','\\begin{tabular}{lrrrr}','\\toprule','Method & Coverage & Median ratio & Range & Median $|\\log_2 r|$ \\\\','\\midrule']
 f=lambda x:'--' if pd.isna(x) else (f'{x:,.1f}x' if x>=10 else f'{x:.2f}x')
 for m in METHODS:
  r=t.loc[m]; name='\\textbf{StrataSCAN}' if m=='StrataSCAN' else m; vals=('--','--','--') if pd.isna(r.median_ratio) else (f(r.median_ratio),f'{f(r.min_ratio)}--{f(r.max_ratio)}',f'{r.median_abs_log2:.2f}'); lines.append(f'{name} & {r.coverage} & {vals[0]} & {vals[1]} & {vals[2]} \\\\')
 lines+=['\\bottomrule','\\end{tabular}','\\end{table}']; (out/'table1_noise99_granularity.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def fig2(path,out):
 d=pd.read_csv(path).rename(columns={'target_f1_mean':'target','noise_f1_mean':'noise','pairwise_f1_mean':'pairwise'}); lev=[.5,.75,.9,.95,.975,.99]; x=np.arange(6); fig,ax=plt.subplots(1,3,figsize=(7.16,3.2),sharey=True); fig.subplots_adjust(left=.07,right=.995,top=.87,bottom=.34,wspace=.12)
 for p,(a,(met,t)) in enumerate(zip(ax,[('target','Target-wise F1'),('noise','Noise F1'),('pairwise','Pairwise F1')])):
  for m in METHODS:
   s=d[d.method.eq(m)].set_index('noise_fraction').reindex(lev); v=pd.to_numeric(s[met],errors='coerce').to_numpy(float); ok=np.isfinite(v)
   if ok.any():a.plot(x[ok],v[ok],marker=MARK[m],color=COL[m],ms=4.2 if m=='StrataSCAN' else 3.1,lw=2 if m=='StrataSCAN' else .9,zorder=5 if m=='StrataSCAN' else 2)
   if (~ok).any():a.scatter(x[~ok],np.full((~ok).sum(),.015),marker='x',s=18,color=COL[m],lw=.9,zorder=6)
  a.set(xlim=(-.2,5.2),ylim=(-.03,1.03),title=t); a.set_xticks(x,['50','75','90','95','97.5','99']); label(a,f'({chr(97+p)})'); a.grid(True)
 ax[0].set_ylabel('F1'); fig.text(.5,.245,'Diffuse-background fraction (%)',ha='center',fontsize=7.4); h=[Line2D([0],[0],marker=MARK[m],color=COL[m],lw=2 if m=='StrataSCAN' else .9,ms=4.2 if m=='StrataSCAN' else 3.1,label=m) for m in METHODS]; fig.legend(handles=h,loc='lower center',bbox_to_anchor=(.5,.08),ncol=5,frameon=True,fancybox=False); fig.text(.5,.018,'Crosses at the lower boundary denote unavailable measurements (timeout or resource skip), not F1=0.',ha='center',fontsize=6); save(fig,out,'fig2_noise_fraction_sweep')
STATUS=[('completed','black',None),('timeout','white','////'),('memory','lightgray','|||'),('error','white','xx'),('skipped','white','\\\\'),('pending','white','...')]
def scale_frame(path):
 d=pd.read_csv(path).rename(columns={'runtime_seconds_median_success':'runtime'}); return d
def fig3(d,out):
 fig=plt.figure(figsize=(7.16,4.05)); gs=fig.add_gridspec(1,2,width_ratios=[1.02,1.43],left=.075,right=.995,top=.88,bottom=.26,wspace=.28); a,b=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
 for m in METHODS:
  s=d[d.method.eq(m)&d.runtime.notna()].sort_values('n')
  if not s.empty:a.plot(s.n,s.runtime,marker=MARK[m],color=COL[m],ms=4.2 if m=='StrataSCAN' else 3.1,lw=2 if m=='StrataSCAN' else .9,zorder=5 if m=='StrataSCAN' else 2)
 a.set_xscale('log');a.set_yscale('log');a.set_xticks([5e5,1e6,2e6,5e6],['0.5M','1M','2M','5M']);a.set(xlabel='Observations',ylabel='Median runtime of successful cells (s)',title='Runtime, successful cells only');label(a,'(a)');a.grid(True,which='both')
 sizes=[500000,1000000,2000000,5000000];b.set(xlim=(0,4),ylim=(0,9));b.invert_yaxis();b.set_xticks(np.arange(4)+.5,['0.5M','1M','2M','5M']);b.xaxis.tick_top();b.set_yticks(np.arange(9)+.5,METHODS);b.tick_params(length=0)
 for i in range(10):b.axhline(i,color='black',lw=.45)
 for j in range(5):b.axvline(j,color='black',lw=.45)
 for i,m in enumerate(METHODS):
  for j,n in enumerate(sizes):
   r=d[d.method.eq(m)&d.n.eq(n)].iloc[0]; c={'completed':int(r.cells_ok),'timeout':int(r.timeouts),'memory':int(r.memory_limits),'error':int(r.errors),'skipped':int(r.skipped_prior_failure),'pending':int(r.pending)}; cur=0
   for st,face,hat in STATUS:
    for _ in range(c[st]):b.add_patch(Rectangle((j+.12+cur*.103,i+.18),.095,.24,facecolor=face,edgecolor='black',hatch=hat,lw=.35));cur+=1
   parts=[f'{c[s]}{code}' for s,code in [('timeout','TO'),('memory','ML'),('error','E'),('skipped','S'),('pending','P')] if c[s]];b.text(j+.5,i+.7,f"{c['completed']}/{int(r.cells_declared)}"+(' + '+' + '.join(parts) if parts else ''),ha='center',va='center',fontsize=5.1)
 b.set_title('Completion and failure state',fontweight='bold',pad=12);label(b,'(b)');h=[Line2D([0],[0],marker=MARK[m],color=COL[m],lw=2 if m=='StrataSCAN' else .9,ms=4 if m=='StrataSCAN' else 3,label=m) for m in METHODS];fig.legend(handles=h,loc='lower left',bbox_to_anchor=(.055,.055),ncol=3,frameon=True,fancybox=False); sh=[Rectangle((0,0),1,1,facecolor=f,edgecolor='black',hatch=h,label=n) for n,f,h in STATUS];fig.legend(handles=sh,loc='lower right',bbox_to_anchor=(.995,.055),ncol=3,frameon=True,fancybox=False);fig.text(.5,.015,'TO: timeout; ML: memory limit; E: error; S: skipped after an earlier failure; P: pending. Pending cells are not scored as failures.',ha='center',fontsize=5.8);save(fig,out,'fig3_scalability')
def galleries(out):
 from benchmarks.datasets import load_synthetic
 specs=[{'family':'multidensity','dimension':2,'noise_fraction':.75},{'family':'ultrasparse','dimension':16,'noise_fraction':.95},{'family':'overlapping_density','dimension':16,'noise_fraction':.8,'signal_fraction':.2},{'family':'moons','dimension':2,'noise_fraction':.25},{'family':'rings','dimension':2,'noise_fraction':.75},{'family':'gaussian_overlap','dimension':8,'noise_fraction':.25}];titles=['Multidensity 2D','Ultrasparse 16D','Overlapping density 16D','Moons 2D','Rings 2D','Gaussian overlap 8D']; fig,ax=plt.subplots(2,3,figsize=(7.16,3.35));fig.subplots_adjust(left=.025,right=.995,top=.89,bottom=.05,hspace=.42,wspace=.08);rng=np.random.default_rng(77)
 for i,(a,s,t) in enumerate(zip(ax.flat,specs,titles)):
  ds=load_synthetic({**s,'n':20000,'tier':'quality'},42);p=ds.X if ds.X.shape[1]==2 else PCA(n_components=2,random_state=42).fit_transform(ds.X);bg=np.flatnonzero(ds.y<0);keep=rng.choice(bg,min(4000,len(bg)),replace=False);a.scatter(p[keep,0],p[keep,1],s=.45,color='#56A0D3',alpha=.24,lw=0,rasterized=True)
  for c in np.unique(ds.y[ds.y>=0]):q=np.flatnonzero(ds.y==c);a.scatter(p[q,0],p[q,1],s=1.3,alpha=.8,lw=0,rasterized=True)
  a.set_title(t+('\n(PCA view)' if ds.X.shape[1]>2 else ''),fontweight='bold',pad=3);a.text(-.01,1.04,f'({chr(97+i)})',transform=a.transAxes,ha='right',fontweight='bold',fontsize=8);a.text(.02,.02,f'noise={np.mean(ds.y<0):.0%}, k={len(np.unique(ds.y[ds.y>=0]))}',transform=a.transAxes,fontsize=6,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5));a.set_xticks([]);a.set_yticks([])
 save(fig,out,'fig4_standard_benchmark_gallery')
 from topology_noise_benchmark.generators import generate
 specs=[('compact_gaussian',4,'Gaussian 4x'),('compact_gaussian',16,'Gaussian 16x'),('compact_gaussian',64,'Gaussian 64x'),('moon_arcs',1,'Moon arcs'),('rings',1,'Rings'),('anisotropic_ellipses',1,'Anisotropic ellipses')];fig,ax=plt.subplots(2,3,figsize=(7.16,3.35));fig.subplots_adjust(left=.025,right=.995,top=.89,bottom=.05,hspace=.42,wspace=.08)
 for i,(a,(top,ratio,t)) in enumerate(zip(ax.flat,specs)):
  ds=generate(topology=top,density_ratio=float(ratio),noise_fraction=.99,geometry_seed=3571,sampling_seed=42);bg=np.flatnonzero(ds.y<0);keep=rng.choice(bg,5000,replace=False);a.scatter(ds.X[keep,0],ds.X[keep,1],s=.42,color='#56A0D3',alpha=.22,lw=0,rasterized=True)
  for c in np.unique(ds.y[ds.y>=0]):q=np.flatnonzero(ds.y==c);a.scatter(ds.X[q,0],ds.X[q,1],s=1.2,alpha=.85,lw=0,rasterized=True)
  a.set_title(t,fontweight='bold',pad=3);a.text(-.01,1.04,f'({chr(97+i)})',transform=a.transAxes,ha='right',fontweight='bold',fontsize=8);a.text(.02,.02,'noise=99%, k=6',transform=a.transAxes,fontsize=6,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5));a.set_xticks([]);a.set_yticks([])
 save(fig,out,'fig5_noise99_benchmark_gallery')
def main():
 p=argparse.ArgumentParser();root=Path(__file__).resolve().parents[3];tab=root/'manuscript/oedm2026/tables/performance_tables_2026-08-06';p.add_argument('--noise99',type=Path,default=tab/'02_noise_aware_6datasets_summary.csv');p.add_argument('--sweep',type=Path,default=tab/'03_noise_fraction_summary_long.csv');p.add_argument('--scale',type=Path,default=tab/'04_scalability_available_to_5m_summary.csv');p.add_argument('--output-dir',type=Path,default=root/'manuscript/oedm2026/figures/performance_2026-08-06');p.add_argument('--skip-galleries',action='store_true');a=p.parse_args();style();a.output_dir.mkdir(parents=True,exist_ok=True);d=noise99(a.noise99);fig1(d,a.output_dir);granularity(d,a.output_dir);fig2(a.sweep,a.output_dir);fig3(scale_frame(a.scale),a.output_dir)
 if not a.skip_galleries:galleries(a.output_dir)
 (a.output_dir/'provenance.json').write_text(json.dumps({'metric_sources':[str(a.noise99),str(a.sweep),str(a.scale)],'gallery_sources':['benchmarks.datasets.load_synthetic','topology_noise_benchmark.generators.generate'],'policy':'No image generation, no heatmaps, no failure-to-zero imputation.'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
