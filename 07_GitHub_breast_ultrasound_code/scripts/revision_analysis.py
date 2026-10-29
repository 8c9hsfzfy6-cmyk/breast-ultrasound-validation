"""R17: exploratory CPU supplements using frozen BUS predictions and features.
No diagnostic model fitting, prediction updating, threshold selection or feature selection.
Run on Kaggle; local execution is a small implementation check only.
"""
from pathlib import Path
import os, json, hashlib, warnings, shutil, sys, platform, time
from collections import Counter
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import brentq, minimize
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = int(os.environ.get('BUS_BOOTSTRAPS', '2000'))
SEED = 20261029
ROOT = Path(os.environ.get('BUS_INPUT_ROOT', '/kaggle/input'))
OUT = Path(os.environ.get('BUS_OUTPUT_ROOT', '/kaggle/working/BUS_R17_20261029'))
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()
LOG = []
def log(s):
    print(s, flush=True); LOG.append(str(s))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d, name):
    pd.DataFrame(d).to_csv(OUT/(name+'.csv'), index=False, float_format='%.17g')
def read(p):
    return pd.read_csv(p, float_precision='round_trip', dtype={
        'uid':str,'image_id':str,'id':str,'patient_id':str,'patient_key':str,'cluster_id':str})
def package(prefix, marker):
    candidates=sorted({p.parent for p in ROOT.rglob(marker) if p.parent.name.startswith(prefix)})
    if not candidates: raise FileNotFoundError((prefix, marker))
    # Duplicate mounts are allowed only when the relevant marker has identical bytes.
    assert len({sha(p/marker) for p in candidates})==1, ('conflicting packages', candidates)
    return candidates[0]

PROTOCOL = {
 'analysis':'R17 exploratory supplements after review; not prospectively prespecified',
 'bootstrap_replicates':B,'seed':SEED,'cluster_sampling':'K patients with replacement; all their images retained',
 'weighting':'image_equal: m_i per image; patient_equal: m_i/n_i per image, original cluster size',
 'pairing':'one shared cluster multiplicity matrix for all configurations and weighting schemes within cohort',
 'calibration':'CITL: ML intercept with logit(clipped p) offset, slope fixed 1; slope from joint ML intercept+slope',
 'calibration_clipping':[1e-6,1-1e-6],
 'calibration_failures':'retain NaN and reason; single-class, separation warnings, nonconvergence or singular Hessian; no Firth substitution',
 'intervals':'95% percentile over valid replicates; valid/failed counts reported, conditional on frozen models',
 'icc':'ICC(A,1), two-way mixed, single-measure absolute agreement, image-weighted point estimate; patient cluster bootstrap',
 'icc_population':'per-feature complete triplets, no imputation; binary indicators reported as transitions; no reliability cutoff',
 'missingness':'descriptive association only; no MAR/MNAR identification, no IPW',
 'thresholds':'archived decisions retained; exact p>=t consistency audited without tolerance adjustment',
 'historical_splits':'five seeds 42-46, model seed42; recovered rounded Notebook summaries only, no fabricated per-split table',
 'refit_diagnostic_model':False,'external_recalibration':False,'gpu':False,
 'calibration_solver':'unpenalized binomial ML Newton method with backtracking; score tolerance 1e-8, 100 iterations',
 'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
}
(OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2),encoding='utf8')

def boot_counts(cluster, seed):
    keys, inv=np.unique(np.asarray(cluster,dtype=str),return_inverse=True)
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(keys), np.ones(len(keys))/len(keys), size=B)
    return keys,inv,counts

def calfit(y,p,w):
    keep=w>0; y=y[keep];p=p[keep];w=w[keep]
    if np.unique(y).size<2: return np.nan,np.nan,'single_class','single_class'
    x=logit(np.clip(p,1e-6,1-1e-6))
    citl=brentq(lambda a: np.sum(w*(expit(a+x)-y)), -60,60,xtol=1e-11)
    # Complete/quasi separation in one predictor has no finite ordinary ML slope.
    if x[y==0].max()<=x[y==1].min() or x[y==1].max()<=x[y==0].min():
        return citl,np.nan,'ok','separation'
    try:
        X=np.column_stack([np.ones(len(x)),x]);theta=np.array([citl,1.])
        wn=w/w.sum()
        def nll(t):
            z=X@t;return np.sum(wn*(np.logaddexp(0,z)-y*z))
        for iteration in range(100):
            mu=expit(X@theta);grad=X.T@(wn*(mu-y))
            h=X.T@((wn*mu*(1-mu))[:,None]*X)
            if np.linalg.matrix_rank(h)<2 or np.linalg.cond(h)>1e14:
                return citl,np.nan,'ok','singular_information'
            if np.max(abs(grad))<1e-8: return citl,float(theta[1]),'ok','ok'
            step=np.linalg.solve(h,grad);rate=1.;old=nll(theta)
            while rate>1e-10:
                candidate=theta-rate*step
                if nll(candidate)<=old-1e-4*rate*np.dot(grad,step): break
                rate*=.5
            if rate<=1e-10: return citl,np.nan,'ok','line_search_failed'
            theta=candidate
        return citl,np.nan,'ok','nonconvergence'
    except Exception as e: return citl,np.nan,'ok',type(e).__name__

def metrics(y,p,d,w):
    pos=np.sum(w*y);neg=np.sum(w*(1-y));valid=pos>0 and neg>0
    return [roc_auc_score(y,p,sample_weight=w) if valid else np.nan,
            np.average((p-y)**2,weights=w),
            np.sum(w*y*d)/pos if pos else np.nan,
            np.sum(w*(1-y)*(1-d))/neg if neg else np.nan]

def cirow(vals,point,**kw):
    vals=np.asarray(vals); good=vals[np.isfinite(vals)]
    lo,hi=np.quantile(good,[.025,.975]) if len(good) else (np.nan,np.nan)
    return dict(**kw,estimate=point,ci_low=lo,ci_high=hi,n_valid=len(good),n_failed=len(vals)-len(good))

def icc_a1(a):
    n,k=a.shape
    if n<2 or not np.isfinite(a).all(): return np.nan
    grand=a.mean();row=a.mean(1);col=a.mean(0)
    msr=k*np.sum((row-grand)**2)/(n-1)
    msc=n*np.sum((col-grand)**2)/(k-1)
    mse=np.sum((a-row[:,None]-col[None,:]+grand)**2)/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return (msr-mse)/den if den>0 else np.nan

def self_checks():
    # Duplicating sampled clusters and multiplicity weighting must be identical.
    y=np.array([0,1,0,1,0]);p=np.array([.1,.7,.4,.6,.3]);ids=np.array([0,0,1,2,2]);m=np.array([2,0,1]);size=np.bincount(ids)
    ix=np.repeat(np.arange(len(y)),m[ids]);w=m[ids]/size[ids]
    assert np.isclose(roc_auc_score(y,p,sample_weight=w),roc_auc_score(y[ix],p[ix],sample_weight=1/size[ids[ix]]))
    a=np.column_stack([np.arange(10.)]*3)
    assert abs(icc_a1(a)-1)<1e-12
    assert icc_a1(a+np.array([0,1,2]))<1
    assert np.isnan(icc_a1(np.ones((8,3))))
    yy=np.array([0,1,0,1,1,0,0,1]);pp=np.array([.1,.2,.3,.4,.6,.7,.8,.9]);ww=np.ones(8)
    c,s,cs,ss=calfit(yy,pp,ww)
    x=logit(pp);X=np.column_stack([np.ones(len(x)),x])
    r=minimize(lambda t: np.mean(np.logaddexp(0,X@t)-yy*(X@t)),[0.,1.],jac=lambda t:X.T@(expit(X@t)-yy)/len(x),method='BFGS',options={'gtol':1e-10})
    assert cs=='ok' and ss=='ok' and abs(s-r.x[1])<1e-6
    assert abs(np.sum(expit(c+x)-yy))<1e-8
    assert calfit(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),np.ones(4))[3]=='separation'
    return {'multiplicity_equals_physical_duplication':True,'icc_identity_shift_constant':True,'citl_offset_score_zero':True,'slope_matches_independent_BFGS':True,'separation_detected':True}

def main():
    checks=self_checks();(OUT/'implementation_checks.json').write_text(json.dumps(checks,indent=2))
    if '__file__' in globals(): shutil.copyfile(__file__,OUT/'revision_analysis.py')
    f=package('bus_v6_features_','features.csv');t=package('bus_v6_external_training_','external_predictions.csv')
    u=package('bus_uclm_external_','predictions_with_labels.csv');c=package('bus_uclm_contour_','perturbed_features.csv')
    v=package('interpretability_v11','build.py')
    inputs=[]
    for folder in [f,t,u,c,v]:
        for p in sorted(folder.rglob('*')):
            if p.is_file(): inputs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    save(inputs,'input_manifest')
    for name,h in json.loads((t/'model_hashes.json').read_text()).items():
        assert sha(t/'models'/name)==h, ('model hash',name)
    for name,h in json.loads((t/'protocol.json').read_text())['input_hashes'].items():
        assert sha(f/name)==h, ('training input hash',name)
    meta=read(f/'cohort.csv');features=read(f/'features.csv');quality=read(f/'quality.csv')
    allframe=meta.merge(features,on='uid',validate='one_to_one').merge(quality,on='uid',validate='one_to_one')
    uc=read(u/'cohort_frozen.csv');uf=uc.merge(read(u/'features.csv'),on='image_id',validate='one_to_one').merge(read(u/'quality.csv'),on='image_id',validate='one_to_one')
    assert len(allframe)==2127 and len(uc)==207 and uc.cluster_id.nunique()==33
    orig=read(c/'perturbed_features.csv'); orig=orig[orig['mode'].eq('original')].set_index('image_id').sort_index()
    prior=read(u/'features.csv').set_index('image_id').loc[orig.index]
    assert np.allclose(orig[prior.columns],prior,atol=0,rtol=0,equal_nan=True), 'original perturbation features differ'
    # Descriptive availability. Every cohort includes all its frozen evaluable cases.
    avail=[]; numeric=[];avdiff=[]
    groups=[(ds,g.copy(),'patient_key') for ds,g in allframe.groupby('dataset')]+[('BUS-UCLM',uf,'cluster_id')]
    for ds,g,key in groups:
        a=g.posterior_usable.astype(int).to_numpy();y=g.label.astype(int).to_numpy()
        assert np.array_equal(a,1-g['postdepth__missing'].astype(int))
        keys,iv,bc=boot_counts(g[key],SEED+1);sz=np.bincount(iv)
        np.savez_compressed(OUT/f'{ds}_availability_resamples.npz',clusters=keys,counts=bc)
        for weighting in ['image_equal','patient_equal']:
            w=np.ones(len(g)) if weighting=='image_equal' else 1/sz[iv]
            for status in [0,1]:
                q=a==status
                avail.append(dict(dataset=ds,weighting=weighting,available=status,n_images=int(q.sum()),n_clusters=g.loc[q,key].nunique(),malignant=int(y[q].sum()),weighted_share=np.sum(w[q])/w.sum(),weighted_malignant_fraction=np.average(y[q],weights=w[q])))
            point=np.average(y[a==1],weights=w[a==1])-np.average(y[a==0],weights=w[a==0])
            draws=[]
            for b in bc:
                wb=w*b[iv]
                draws.append(np.average(y[a==1],weights=wb[a==1])-np.average(y[a==0],weights=wb[a==0]) if wb[a==1].sum()>0 and wb[a==0].sum()>0 else np.nan)
            avdiff.append(cirow(draws,point,dataset=ds,weighting=weighting,metric='malignant_fraction_available_minus_unavailable'))
        for status in [0,1]:
            for col in ['shape__area','shape__aspect','posterior_coverage','left_coverage','right_coverage','matched_row_coverage']:
                vv=g.loc[a==status,col].dropna()
                numeric.append(dict(dataset=ds,available=status,variable=col,n=len(vv),q25=vv.quantile(.25),median=vv.median(),q75=vv.quantile(.75)))
    save(avail,'availability_summary');save(numeric,'availability_geometry');save(avdiff,'availability_association_intervals')
    log('Availability analysis complete')
    # Frozen external predictions: no inferencing, rounding, or threshold changes.
    bp=read(t/'external_predictions.csv');bp['configuration']=bp.feature_set+'_'+bp.model
    bp=bp.rename(columns={'patient_key':'id','p':'probability','prediction':'predicted_label'});bp['cluster_id']=bp.id
    up=read(u/'predictions_with_labels.csv').rename(columns={'image_id':'id'})
    metric_rows=[];cal_rows=[];delta_rows=[];failure_rows=[];decision_rows=[];boots=[]
    for ds,pr,expected in [('BrEaST',bp,252),('BUS-UCLM',up,207)]:
        configs=sorted(pr.configuration.unique());ref=pr[pr.configuration.eq(configs[0])].sort_values('id')
        assert len(ref)==expected and ref.id.is_unique
        keys,iv,bc=boot_counts(ref.cluster_id,SEED);sz=np.bincount(iv)
        np.savez_compressed(OUT/f'{ds}_prediction_resamples.npz',clusters=keys,counts=bc)
        ys=ref.label.to_numpy(int)
        for weighting in (['image_equal'] if ds=='BrEaST' else ['image_equal','patient_equal']):
            w=np.ones(len(ref)) if weighting=='image_equal' else 1/sz[iv]
            draws_by={};points_by={}
            for config in configs:
                g=pr[pr.configuration.eq(config)].set_index('id').loc[ref.id].reset_index()
                assert np.array_equal(g.label,ys) and np.array_equal(g.cluster_id,ref.cluster_id)
                p=g.probability.to_numpy(float);d=g.predicted_label.to_numpy(int)
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                decision_rows.append(dict(dataset=ds,weighting=weighting,configuration=config,n=expected,exact_threshold_disagreements=int(np.sum((p>=g.threshold.to_numpy())!=d))))
                assert np.array_equal(p>=g.threshold.to_numpy(),d)
                point=metrics(ys,p,d,w);cp,sp,cs,ss=calfit(ys,p,w)
                arr=np.full((B,6),np.nan);reasons=Counter()
                for b,m in enumerate(bc):
                    wb=w*m[iv]; arr[b,:4]=metrics(ys,p,d,wb)
                    ci,sl,rc,rs=calfit(ys,p,wb);arr[b,4:]=[ci,sl]
                    reasons[('citl',rc)]+=1;reasons[('slope',rs)]+=1
                    if rc!='ok' or rs!='ok': failure_rows.append(dict(dataset=ds,weighting=weighting,configuration=config,replicate=b,citl_status=rc,slope_status=rs))
                for j,name in enumerate(['auc','brier','sensitivity','specificity']):
                    metric_rows.append(cirow(arr[:,j],point[j],dataset=ds,weighting=weighting,configuration=config,metric=name))
                for j,name,pointv,status in [(4,'citl_offset',cp,cs),(5,'calibration_slope',sp,ss)]:
                    cal_rows.append(cirow(arr[:,j],pointv,dataset=ds,weighting=weighting,configuration=config,metric=name,point_status=status))
                draws_by[config]=arr;points_by[config]=point
                boots.append((ds+'_'+weighting+'_'+config,arr))
                log(f'{ds} {weighting} {config}: calibration and metrics complete ({B} resamples)')
            for model in ['LR','ensemble']:
                ca='base_posterior_'+model;cb='base_missing_'+model
                for j,name in enumerate(['auc','brier','sensitivity','specificity']):
                    delta_rows.append(cirow(draws_by[ca][:,j]-draws_by[cb][:,j],points_by[ca][j]-points_by[cb][j],dataset=ds,weighting=weighting,model=model,metric=name,comparison='base_posterior_minus_base_missing'))
    save(metric_rows,'frozen_metrics_with_intervals');save(cal_rows,'calibration_intervals');save(delta_rows,'paired_primary_increment')
    save(decision_rows,'frozen_decision_checks')
    save(pd.DataFrame(failure_rows,columns=['dataset','weighting','configuration','replicate','citl_status','slope_status']),'calibration_failed_replicates')
    np.savez_compressed(OUT/'metric_bootstrap_draws.npz',**dict(boots))
    # ICC on per-feature complete triplets, resampling independent patient clusters.
    pf=read(c/'perturbed_features.csv'); modes=['original','erode1','dilate1']
    assert not pf.duplicated(['image_id','mode']).any()
    ids=uc.sort_values('image_id').image_id;cl=uc.set_index('image_id').loc[ids,'cluster_id']
    keys,iv,bc=boot_counts(cl,SEED+2);np.savez_compressed(OUT/'icc_resamples.npz',clusters=keys,counts=bc)
    iccs=[];binary=[];change=[]
    for col in [x for x in pf.columns if '__' in x]:
        a=pf.pivot(index='image_id',columns='mode',values=col).loc[ids,modes].to_numpy(float)
        valid=np.isfinite(a).all(1);aa=a[valid];vi=iv[valid]
        if col.endswith('__missing'):
            for j,mode in enumerate(modes[1:],1):
                for before in [0,1]:
                    for after in [0,1]:
                        binary.append(dict(feature=col,mode=mode,before=before,after=after,n_images=int(((a[:,0]==before)&(a[:,j]==after)).sum())))
            continue
        point=icc_a1(aa);dr=[]
        for m in bc:
            xx=np.repeat(np.arange(len(aa)),m[vi]);dr.append(icc_a1(aa[xx]))
        iccs.append(cirow(dr,point,feature=col,n_complete=len(aa),n_excluded=int((~valid).sum()),n_clusters=int(len(np.unique(vi))),point_status='ok' if np.isfinite(point) else 'undefined'))
        for j,mode in enumerate(modes[1:],1):
            vv=np.isfinite(a[:,0])&np.isfinite(a[:,j]);diff=a[vv,j]-a[vv,0]
            change.append(dict(feature=col,mode=mode,n_pairs=len(diff),median_signed_change=np.median(diff),median_absolute_change=np.median(abs(diff)),p95_absolute_change=np.quantile(abs(diff),.95)))
    save(iccs,'feature_icc_cluster_intervals');save(binary,'binary_missingness_transitions');save(change,'feature_paired_changes')
    es=read(c/'extraction_status.csv');save(es.groupby('mode').dice_with_original.agg(['count','min','median','max']).reset_index(),'contour_dice_summary')
    # Preserve archived flip directions; augment with distance to original threshold.
    pp=read(c/'all_predictions.csv');margins=[];flip_summary=[]
    for config,g in pp.groupby('configuration'):
        original=g[g['mode'].eq('original')].set_index('image_id').sort_index()
        for mode in modes[1:]:
            alt=g[g['mode'].eq(mode)].set_index('image_id').loc[original.index]
            assert np.array_equal(original.label,alt.label) and np.array_equal(original.threshold,alt.threshold)
            d0=original.predicted_label.to_numpy(int);d1=alt.predicted_label.to_numpy(int);y=original.label.to_numpy(int)
            delta=alt.probability.to_numpy()-original.probability.to_numpy();margin=original.probability.to_numpy()-original.threshold.to_numpy()
            flip=d0!=d1
            for j,key in enumerate(original.index):
                margins.append(dict(configuration=config,mode=mode,image_id=key,cluster_id=original.iloc[j].cluster_id,label=y[j],signed_original_margin=margin[j],absolute_original_margin=abs(margin[j]),probability_change=delta[j],threshold_crossed=bool(flip[j])))
            flip_summary.append(dict(configuration=config,mode=mode,n=len(y),crossings=int(flip.sum()),new_FN=int(((y==1)&(d0==1)&(d1==0)).sum()),corrected_FN=int(((y==1)&(d0==0)&(d1==1)).sum()),new_FP=int(((y==0)&(d0==0)&(d1==1)).sum()),corrected_FP=int(((y==0)&(d0==1)&(d1==0)).sum()),median_margin_crossed=float(np.median(abs(margin[flip]))) if flip.any() else np.nan))
    save(margins,'threshold_margin_cases');save(flip_summary,'threshold_flip_summary')
    # V11 independent additive reconstruction using archived feature contributions.
    vr=[]
    for ds in ['BrEaST','BUS-UCLM']:
        fc=read(v/f'{ds}_all_feature_contributions.csv').set_index('id')
        gc=read(v/f'{ds}_all_group_contributions.csv').set_index('id').loc[fc.index]
        reconstructed=expit(gc.baseline+fc.sum(1)+gc.clipping_correction)
        pred=bp if ds=='BrEaST' else up
        ref=pred[pred.configuration.eq('base_LR')].set_index('id').loc[fc.index]
        err=np.max(abs(reconstructed-ref.probability))
        assert err<1e-8 and np.array_equal(gc.label,ref.label)
        vr.append(dict(dataset=ds,n=len(fc),n_features=fc.shape[1],max_abs_probability_error=err,build_script_sha256=sha(v/'build.py'),scope='archived contribution additivity and frozen prediction agreement; no new image extraction'))
    save(vr,'v11_additivity_validation')
    # Optional recovered historical rounded summaries are injected by the notebook builder.
    for name,records in globals().get('HISTORICAL_TABLES',{}).items(): save(records,name)
    (OUT/'historical_split_limitations.txt').write_text('Historical Notebook cell 19 confirms five completed full refits, split seeds42-46/model seed42. Embedded summary tables are rounded historical outputs; raw metrics_by_split.csv was not recovered. No new training or synthetic per-split values. SD is descriptive, not SE or an independent-test CI.\n')
    # Compact export figure from current results.
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    av=pd.DataFrame(avail);sel=av[(av.weighting=='image_equal')&(av.available==1)]
    axes[0].bar(sel.dataset,sel.weighted_share,color='#247d91');axes[0].set(title='Posterior measurement availability',ylabel='Fraction of frozen cohort',ylim=(0,1))
    cr=pd.DataFrame(cal_rows);sel=cr[(cr.configuration=='base_LR')&(cr.metric=='calibration_slope')]
    lab=sel.dataset+' / '+sel.weighting
    axes[1].errorbar(np.arange(len(sel)),sel.estimate,yerr=np.vstack([sel.estimate-sel.ci_low,sel.ci_high-sel.estimate]),fmt='o',color='#247d91',capsize=3)
    axes[1].axhline(1,ls='--',color='gray');axes[1].set(title='Frozen base LR calibration slope',xticks=np.arange(len(sel)),xticklabels=lab);axes[1].tick_params(axis='x',rotation=25)
    ir=pd.DataFrame(iccs).sort_values('estimate').head(8)
    axes[2].errorbar(ir.estimate,np.arange(len(ir)),xerr=np.vstack([ir.estimate-ir.ci_low,ir.ci_high-ir.estimate]),fmt='o',color='#247d91',capsize=3)
    axes[2].set(title='Eight lowest feature ICC estimates',yticks=np.arange(len(ir)),yticklabels=ir.feature,xlabel='ICC(A,1); patient-cluster 95% CI')
    fig.savefig(OUT/'supplement_overview.png',dpi=160);fig.savefig(OUT/'supplement_overview.pdf');plt.close(fig)
    status={'status':'complete','bootstrap_replicates':B,'elapsed_seconds':time.time()-START,'execution_environment':'Kaggle' if Path('/kaggle/input').exists() else 'local implementation check','new_training':False,'historical_per_split_csv_recovered':False,'v11_upstream_recovered':True,'n_icc_features':len(iccs),'n_calibration_estimates':len(cal_rows)}
    (OUT/'completion.json').write_text(json.dumps(status,indent=2))
    log(json.dumps(status));(OUT/'run.log').write_text('\n'.join(LOG))
    manifest=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    save(manifest,'output_manifest')
    archive=shutil.make_archive(str(OUT),'zip',OUT)
    log('RESULT_ARCHIVE '+archive)
    return status

if __name__=='__main__': main()
