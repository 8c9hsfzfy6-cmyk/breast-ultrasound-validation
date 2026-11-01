from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from sklearn.metrics import roc_auc_score
P=Path(__file__).parent; E=P/'evidence';O=P/'new_results';O.mkdir(exist_ok=True)
B=2000;SEED=20261031
protocol={'bootstrap':B,'seed':SEED,'analysis':'exploratory frozen-model measurable-case diagnostic comparison','primary_contrast':'base_posterior - base_missing','secondary_contrasts':['base_missing - base','base_posterior - base'],'BrEaST':'diagnosis-stratified case bootstrap','BUS-UCLM':'patient-cluster bootstrap; equal cluster weight normalized within measurable subset','interval':'percentile 95%; conditional on fitted models','training':False}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
f=pd.read_csv(next(E.glob('bus_v6_features*/features.csv')));c=pd.read_csv(next(E.glob('bus_v6_features*/cohort.csv')));x=c.merge(f,on='uid',validate='one_to_one');x=x[x.dataset=='BrEaST']
b=pd.read_csv(next(E.glob('bus_v6_external_training*/external_predictions.csv')));b['configuration']=b.feature_set+'_'+b.model;b=b.rename(columns={'patient_key':'id','p':'probability','prediction':'predicted_label'});b['cluster_id']=b.id
b=b.merge(x[['patient_key','postdepth__missing','label']],left_on='id',right_on='patient_key',validate='many_to_one',suffixes=('','_feature'));assert (b.label==b.label_feature).all()
u=pd.read_csv(next(E.glob('bus_uclm_external*/predictions_with_labels.csv'))).rename(columns={'image_id':'id'});uf=pd.read_csv(next(E.glob('bus_uclm_external*/features.csv')))
u=u.merge(uf[['image_id','postdepth__missing']],left_on='id',right_on='image_id',validate='many_to_one')
rows=[];diffs=[];draws={};included=[]
for ds,pr in [('BrEaST',b),('BUS-UCLM',u)]:
 pr=pr[pr['postdepth__missing']==0];ref=pr[pr.configuration=='base_LR'].sort_values('id');assert ref.id.is_unique
 ids=ref.id.to_numpy();y=ref.label.to_numpy();cl,iv=np.unique(ref.cluster_id,return_inverse=True);sz=np.bincount(iv);n=len(y)
 assert n==({'BrEaST':170,'BUS-UCLM':142}[ds]);rng=np.random.default_rng(SEED+(ds=='BUS-UCLM'))
 counts=np.zeros((B,len(cl)),int)
 for r in range(B):
  if ds=='BrEaST':
   for label in [0,1]:
    ix=np.flatnonzero(y==label);sample=rng.choice(ix,len(ix),replace=True);counts[r]+=np.bincount(iv[sample],minlength=len(cl))
  else:counts[r]=rng.multinomial(len(cl),np.ones(len(cl))/len(cl))
 np.savez_compressed(O/(ds+'_resamples.npz'),clusters=cl,counts=counts)
 for weighting in (['case_equal'] if ds=='BrEaST' else ['image_equal','patient_equal']):
  w=np.ones(n) if weighting!='patient_equal' else 1/sz[iv]
  for j in range(n):included.append(dict(dataset=ds,weighting=weighting,id=ids[j],cluster=ref.cluster_id.iloc[j],label=int(y[j]),weight=float(w[j])))
  pts={};boot={}
  for config in sorted(pr.configuration.unique()):
   g=pr[pr.configuration==config].set_index('id').loc[ids];assert np.array_equal(g.label,y) and np.array_equal(g.cluster_id,ref.cluster_id)
   p=g.probability.to_numpy();d=g.predicted_label.to_numpy();assert np.isfinite(p).all() and np.array_equal(d,p>=g.threshold.to_numpy())
   def metric(ww):
    if ww[y==0].sum()==0 or ww[y==1].sum()==0:return np.full(4,np.nan)
    return np.array([roc_auc_score(y,p,sample_weight=ww),np.average((y-p)**2,weights=ww),np.average(d[y==1],weights=ww[y==1]),np.average(1-d[y==0],weights=ww[y==0])])
   pts[config]=metric(w);boot[config]=np.array([metric(w*ct[iv]) for ct in counts]);draws[ds+'_'+weighting+'_'+config]=boot[config]
   for j,m in enumerate(['auc','brier','sensitivity','specificity']):
    v=boot[config][:,j];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]);rows.append(dict(dataset=ds,weighting=weighting,configuration=config,n=n,malignant=int(y.sum()),clusters=len(cl),metric=m,estimate=pts[config][j],ci_low=lo,ci_high=hi,n_valid=len(v)))
  for model in ['LR','ensemble']:
   for a,z in [('base_posterior','base_missing'),('base_missing','base'),('base_posterior','base')]:
    for j,m in enumerate(['auc','brier','sensitivity','specificity']):
     v=boot[a+'_'+model][:,j]-boot[z+'_'+model][:,j];v=v[np.isfinite(v)];lo,hi=np.quantile(v,[.025,.975]);diffs.append(dict(dataset=ds,weighting=weighting,model=model,comparison=a+'_minus_'+z,metric=m,estimate=pts[a+'_'+model][j]-pts[z+'_'+model][j],ci_low=lo,ci_high=hi,n_valid=len(v),n_failed=B-len(v)))
  print(ds,weighting,'complete',flush=True)
pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);pd.DataFrame(diffs).to_csv(O/'paired_increments.csv',index=False);pd.DataFrame(included).to_csv(O/'included_cases.csv',index=False);np.savez_compressed(O/'metric_draws.npz',**draws)
# Check original UCLM subgroup metrics independently against archived point estimates.
a=pd.read_csv(next(E.glob('bus_uclm_external*/descriptive_subgroups.csv')));a=a[a.subgroup=='posterior_measurable'];r=pd.DataFrame(rows)
for _,v in a.iterrows():
 for m in ['auc','brier','sensitivity','specificity']:
  got=r[(r.dataset=='BUS-UCLM')&(r.weighting=='image_equal')&(r.configuration==v.configuration)&(r.metric==m)].estimate.iloc[0];assert abs(got-v[m])<1e-12
(O/'verification.json').write_text(json.dumps({'archived_UCLM_metrics_match':True,'all_pairing_assertions_passed':True,'new_training':False,'bootstrap_replicates':B},indent=2))
