from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from sklearn.metrics import roc_auc_score,brier_score_loss
root=Path(__file__).resolve().parent
runs=list(root.glob('formal/bus_v6_stability_*'))
run=runs[0] if runs else root/'formal'
m=pd.read_csv(run/'metrics_by_split.csv');p=pd.read_csv(run/'external_predictions.csv');spl=pd.read_csv(run/'patient_splits.csv');th=pd.read_csv(run/'threshold_audit.csv')
assert len(m)==30 and set(m.split_seed)==set(range(42,47))
assert len(p)==7560 and not p.duplicated(['split_seed','patient_key','feature_set','model']).any()
assert spl.groupby(['split_seed','patient_key']).size().eq(1).all()
assert spl.groupby('split_seed').size().eq(1064).all()
assert len(th)==30 and (th.development_sensitivity>=.95).all()
assert json.load(open(run/'run_summary.json'))['seed42_reproduced']
assert pd.read_csv(run/'reproducibility_check.csv').passed.all()
for keys,g in p.groupby(['split_seed','feature_set','model']):
 r=m.set_index(['split_seed','feature_set','model']).loc[keys]
 assert len(g)==252 and g.patient_key.nunique()==252
 assert abs(roc_auc_score(g.label,g.p)-r.auc)<1e-12
 assert abs(brier_score_loss(g.label,g.p)-r.brier)<1e-12
 assert (((g.label==1)&(g.prediction==0)).sum()==r.missed_malignant)
 assert (((g.label==0)&(g.prediction==1)).sum()==r.false_positive)
for name,h in json.load(open(run/'model_hashes.json')).items():assert hashlib.sha256((run/'models'/name).read_bytes()).hexdigest()==h
hist=pd.read_csv(root.parent/'r18_recovered/project/results/historical_split_summary.csv')
new=pd.read_csv(run/'descriptive_stability.csv')
comp=new.merge(hist,on=['feature_set','model'],suffixes=('_new','_historical'))
rows=[]
for metric in ['auc','brier','sensitivity','specificity']:
 for stat in ['mean','std','min','max']:
  col=f'{metric}_{stat}';diff=abs(comp[col+'_new']-comp[col+'_historical'])
  rows.append({'column':col,'max_abs_diff_from_rounded_history':diff.max(),'consistent_with_4dp_rounding':bool((diff<=.000051).all())})
pd.DataFrame(rows).to_csv(root/'historical_rounding_comparison.csv',index=False)
checks={'status':'complete','five_seeds':list(range(42,47)),'metric_rows':len(m),'predictions':len(p),'patient_split_records':len(spl),'model_files':len(json.load(open(run/'model_hashes.json'))),'patient_disjoint_within_split':True,'metrics_independently_recomputed':True,'development_sensitivity_at_least_95percent':True,'model_hashes_verified':True,'history_4dp_agreement':all(r['consistent_with_4dp_rounding'] for r in rows)}
(root/'independent_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks));print(new.to_string(index=False));print(pd.read_csv(run/'paired_difference_summary.csv').to_string(index=False))
