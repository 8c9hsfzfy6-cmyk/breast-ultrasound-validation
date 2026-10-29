from pathlib import Path
import pandas as pd,numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
root=Path(__file__).resolve().parent.parent/'r18_recovered/project';res=root/'results';out=Path(__file__).resolve().parent
p=next((root/'source_evidence/bus-compare').glob('bus_v6_external_training*/external_predictions.csv'))
d=pd.read_csv(p,float_precision='round_trip',dtype={'patient_key':str});d['configuration']=d.feature_set+'_'+d.model
b=np.load(res/'BrEaST_prediction_resamples.npz');draws=np.load(res/'metric_bootstrap_draws.npz');rows=[]
for r in pd.read_csv(res/'calibration_failed_replicates.csv').to_dict('records'):
 g=d[d.configuration.eq(r['configuration'])].sort_values('patient_key');assert np.array_equal(g.patient_key,b['clusters'])
 w=b['counts'][r['replicate']].astype(float);y=g.label.to_numpy();x=logit(np.clip(g.p.to_numpy(),1e-6,1-1e-6));X=np.c_[np.ones(len(x)),x];w/=w.sum()
 def fun(t):
  z=X@t;return np.sum(w*(np.logaddexp(0,z)-y*z))
 def jac(t):return X.T@(w*(expit(X@t)-y))
 opt=minimize(fun,[0,1],jac=jac,method='BFGS',options={'gtol':1e-10,'maxiter':2000})
 a=draws['BrEaST_image_equal_'+r['configuration']][:,5].copy();assert np.isnan(a[r['replicate']]);lo,hi=np.nanquantile(a,[.025,.975]);a[r['replicate']]=opt.x[1];lo2,hi2=np.quantile(a,[.025,.975])
 rows.append(dict(**r,independent_slope=opt.x[1],joint_intercept=opt.x[0],max_abs_normalized_score=max(abs(jac(opt.x))),optimizer_success=bool(opt.success),original_ci_low=lo,original_ci_high=hi,sensitivity_ci_low=lo2,sensitivity_ci_high=hi2,max_ci_endpoint_change=max(abs(lo-lo2),abs(hi-hi2))))
pd.DataFrame(rows).to_csv(out/'failed_calibration_independent_audit.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
