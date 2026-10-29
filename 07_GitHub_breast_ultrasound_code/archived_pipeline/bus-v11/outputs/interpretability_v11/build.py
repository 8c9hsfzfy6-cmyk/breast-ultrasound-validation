from pathlib import Path
import io,json,zipfile,hashlib,importlib.util,shutil
import numpy as np,pandas as pd,joblib
from PIL import Image
from scipy.special import expit,logit
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path.cwd();DATA=ROOT/'BUS-BRA结果';OUT=ROOT/'outputs/interpretability_v11'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def package(stem):return zipfile.ZipFile(next(DATA.rglob(stem)))
def read(z,name):
 ns=[n for n in z.namelist() if n==name or n.endswith('/'+name)]
 assert len(ns)==1,(name,len(ns))
 return z.read(ns[0])
def table(z,name):return pd.read_csv(io.BytesIO(read(z,name)),dtype={'uid':str,'image_id':str,'id':str,'cluster_id':str,'patient_key':str},float_precision='round_trip')
f=package('bus_v6_features_20261013_120420_94529.zip')
t=package('bus_v6_external_training_20261013_121259_911015.zip')
u=package('bus_uclm_external_20261013_132503_790776.zip')
v8=zipfile.ZipFile('/Users/xueyue/Downloads/bus_v8_model_comparison_20261014_141357_851028.zip')
pred=table(v8,'predictions.csv')
mb=read(t,'models/base_LR.joblib');assert hashlib.sha256(mb).hexdigest()==json.loads(read(t,'model_hashes.json'))['base_LR.joblib']
b=joblib.load(io.BytesIO(mb));pipe=b['models']['LR'];cal=b['calibrator'];cols=b['feature_columns']
a=float(cal.coef_[0,0]);c=float(cal.intercept_[0]);baseline=a*float(pipe[-1].intercept_[0])+c
GROUPS={'Shape':[j for j,k in enumerate(cols) if k.startswith('shape__')],
 'Core':[j for j,k in enumerate(cols) if k.startswith('core__')],
 'Boundary':[j for j,k in enumerate(cols) if k.startswith('rim__')],
 'Boundary-core contrast':[j for j,k in enumerate(cols) if k.startswith('rim_core__')]}
assert sorted(sum(GROUPS.values(),[]))==list(range(28))
# Protocol saved before selecting or inspecting representative images.
protocol={'analysis':'Frozen LR exact calibrated-logit attribution; no model fitting',
 'reference':'Zero standardized features (training means after imputation), not a healthy patient',
 'formula':'logit(p)=a*(intercept+sum(beta*z))+c+clipping_correction',
 'selection':'Lower median LR probability in each category, tie by ID; remove already-selected cases for later categories',
 'categories':['TN','TP','FP','FN','LR correct / TabICLv2 wrong','LR wrong / TabICLv2 correct'],
 'group_size_warning':'Groups contain different feature counts. Absolute attribution mass is not an intrinsic biological importance measure.',
 'population':'BrEaST 252 cases; BUS-UCLM 207 images/33 clusters. Both image-weighted and cluster-equal summaries exported.',
 'scope':'All cases in these two external evaluation cohorts; not all public-dataset images or source patients',
 'interpretation':'Model accounting, not causal biology. Correlated predictors share contribution. Region overlays are not saliency maps.',
 'model_sha256':hashlib.sha256(mb).hexdigest(),'refit':False}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
extractor_path=OUT/'v6_extractor.py';extractor_path.write_bytes(read(f,'v6_extractor.py'))
spec=importlib.util.spec_from_file_location('ex',extractor_path);ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
images={'BrEaST':package('BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip'),
        'BUS-UCLM':package('BUS-UCLM Breast ultrasound lesion segmentation dataset.zip')}
def image_data(ds,row):
 z=images[ds]
 # UCLM image and mask filenames can be identical: resolve folder explicitly.
 if ds=='BUS-UCLM':
  def exact(kind):
   name=Path(row[kind+'_path']).name
   ns=[n for n in z.namelist() if n.endswith('/'+('images' if kind=='image' else 'masks')+'/'+name)]
   assert len(ns)==1
   return z.read(ns[0])
  ib,mb=exact('image'),exact('mask')
  ih,mh=row.image_sha256,row.mask_sha256
 else:
  ib=read(z,Path(row.image_path).name);mb=read(z,Path(row.mask_path).name);ih,mh=row.expected_image_sha,row.expected_mask_sha
 assert hashlib.sha256(ib).hexdigest()==ih
 assert hashlib.sha256(mb).hexdigest()==mh
 im=Image.open(io.BytesIO(ib));mi=Image.open(io.BytesIO(mb))
 gray=np.asarray(im.convert('L'));valid=np.asarray(im.convert('RGBA'))[:,:,3]>0
 if ds=='BUS-UCLM':
  rgb=np.asarray(mi.convert('RGB'));mask=((rgb==[0,255,0]).all(2)|(rgb==[255,0,0]).all(2))&(np.asarray(mi.convert('RGBA'))[:,:,3]>0)
 else:mask=np.asarray(mi.convert('L'))>0
 scale=192/max(gray.shape);size=(round(gray.shape[1]*scale),round(gray.shape[0]*scale))
 x=np.asarray(Image.fromarray(gray.astype('float32')/255).resize(size,Image.Resampling.BILINEAR))
 def resize(v):return np.asarray(Image.fromarray(v.astype('uint8')).resize(size,Image.Resampling.NEAREST))>0
 mask,valid=resize(mask),resize(valid);vals,_=ex.extract_features(x,mask,valid)
 return x,mask,vals
all_summaries=[];validations=[];selected_records=[]
for ds in ['BrEaST','BUS-UCLM']:
 if ds=='BrEaST':
  co=table(f,'cohort.csv');co=co[co.dataset.eq(ds)]
  frame=co.merge(table(f,'features.csv'),on='uid',validate='one_to_one')
  frame['id']=frame.patient_key
 else:
  frame=table(u,'cohort_frozen.csv').merge(table(u,'features.csv'),on='image_id',validate='one_to_one')
  frame['id']=frame.image_id
 frame=frame.set_index('id').sort_index()
 ref=pred[pred.dataset.eq(ds)&pred.model.eq('LR')].set_index('id').loc[frame.index]
 tab=pred[pred.dataset.eq(ds)&pred.model.eq('TabICLv2')].set_index('id').loc[frame.index]
 assert len(ref)==len(frame) and np.array_equal(ref.label,frame.label)
 X=frame[cols];Z=pipe[:-1].transform(X);lin=pipe[-1].decision_function(Z)
 C=Z*pipe[-1].coef_[0]*a
 raw=pipe.predict_proba(X)[:,1];correction=a*(logit(np.clip(raw,1e-6,1-1e-6))-lin)
 p=expit(baseline+C.sum(1)+correction)
 err=float(np.max(np.abs(p-ref.p)));assert err<1e-8,err
 contribution=pd.DataFrame(C,index=frame.index,columns=cols)
 contribution.to_csv(OUT/(ds+'_all_feature_contributions.csv'))
 group=pd.DataFrame({name:C[:,idx].sum(1) for name,idx in GROUPS.items()},index=frame.index)
 mass=pd.DataFrame({name:np.abs(C[:,idx]).sum(1) for name,idx in GROUPS.items()},index=frame.index)
 details=ref[['cluster_id','label','p','threshold','prediction']].copy()
 for name in GROUPS:
  details[name+'_signed']=group[name]
  details[name+'_absolute_mass']=mass[name]
  details[name+'_fraction']=mass[name]/mass.sum(axis=1).replace(0,np.nan)
 details['dominant_group']=mass.idxmax(axis=1)
 details['baseline']=baseline;details['clipping_correction']=correction
 details.to_csv(OUT/(ds+'_all_group_contributions.csv'))
 weights=1/ref.cluster_id.map(ref.cluster_id.value_counts()).to_numpy();weights/=weights.sum()
 for j,feature in enumerate(cols):
  all_summaries.append({'dataset':ds,'feature':feature,'image_mean_abs':np.abs(C[:,j]).mean(),
      'cluster_equal_mean_abs':np.sum(weights*np.abs(C[:,j])),'positive_fraction':(C[:,j]>0).mean()})
 top=np.argsort(np.sum(weights[:,None]*np.abs(C),axis=0))[-10:]
 fig,ax=plt.subplots(figsize=(8,5));ax.barh(np.array(cols)[top],np.sum(weights[:,None]*np.abs(C),axis=0)[top],color='#287da5')
 ax.set(xlabel='Mean absolute calibrated log-odds contribution\n(equal total weight per patient cluster)',title=ds+' — frozen LR')
 fig.tight_layout();fig.savefig(OUT/(ds+'_global_contributions.png'),dpi=220);fig.savefig(OUT/(ds+'_global_contributions.pdf'));plt.close(fig)
 correct=ref.prediction.eq(ref.label);tcorrect=tab.prediction.eq(ref.label)
 cats=[('TN',ref.label.eq(0)&correct),('TP',ref.label.eq(1)&correct),
       ('FP',ref.label.eq(0)&~correct),('FN',ref.label.eq(1)&~correct),
       ('LR correct / TabICLv2 wrong',correct&~tcorrect),('LR wrong / TabICLv2 correct',~correct&tcorrect)]
 used=set();chosen=[]
 for category,mask in cats:
  g=ref.loc[mask & ~ref.index.isin(used)].reset_index().sort_values(['p','id'])
  if g.empty:continue
  case=g.iloc[(len(g)-1)//2]['id'];used.add(case);chosen.append((category,case,len(g)))
 for page in range(0,len(chosen),3):
  subset=chosen[page:page+3];fig,axes=plt.subplots(len(subset),2,figsize=(12,4.4*len(subset)),squeeze=False,layout='constrained')
  for (category,case,n),ax in zip(subset,axes):
   x,m,vals=image_data(ds,frame.loc[case])
   for k,val in vals.items():assert np.isclose(val,frame.loc[case,k],atol=1e-6,equal_nan=True),(ds,case,k)
   thickness=max(1,round(np.sqrt(m.sum())*.06));core=ndi.binary_erosion(m,iterations=thickness)
   if core.sum()<9:core=m
   ax[0].imshow(x,cmap='gray',vmin=0,vmax=1)
   for region,color in [(ndi.binary_dilation(m,iterations=thickness),'#ffa726'),(m,'white'),(core,'#33d1a0')]:
    ax[0].contour(region,levels=[.5],colors=[color],linewidths=.8)
   ax[0].axis('off');ax[0].set_title(f'{category}\n{case}; label={int(ref.loc[case,"label"])}',fontsize=10)
   offset=baseline-logit(float(ref.loc[case,'threshold']))
   values=np.r_[offset,group.loc[case].values,details.loc[case,'clipping_correction']]
   labels=['Baseline minus threshold']+list(GROUPS)+['Numerical clipping correction']
   ax[1].barh(labels,values,color=['#777777']+['#c85c46' if v>0 else '#287da5' for v in values[1:]])
   ax[1].axvline(0,color='gray',lw=.8);ax[1].invert_yaxis();ax[1].set_xlabel('Contribution to LR decision margin (log-odds)')
   ax[1].set_title(f'LR p={ref.loc[case,"p"]:.3f}, t={ref.loc[case,"threshold"]:.3f}\nTabICLv2 p={tab.loc[case,"p"]:.3f}, t={tab.loc[case,"threshold"]:.3f}',fontsize=10)
   selected_records.append({'dataset':ds,'category':category,'id':case,'eligible_after_exclusion':n,
       'LR_p':ref.loc[case,'p'],'TabICLv2_p':tab.loc[case,'p'],'feature_reproduction':'all V6 values within 1e-6',
       'selection':'lower median LR p; tie by ID; exclude earlier cases'})
  fig.suptitle(ds+' — regions and LR contributions (not pixel saliency)',fontsize=13)
  fig.savefig(OUT/f'{ds}_cases_{page//3+1}.png',dpi=200);fig.savefig(OUT/f'{ds}_cases_{page//3+1}.pdf');plt.close(fig)
 validations.append({'dataset':ds,'n':len(frame),'clusters':ref.cluster_id.nunique(),'max_abs_prediction_error':err,
  'n_raw_probabilities_clipped':int(((raw<1e-6)|(raw>1-1e-6)).sum())})
pd.DataFrame(all_summaries).to_csv(OUT/'global_feature_summary.csv',index=False)
pd.DataFrame(selected_records).to_csv(OUT/'selected_case_provenance.csv',index=False)
(OUT/'validation.json').write_text(json.dumps(validations,indent=2))
print(json.dumps(validations,indent=2))
