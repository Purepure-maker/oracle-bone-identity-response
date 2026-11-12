"""Frozen image-to-descriptor inference with explicit, verified assets."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import cv2,torch
from encoder_configuration import configure
from core import builders
from relation import FusionModel
from preprocessing import CenterCrop
from canvas import fit_canvas
from calibration import fixed_global_features

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def state_hash(model):
 h=hashlib.sha256()
 for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class FrozenEncoder:
 def __init__(self,assets,device='cpu',vocabulary='obc'):
  self.assets=Path(assets);self.device=torch.device(device);self.vocabulary=vocabulary
  if vocabulary not in ['obc','hwobc']:raise ValueError('Choose obc or hwobc supported vocabulary')
  self.manifest=json.loads((self.assets/'MANIFEST.json').read_text())
  names=['global_s1_state.pt','auxiliary_s42_state.pt','geometry_s42.npz','prototypes_'+('obc_s42' if vocabulary=='obc' else 'hwobc_train')+'.npz']
  for n in names:
   if sha(self.assets/n)!=self.manifest['assets'][n]['sha256']:raise ValueError('Frozen asset hash mismatch: '+n)
  torch.set_num_threads(1);cv2.setNumThreads(1)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
  c=json.loads((Path(__file__).resolve().parent/'model_settings.json').read_text())
  configure(SimpleNamespace(seed=42,image_size=512,batch_size=16,lr=.000625,epochs=50,val_period=5,workers=0,s1_checkpoint=self.assets/'global_s1_state.pt',output=self.assets))
  teacher=builders.build_algo().to(self.device)
  saved=torch.load(self.assets/'global_s1_state.pt',map_location='cpu',weights_only=True);teacher.load_state_dict(saved['model'],strict=True);teacher.eval().requires_grad_(False);self.global_class_order=sorted(saved['label_to_index']);del saved
  self.teacher=teacher;self.model=FusionModel(teacher,c,'F0').to(self.device).eval().requires_grad_(False)
  saved=torch.load(self.assets/'auxiliary_s42_state.pt',map_location='cpu',weights_only=True);self.model.load_branch(saved['branch']);del saved
  with np.load(self.assets/'geometry_s42.npz') as f:self.center=f['center'].copy();self.matrix=f['GLOBAL_FIXED'].copy()
  with np.load(self.assets/names[-1]) as f:
   self.prototypes=f['means'].copy();self.class_order=f['class_order'].tolist() if 'class_order' in f else self.global_class_order
  assert self.prototypes.shape==(len(self.class_order),640)
  self.crop=CenterCrop();self.initial_state=state_hash(self.model)
  fingerprint=hashlib.sha256(self.initial_state.encode())
  for value in [self.center,self.matrix,self.prototypes]:
   a=np.ascontiguousarray(value);fingerprint.update(str(a.dtype).encode());fingerprint.update(str(a.shape).encode());fingerprint.update(a.tobytes())
  fingerprint.update(json.dumps({'vocabulary':vocabulary,'class_order':self.class_order},sort_keys=True).encode())
  for name in ['image_encoder.py','preprocessing.py','relation.py','canvas.py','calibration.py','encoder_configuration.py','model_settings.json']:
   fingerprint.update(name.encode());fingerprint.update(sha(Path(__file__).resolve().parent/name).encode())
  self.mapping_sha256=fingerprint.hexdigest()

 def prepare(self,data):
  image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
  if image is None:raise ValueError('Unable to decode image')
  full,_,_=fit_canvas(image,512)
  return self.crop._prepare(image),self.crop._prepare(full)

 @torch.inference_mode()
 def encode_bytes(self,images):
  if not 1<=len(images)<=8:raise ValueError('Frozen forward batches contain 1 to 8 images')
  views=[self.prepare(b) for b in images];n=len(views)
  xc=torch.stack([x[0] for x in views]).to(self.device);xf=torch.stack([x[1] for x in views]).to(self.device)
  if n<8:
   xc=torch.cat([xc,xc[-1:].expand(8-n,-1,-1,-1)]);xf=torch.cat([xf,xf[-1:].expand(8-n,-1,-1,-1)])
  g=torch.nn.functional.normalize(self.teacher(xc)['global_feature'],dim=1)
  # F0 ignores keys. No true query identity or candidate image is supplied.
  _,_,local=self.model(xf,['image']*8)
  raw=torch.cat([np.sqrt(.75)*g[:n],.5*local[:n]],1).cpu().numpy()
  return raw,fixed_global_features(raw,self.center,self.matrix)

 def responses(self,geo):
  """Retain the recorded numeric route for each published frozen vocabulary."""
  if self.vocabulary=='obc':
   logits=np.asarray(geo,dtype=np.float32)@np.asarray(self.prototypes,dtype=np.float32).T/.05
   logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
   base=np.sqrt(p).astype(np.float32);z=base.astype(np.float64)**2;z/=np.linalg.norm(z,axis=1,keepdims=True)
   return z.astype(np.float32)
  logits=np.asarray(geo,dtype=np.float32)@np.asarray(self.prototypes,dtype=np.float32).T/.025
  logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
  return np.sqrt(p).astype(np.float32)

 def verify_unchanged(self):
  if state_hash(self.model)!=self.initial_state:raise RuntimeError('Frozen model state changed during inference')
