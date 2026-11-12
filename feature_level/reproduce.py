"""Portable post-encoder reproduction entry point for Paper A."""
from pathlib import Path
import argparse,csv,json,statistics
import numpy as np
from threadpoolctl import threadpool_limits
from geometry import statistics as geometry_statistics,whitening,fixed_global_features
from evaluation import evaluate

def normalize(x):
    z=np.asarray(x,dtype=np.float64)
    n=np.linalg.norm(z,axis=1,keepdims=True)
    if not np.isfinite(z).all() or np.any(n==0):raise ValueError('Expected finite nonzero feature rows')
    return z/n

def prototypes(features,labels):
    z=np.asarray(features,dtype=np.float32);labels=np.asarray(labels,dtype=str)
    if len(z)!=len(labels) or z.ndim!=2 or not np.isfinite(z).all():raise ValueError('Support feature/label mismatch')
    classes,inverse=np.unique(labels,return_inverse=True);counts=np.bincount(inverse,minlength=len(classes))
    sums=np.zeros((len(classes),z.shape[1]),dtype=np.float32)
    np.add.at(sums,inverse,z)
    means=normalize(sums/counts[:,None]).astype(np.float32)
    return means,classes,counts

def responses(features,means,temperature=.025,chunk=256):
    if temperature<=0:raise ValueError('Temperature must be positive')
    z=np.asarray(features,dtype=np.float32);p=np.asarray(means,dtype=np.float32)
    if z.ndim!=2 or p.ndim!=2 or z.shape[1]!=p.shape[1] or not np.isfinite(z).all() or not np.isfinite(p).all():raise ValueError('Feature/prototype dimensions or values invalid')
    result=np.empty((len(z),len(p)),dtype=np.float32)
    for start in range(0,len(z),chunk):
        logits=z[start:start+chunk]@p.T/temperature
        logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
        result[start:start+chunk]=np.sqrt(prob)
    return result

def summary(folder):
    def rows(name):
        with (folder/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    pairs=rows('test_three_seed_paired_readouts.csv');out={}
    for key in ['Query_delta_pp','Class_delta_pp','identity_Query_mAP_pct','identity_Class_mAP_pct']:
        v=[float(x[key]) for x in pairs];out[key]={'mean':statistics.mean(v),'sample_SD':statistics.stdev(v)}
    supports=rows('train_all277_support_counts.csv')
    out['supports']={'identities':len(supports),'images':sum(int(x['prototype_support_images']) for x in supports)}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for cmd in ['fit-geometry','calibrate','prototypes','responses','evaluate']:
        q=sub.add_parser(cmd);q.add_argument('--features',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
        if cmd in ['fit-geometry','prototypes']:q.add_argument('--labels',type=Path,required=True)
        if cmd=='calibrate':q.add_argument('--geometry',type=Path,required=True)
        if cmd=='responses':q.add_argument('--prototypes',type=Path,required=True);q.add_argument('--temperature',type=float,default=.025)
        if cmd=='evaluate':q.add_argument('--paths',type=Path,required=True);q.add_argument('--workers',type=int,default=1)
    q=sub.add_parser('table-summary');q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing outputs: '+str(a.output))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if a.command=='table-summary':a.output.write_text(json.dumps(summary(Path(__file__).parent/'evidence'),indent=2),encoding='utf8');return
        x=np.load(a.features,mmap_mode='r',allow_pickle=False)
        if a.command in ['fit-geometry','prototypes']:labels=json.loads(a.labels.read_text(encoding='utf8'))
        if a.command=='fit-geometry':
            if x.shape[1]!=640 or len(labels)!=len(x):raise ValueError('Expected raw weighted 640-D statistics and row-aligned labels')
            center,cov,_,_=geometry_statistics(x,labels);cov[:512,512:]=0;cov[512:,:512]=0;w,_=whitening(cov)
            np.savez(a.output,center=center,GLOBAL_FIXED=w)
        elif a.command=='calibrate':
            with np.load(a.geometry,allow_pickle=False) as f:result=fixed_global_features(x,f['center'],f['GLOBAL_FIXED'])
            np.save(a.output,result)
        elif a.command=='prototypes':
            means,order,counts=prototypes(x,labels);np.savez(a.output,means=means,class_order=order,counts=counts)
        elif a.command=='responses':
            with np.load(a.prototypes,allow_pickle=False) as f:result=responses(x,f['means'],a.temperature)
            np.save(a.output,result)
        elif a.command=='evaluate':
            paths=json.loads(a.paths.read_text(encoding='utf8'));evaluate(x,paths,a.output,workers=a.workers)

if __name__=='__main__':main()
