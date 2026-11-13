"""Apply the fixed native-image protocol and score against a clean stored gallery."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from .rectangular_evaluation import score

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def put(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def select_indices(keys,cap=16,salt='paper-a-robustness-val-v1|'):
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate image keys')
    selected = []
    counts = Counter(k.split('/')[0] for k in keys)
    for identity in sorted(c for c,n in counts.items() if n > 1):
        candidates = [i for i,k in enumerate(keys) if k.split('/')[0] == identity]
        candidates.sort(key=lambda i:hashlib.sha256((salt+keys[i]).encode()).digest())
        selected.extend(candidates[:cap])
    return np.array(sorted(selected),dtype=np.int64)


def checked_image(root,record):
    path = Path(record['path'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Manifest paths must stay inside image-root')
    resolved = (root/path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Image path escapes image-root')
    data = resolved.read_bytes()
    if not record.get('sha256') or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Image hash mismatch: '+str(path))
    return data


def run(args):
    # Import heavyweight dependencies only for actual image inference.
    import cv2
    from threadpoolctl import threadpool_limits
    from image_encoder import FrozenEncoder
    from .corruptions import apply
    protocol = read(Path(__file__).with_name('protocol.json'))
    gallery = args.gallery.resolve()
    root = args.image_root.resolve()
    info = read(gallery/'complete.json')
    if info.get('state') != 'COMPLETE' or info.get('vocabulary') != 'obc':
        raise ValueError('A complete OBC clean gallery is required')
    keys = read(gallery/'paths.json')
    manifest = read(args.manifest)
    if [r.get('key',r['path']) for r in manifest] != keys:
        raise ValueError('Manifest order differs from clean gallery')
    if len(keys) != 30810 or len({k.split('/')[0] for k in keys}) != 258:
        raise ValueError('This fixed protocol requires the Standard Val gallery')
    # The public encoder source fingerprint protects geometry and prototype compatibility.
    encoder = FrozenEncoder(args.assets,args.device,'obc')
    if info['model_state_sha256'] != encoder.initial_state or info['mapping_sha256'] != encoder.mapping_sha256:
        raise ValueError('Gallery and query encoder mappings differ')
    arrays = {}
    for arm,filename in [('raw_fusion640','RAW640.npy'),('calibrated_visual640','GEO640.npy'),('identity277','ID.npy')]:
        if sha(gallery/filename) != info['arrays_sha256'][filename]:
            raise ValueError('Gallery array hash mismatch: '+filename)
        arrays[arm] = np.load(gallery/filename,mmap_mode='r')
    arrays['global512'] = arrays['raw_fusion640'][:,:512]
    ids = select_indices(keys,protocol['queries']['cap_per_class'],protocol['queries']['salt'])
    query_keys = [keys[i] for i in ids]
    if query_keys != read(Path(__file__).with_name('evidence')/'query_panel.json')['keys']:
        raise ValueError('The fixed query panel does not match the released one')
    # Validate every selected original image before creating an output directory.
    for i in ids:
        checked_image(root,manifest[i])
    args.output.mkdir(parents=True,exist_ok=False)
    put(args.output/'query_panel.json',{'indices':ids.tolist(),'keys':query_keys,'gallery_images':len(keys)})
    results = []
    dimensions = {'global512':512,'raw_fusion640':640,'calibrated_visual640':640,'identity277':277}
    with threadpool_limits(limits=1):
        for condition in protocol['conditions']:
            print(json.dumps({'phase':'encoding','condition':condition['name']}),flush=True)
            features = {arm:np.empty((len(ids),d),np.float32) for arm,d in dimensions.items()}
            for first in range(0,len(ids),8):
                batch = []
                for i in ids[first:first+8]:
                    data = checked_image(root,manifest[i])
                    if condition['family'] != 'clean':
                        native = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                        if native is None:
                            raise ValueError('Unable to decode image')
                        ok,buffer = cv2.imencode('.png',apply(native,condition))
                        if not ok:
                            raise ValueError('Unable to encode transformed input')
                        data = buffer.tobytes()
                    batch.append(data)
                raw,geo = encoder.encode_bytes(batch)
                stop = first+len(batch)
                features['global512'][first:stop] = raw[:,:512]
                features['raw_fusion640'][first:stop] = raw
                features['calibrated_visual640'][first:stop] = geo
                features['identity277'][first:stop] = encoder.responses(geo)
            if condition['family'] == 'clean':
                parity = {arm:float(np.max(np.abs(features[arm]-arrays[arm][ids]))) for arm in dimensions}
                put(args.output/'clean_parity.json',parity)
                if max(parity.values()) >= 1e-5:
                    raise ValueError('Clean query/gallery parity check failed; scoring was not started')
            for arm in protocol['arms']:
                folder = args.output/condition['name']/arm
                folder.mkdir(parents=True)
                metrics,rows,classes,checks = score(features[arm],arrays[arm],query_keys,keys,workers=args.workers)
                for name,value in [('metrics',metrics),('per_query',rows),('per_class',classes),('independent_checks',checks)]:
                    put(folder/(name+'.json'),value)
                results.append({'condition':condition,'arm':arm,'metrics':metrics})
                print(json.dumps({'phase':'scored','condition':condition['name'],'arm':arm,'Class_AP':metrics['Class_AP']}),flush=True)
            encoder.verify_unchanged()
    put(args.output/'summary.json',{'state':'COMPLETE','results':results,'optimizer_updates':0,
                                  'new_Test_evaluations':0,'gallery_images':len(keys),
                                  'queries':len(query_keys),'protocol_sha256':sha(Path(__file__).with_name('protocol.json')),
                                  'scope':'Fresh pixel inference on the fixed development-Val panel'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'assets')
    parser.add_argument('--gallery',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        raise ValueError('workers must be positive')
    run(args)


if __name__ == '__main__':
    main()
