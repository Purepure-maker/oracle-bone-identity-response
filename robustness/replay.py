"""Reconstruct all fixed-panel metrics and Figure 6 intervals from public evidence."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import json
import numpy as np


def replay(folder):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    panel = json.loads((folder/'query_panel.json').read_text(encoding='utf-8'))
    assert summary['state'] == 'COMPLETE'
    assert summary['optimizer_updates'] == summary['new_Test_evaluations'] == 0
    keys = panel['keys']
    classes = sorted(panel['counts'])
    assert len(keys) == len(set(keys)) == 3345 and len(classes) == 258
    with gzip.open(folder/'per_query_metrics.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        groups = defaultdict(list)
        for r in csv.DictReader(f):
            groups[r['condition'], r['arm']].append(r)
    with (folder/'per_class_metrics.csv').open(encoding='utf-8', newline='') as f:
        class_reference = {(r['condition'],r['arm'],r['class_id']):r for r in csv.DictReader(f)}
    with (folder/'Class_mAP_and_CI.csv').open(encoding='utf-8-sig', newline='') as f:
        intervals = {(r['condition'],r['arm']):r for r in csv.DictReader(f)}
    draws = np.random.default_rng(20261113).integers(258, size=(2000,258))
    results = []
    error = 0.
    for r in summary['results']:
        name, arm = r['condition']['name'], r['arm']
        rows = groups[name,arm]
        assert [q['key'] for q in rows] == keys
        by_class = defaultdict(list)
        for q in rows:
            assert q['key'].split('/')[0] == q['class_id']
            by_class[q['class_id']].append(q)
        reconstructed = {}
        for metric in ('AP','Hit1','AP_hash','Hit1_hash'):
            values = np.array([np.mean([float(q[metric]) for q in by_class[c]]) for c in classes])
            for c, value in zip(classes,values):
                error = max(error, abs(value-float(class_reference[name,arm,c][metric])))
            reconstructed['PanelQuery_'+metric] = float(np.mean([float(q[metric]) for q in rows]))
            reconstructed['Class_'+metric] = float(values.mean())
        for metric,value in reconstructed.items():
            error = max(error, abs(value-r['metrics'][metric]))
        values = np.array([np.mean([float(q['AP']) for q in by_class[c]]) for c in classes])
        ci = np.quantile(values[draws].mean(1)*100,[.025,.975])
        reference = intervals[name,arm]
        assert np.allclose(ci,[float(reference['CI95_lower']),float(reference['CI95_upper'])],rtol=0,atol=1e-10)
        assert abs(values.mean()*100-float(reference['Class_mAP_pct'])) < 1e-10
        results.append({'condition':name,'arm':arm,**reconstructed,'Class_mAP_CI95_percent':ci.tolist()})
    assert len(results) == len(groups) == 28 and len(class_reference) == 28*258
    assert error < 1e-12
    return {'state':'PASS','comparisons':28,'query_count':3345,'class_count':258,
            'gallery_images':30810,'max_aggregate_error':error,'all_plotted_intervals_verified':True,
            'scope':'Numerical evidence replay; fixed trained model and development-Val query panel',
            'results':results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent/'evidence')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = replay(args.evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}))


if __name__ == '__main__':
    main()
