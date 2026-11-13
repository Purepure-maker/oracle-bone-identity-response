# Fixed-query Val perturbations

This study corresponds to Section 5.7, Figure 6 and Supplement R/Table S23. It changes query inputs while retaining one frozen seed-42 mapping and the complete clean Val gallery. The panel contains 3,345 queries from 258 identities; the gallery contains 30,810 images.

## Reconstruct the released results

From the repository root, install `requirements.txt` and run:

```bash
python -m robustness.replay --output robustness_replay.json
```

No image files or weights are needed. The command reads all 93,660 per-query records, reconstructs all 28 query/class AP and Hit@1 aggregates, checks per-class means, and independently reconstructs the plotted pointwise intervals. AP/Hit1 in the compressed query and class tables are fractions; `Class_mAP_and_CI.csv` uses percent. Expected maximum aggregate discrepancy is below 1e-12; the archived export reconstructs exactly.

The shaded Figure 6 bands use 2,000 identity-bootstrap draws (seed 20261113) conditional on the fixed model, queries and gallery. The panel scores differ from full-query Val scores. Numerical replay checks released evidence; it does not run the image model.

## Run image inference

Obtain the original OBC306 images from the provider and extract the matching frozen assets and `benchmark_manifests/OBC_val.json` from Online Resource 4. Install PyTorch 2.6 and `requirements-images.txt` as described in the root README.

```bash
python reproduce_images.py encode --assets assets --image-root /your/OBC/images --manifest benchmark_manifests/OBC_val.json --vocabulary obc --device cpu --output outputs/OBC_Val_clean
python -m robustness.run_study --assets assets --image-root /your/OBC/images --manifest benchmark_manifests/OBC_val.json --gallery outputs/OBC_Val_clean --device cpu --output outputs/query_perturbations
```

Select `--device cuda:0` to use a suitable GPU. The clean gallery is encoded once. The study checks asset and descriptor hashes, mapping compatibility, the fixed query selection, and original-image hashes. Clean query/gallery parity must pass before the first ranking. Output directories are never overwritten. Query origins are excluded by their original keys after transformation. Four readouts are evaluated: global 512D, raw auxiliary fusion 640D, calibrated visual 640D, and identity response 277D.

The seven fixed conditions are clean input; native width/height scales 0.75 and 0.50 (area interpolation); Gaussian blur at sigma 0.35 and 0.70 native pixels; and 10%/20% additional left and top margins filled by the native border median. Margin changes retain all original pixels and alter placement/context. Each transformation precedes the original encoder preprocessing. All severities are fixed in `protocol.json`.

The configurable image runner uses the unchanged public frozen encoder. The archived experiment used the protected historical clean-gallery caches; fresh image forwards may differ at floating-point precision. The public numerical tables preserve the measured values used in the paper. No training or Test evaluation is performed by this study.

## Files

- `corruptions.py`: exact native-image transformations from the recorded study
- `rectangular_evaluation.py`: transformed-query versus clean-gallery ranking with self exclusion and exact-tie expectations
- `run_study.py`: configurable frozen image-inference and evaluation entry point
- `replay.py`: reconstruction from the public numerical tables
- `evidence/`: fixed query panel, complete per-query/per-class AP/Hit1, summary, plotted intervals and source hashes

The original images and frozen model files are obtained separately. No server configuration or authentication data is included.
