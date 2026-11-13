# Oracle bone identity-response retrieval

Reproduction code for **Compact Identity-Response Representations for Oracle Bone Character Image Retrieval**, by Jie Zhou, Zhonglin Lu and Junfeng Bi. Corresponding author: Jie Zhou. The accompanying manuscript is not yet published.

[中文说明](README_zh.md) · [MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

The method maps a calibrated 640-D visual descriptor to complete responses over training-supported identities. OBC306 uses 277 response coordinates. Query images are encoded independently, and galleries can be precomputed. Class prototypes, softmax and square-root probability embeddings are established operations; this repository implements their evaluated oracle-bone retrieval adaptation.

## Quick start without images or model files

Use Python 3.10–3.13 in a separate environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s feature_level/tests -v
python -m unittest discover -s tests -v
python feature_level/reproduce.py table-summary --output table_summary.json
python examples/synthetic_demo.py --output demo_output
```

The synthetic demo creates artificial feature vectors to exercise Train-only geometry, prototype construction and independent gallery/query matching. Its output is a functional example, not a manuscript result. The table-summary command reconstructs the three-seed matched-readout summary from the released numerical CSVs.

## Frozen image-to-retrieval inference

Install the appropriate PyTorch 2.6 build for your CPU/GPU, then install `requirements-images.txt`. The image implementation is validated separately from the light feature-only tests. Runtime records are in `validation_reference/ARCHIVED_VALIDATION.json`.

Extract the accompanying **Online Resource 4** (`ESM_4.zip` / `frozen_assets_and_manifests.zip`) so `assets/` and `benchmark_manifests/` sit beside `reproduce_images.py`. The root asset manifest specifies the required file hashes. Model weights and original images are distributed separately from Git source; there is no model download URL in this source-only release. Obtain OBC306 and HWOBC images from their original providers.

```bash
python tools/verify_assets.py --assets assets
python reproduce_images.py encode --assets assets --image-root /your/OBC/images --manifest benchmark_manifests/OBC_test.json --vocabulary obc --device cpu --output outputs/OBC_Test
python reproduce_images.py evaluate --gallery outputs/OBC_Test --readout visual --output outputs/OBC_visual_metrics
python reproduce_images.py evaluate --gallery outputs/OBC_Test --readout identity --output outputs/OBC_identity_metrics
```

Use `--device cuda:0` for a suitable GPU. Complete-gallery evaluation has quadratic distance work; start with the synthetic demo and a small custom image manifest to check your environment. Existing output directories are preserved rather than overwritten.

For HWOBC, use `--vocabulary hwobc`, `benchmark_manifests/HWOBC_test.json`, and the provider directory containing `images/`. All visual/geometric assets remain fixed; only the target Train prototypes define the 3,881 response coordinates. This target setting tests adaptation, without OBC's dimensional compression.

## New queries

A custom manifest is a UTF-8 JSON list. True query identities are not needed:

```json
[{"path": "query.png", "key": "query_001"}]
```

```bash
python reproduce_images.py encode --assets assets --image-root /your/query/images --manifest queries.json --vocabulary obc --device cpu --output outputs/query_vectors
python reproduce_images.py retrieve --queries outputs/query_vectors --gallery outputs/OBC_Test --top-k 100 --output outputs/candidates.json
```

Paths must stay inside image-root. Benchmark manifests include image SHA-256 checks; these are optional for custom queries. Matching requires the same model, calibrated map and prototype order; fingerprints reject incompatible mappings. Self exclusion uses equal image keys. Scientific evaluation obtains labels only from separate class-prefixed benchmark keys, not from the image encoder.

## Reported results

| Dataset / readout | Gallery images | Query mAP (%) | Class mAP (%) |
| --- | ---: | ---: | ---: |
| OBC306 calibrated visual, seed 42 | 61,736 | 83.4749 | 58.8175 |
| OBC306 identity response, seed 42 | 61,736 | 87.5575 | 71.2085 |
| HWOBC visual, frozen visual assets | 18,706 | 68.2909 | 68.3358 |
| HWOBC target-Train identity response | 18,706 | 88.5306 | 88.5404 |

Across three auxiliary-adaptation seeds sharing one global encoder, the matched Query/Class mAP gains average 4.0866/12.6313 percentage points (sample SD 0.0222/0.2121). These seeds differ from the three complete PA baseline training runs. The complete-checkpoint PA controls and numerical evidence retain their distinct comparison roles.

## Evaluation and numerical fidelity

The evaluator re-normalizes float32 descriptors in float64, excludes the query itself, keeps singleton images as negative candidates, and omits queries with no other positive. Query mAP averages query AP; Class mAP averages class means. Exact equal-distance groups use expected AP and Hit@1. The deterministic path-hash order is used for display metrics and candidate lists.

Image forwards use batches of eight, padding the final short batch. The OBC replay retains its archived finite-precision route: FP32 prototype products, temperature-0.05 probabilities, and L2-normalized probability vectors. This is algebraically equivalent to the square-root map at temperature 0.025 for the same responses. HWOBC uses the direct temperature-0.025 square-root route in response batches of 256. These numeric routes remain unchanged from the validated inference package.

The archived GPU probe checks cover 32 OBC Val images and eight HWOBC Val images. Maximum OBC visual/identity errors are 4.47e-08/1.79e-06; HWOBC errors are zero. A fresh complete-Test pixel replay on every supported environment is not claimed. CPU/GPU differences and exact tested versions remain in the validation record.

## Repository scope

`reproduce_images.py` provides frozen image encoding, stored-gallery retrieval and evaluation. `feature_level/` provides Train-only geometry, normalized class means, identity response construction and summary reconstruction. `examples/` and `tests/` run without original images or model weights. `vendor/CFCD/` retains the pinned original implementation and notices.

This release covers frozen inference and evaluation. Full encoder and baseline training pipelines, original dataset images, private manuscripts, server helpers and authentication data are outside the repository. The paper's model states and full benchmark manifests are supplied separately in Online Resource 4.


## Query perturbation study

The fixed development-Val study in Section 5.7 uses 3,345 queries from 258 identities against an unchanged clean gallery of 30,810 images. All seven conditions and four readouts are retained. [Study instructions](robustness/README.md) provide the perturbation definitions, an image-inference entry point, and the numerical evidence behind Figure 6 and Table S23.

The released per-query metrics reproduce all 28 aggregate results and the plotted pointwise intervals without images or model files:

```bash
python -m robustness.replay --output robustness_replay.json
```

This source release is version **v0.2.0**. Frozen model files and full image manifests remain in Online Resource 4; original images are obtained from the dataset providers.

## Citation

See `CITATION.cff`. The manuscript is unpublished; no acceptance, DOI or publication venue is asserted. Project code is MIT-licensed; third-party software keeps its original notices.
