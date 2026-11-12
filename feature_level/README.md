# Feature-level reproduction

Run from the repository root using the main README. These commands fit geometry on Train statistics, construct prototypes from labeled Train supports, and evaluate exported descriptors:

```bash
python feature_level/reproduce.py fit-geometry --features Train_statistics_RAW640.npy --labels statistics_labels.json --output geometry.npz
python feature_level/reproduce.py calibrate --features RAW640.npy --geometry geometry.npz --output GEO640.npy
python feature_level/reproduce.py prototypes --features Train_support_GEO640.npy --labels support_labels.json --output prototypes.npz
python feature_level/reproduce.py responses --features Gallery_GEO640.npy --prototypes prototypes.npz --output Gallery_ID.npy
python feature_level/reproduce.py evaluate --features Gallery_ID.npy --paths gallery_keys.json --output identity_metrics
```

Feature arrays are finite float32 (N,D), in manifest row order. The raw 640-D vector includes branch shares sqrt(0.75) and sqrt(0.25). Labels retain their original string codes. Evaluation keys use `class_code/image_key`; they are not image-encoder inputs. Statistics and supports must come from Train. OBC uses 21,546 fixed statistics images and 17,329 fixed supports; all 277 prototype coordinates are retained. Target HWOBC reconstruction uses 56,745 target Train images and preserves the visual/geometric map.

`evidence/` contains the numerical CSVs. `PROVENANCE.json` is an archived research record; source hashes are retained for files it describes, including records distributed in the paper's resources rather than this minimal repository. See the root RELEASE_MANIFEST.json for the current repository file inventory. Full image encoding additionally requires the separate frozen assets.
