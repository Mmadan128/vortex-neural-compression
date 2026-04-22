# CMS Experiment

This folder provides a CMS dataset flow aligned with ATLAS/HEPMC experiment layout.

It wraps `experiments/cms_experiment_lg/download.py` but defaults outputs to
`experiments/cms_experiment/data/` so training paths match this folder's config.

## Prepare data

From repository root:

```bash
# Atlas-style default (direct OpenData download + create-bin path)
python experiments/cms_experiment/download.py

# explicit URL + custom event cap
python experiments/cms_experiment/download.py \
    --url https://opendata.cern.ch/record/30525/files/CMS_Run2016G_JetHT_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_260000_file_index.json_12 \
    --create-bin \
    --out-dir experiments/cms_experiment/data \
    --bin-out cms_events.bin \
    --nmax 1000000
```

Expected outputs in `experiments/cms_experiment/data/`:

- `cms.root`
- `cms_events.bin`
- `cms_events.meta.json`
- `cms_events_train.bin`
- `cms_events_val.bin`
- `cms_events_test.bin`

## Train

```bash
python scripts/train.py --config experiments/cms_experiment/cms_experiment.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
    --model experiments/cms_experiment/checkpoints/best.pt \
    --data experiments/cms_experiment/data/cms_events_test.bin \
    --config experiments/cms_experiment/cms_experiment.yaml \
    --device cuda \
    --batch-size 256
```
