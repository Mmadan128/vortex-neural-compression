# CMS Round-Trip Pipeline (ROOT ↔ bin) [Large (lg) Dataset]

This experiment downloads a CMS NANOAOD sample reference, extracts the first N events, encodes selected branches into a padded float32 matrix saved as a `.bin` buffer with a JSON sidecar, reconstructs a ROOT file from that buffer, and verifies round-trip equality against the original for those branches.

Note: We compare only branches that are primitives or lists of primitives (no nested records/objects), converted to float32. For list branches, per-event lengths are preserved, enabling exact reconstruction of jagged shapes.

## Data source

OpenData record URL (index JSON listing files):

- https://opendata.cern.ch/record/30525/files/CMS_Run2016G_JetHT_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_260000_file_index.json_12

The script resolves the first available ROOT file URL from this index (or uses a direct `.root` URL if you provide one), downloads it locally as `cms.root` under your `--out-dir`, and reuses it on subsequent runs (skipping download if the file already exists). If a `root://` XRootD URL is found, it tries an HTTPS mirror to avoid requiring XRootD client bindings. You can override the `--url` argument to choose a different index or file.

## Binary format

We write two artifacts:

- `<name>.bin`: raw float32 values of shape (N, total_columns), row-major, with per-branch padding to each branch's max length.
- `<name>.meta.json`: metadata required to reconstruct jagged arrays and column mapping.

Example schema (simplified):

```
{
  "n_events": 50000,
  "tree_key": "Events",
  "branches": [
    {"name": "Jet_pt", "is_list": true,  "max_len": 12, "dtype": "float32", "col_offset": 0},
    {"name": "Muon_pt", "is_list": true,  "max_len": 4,  "dtype": "float32", "col_offset": 12},
    {"name": "MET_pt",  "is_list": false, "max_len": 1,  "dtype": "float32", "col_offset": 16}
  ],
  "lengths": {
    "Jet_pt":  [L0, L1, ..., LN-1],
    "Muon_pt": [ ... ],
    "MET_pt":  []
  }
}
```

- `branches[*].col_offset` gives the start column in the flat matrix for each branch;
- list branches use `max_len` columns, scalars use 1 column;
- `lengths` holds per-event jagged lengths for list branches to reconstruct exact shapes.

## How to run

The script supports two paths controlled by flags. It will first ensure a local `cms.root` exists in `--out-dir` (downloading it if missing), then operate on that local file.

1) Create compressible binaries from the original ROOT — all events, float32 (default if no flags given):

```
python download.py \
  --url https://opendata.cern.ch/record/30525/files/CMS_Run2016G_JetHT_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_260000_file_index.json_12 \
  --out-dir ./data \
  --create-bin \
  --bin-out cms_large_padded.bin
```

On first run this will download `cms.root` into `./data/`, then write `cms_large_padded.bin` and `cms_large_padded.meta.json`, generate 80/10/10 train/val/test splits, reconstruct `cms_roundtrip.root`, and produce `compare_report.txt`.

Split outputs in `out-dir`:
- `cms_large_padded_train.bin` — 80% of events
- `cms_large_padded_val.bin` — 10% of events
- `cms_large_padded_test.bin` — 10% of events

2) Validate a compress-decompressed file from another algorithm against the original ROOT:

```
python download.py \
  --url https://opendata.cern.ch/record/30525/files/CMS_Run2016G_JetHT_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_260000_file_index.json_12 \
  --out-dir ./ \
  --validate-bin \
  --decompressed-bin cms_experiment_lg_decompressed.bin \
  --use-meta cms_50k_padded.meta.json \
  --fitted-root cms_fitted.root
```

On first run this will download `cms.root` into `./data/` if missing. It then reads the decompressed `.bin` and matching `.meta.json`, reconstructs `cms_fitted.root`, and produces `compare_report_validate.txt` comparing to the original ROOT (first `n_events` from meta; capped by the source file size).

Outputs in `out-dir`:

- `cms.root` – downloaded original CMS NANOAOD file (reused across runs)
- `cms_50k_padded.bin` – float32 matrix buffer (create path)
- `cms_50k_padded.meta.json` – metadata sidecar (create path)
- `cms_roundtrip.root` – reconstructed ROOT from created bin (create path)
- `compare_report.txt` – per-branch comparison for create path
- `cms_fitted.root` – reconstructed ROOT from external decompressed bin (validate path)
- `compare_report_validate.txt` – comparison for validate path

Exit code is non-zero if comparison fails.

## Reversibility and comparison

- Scalar branches are compared with `np.allclose` on float32 values.
- List branches are compared by verifying per-event lengths match exactly, then values up to each event's length.
- Any mismatch will be recorded in `compare_report.txt`.

## Dependencies

- `uproot>=5`, `awkward>=2`, `numpy`
- The repo’s top-level `requirements.txt` already includes `uproot`. If `awkward` isn’t available in your environment, install it:

```
pip install awkward
```

XRootD bindings are not required because the script prefers HTTPS mirrors for `root://` URLs. If you have XRootD and prefer it, pass the `root://…` URL directly.

## Notes on validation mode

- The `.meta.json` must correspond to the `.bin` you are validating. It contains the schema (selected branches, padding widths, and per-event lengths) needed to reconstruct jagged arrays. If you generated the `.bin` with this script, the matching `.meta.json` will be written alongside it.
- The script validates only primitive and list-of-primitive branches included in the metadata. It reconstructs and compares those branches against the original ROOT using `np.allclose` for float32 values and exact length matches for jagged arrays.

## Notes

- Only primitive and list-of-primitive branches are included, cast to float32. This matches the encoding used by the binary representation and ensures clean round-tripping.
- If you need byte-for-byte equality with original ROOT encodings or to include complex/nested branches, you’ll need a richer format to store types and nested schemas.

## Training (AMD MI300X)

After running `download.py` to generate the splits:

```bash
python scripts/train.py --config experiments/cms_experiment_lg/cms_experiment_lg.yaml
```

Config is pre-tuned for MI300X: batch=256, bf16, 25M params (d=512, L=8), 300k steps, TDT embeddings for float32 data.
