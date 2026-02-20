# Usage: python experiments/atlas_experiment/prepare.py
import os, sys
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

COMBINED = os.path.join(DATA_DIR, "atlas_combined.bin")

SPLITS = {
    "atlas_train.bin": (0.00, 0.80),
    "atlas_val.bin":   (0.80, 0.90),
    "atlas_test.bin":  (0.90, 1.00),
}

def slice_bytes(src: str, dst: str, start_byte: int, end_byte: int):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    size = end_byte - start_byte
    bufsize, rem = 1 << 20, size
    with open(src, "rb") as f, open(dst, "wb") as out:
        f.seek(start_byte)
        while rem > 0:
            buf = f.read(min(bufsize, rem))
            if not buf: break
            out.write(buf); rem -= len(buf)
    print(f"  {dst}  ({os.path.getsize(dst)/1e6:.0f} MB)")

def main():
    if not os.path.exists(COMBINED):
        sys.exit(
            f"ERROR: {COMBINED} not found.\n"
            f"Run: python experiments/atlas_experiment/download.py --all-steps"
        )

    total = os.path.getsize(COMBINED)
    print(f"Source: {COMBINED}  ({total/1e6:.0f} MB)\n")

    for name, frac in SPLITS.items():
        dst = os.path.join(DATA_DIR, name)
        if os.path.exists(dst):
            print(f"  Already present: {dst}")
            continue
        start_byte = int(total * frac[0])
        end_byte   = int(total * frac[1])
        slice_bytes(COMBINED, dst, start_byte, end_byte)

    print("\nDone. Update config.yaml paths if needed:")
    print("  train_data: experiments/atlas_experiment/data/atlas_train.bin")
    print("  val_data:   experiments/atlas_experiment/data/atlas_val.bin")

if __name__ == "__main__":
    main()