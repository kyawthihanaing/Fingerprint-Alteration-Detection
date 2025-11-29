# src/prepare_metadata.py
from . import config
import os, re, glob, argparse, pandas as pd
from pathlib import Path

# map raw finger tokens -> canonical names used elsewhere
FINGER_MAP = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Little",
    "pinky": "Little",  # just in case
}

def infer_subset(p: str) -> str:
    lower = p.lower()
    if "altered-easy" in lower:
        return "Altered-Easy"
    if "altered-medium" in lower:
        return "Altered-Medium"
    if "altered-hard" in lower:
        return "Altered-Hard"
    return "Real"

def normalize_to_local(image_path: str) -> str:
    """
    Convert Kaggle-style absolute paths to a path rooted at config.DATA_DIR.
    Anything after the first 'SOCOFing/' segment is preserved.
    """
    p = image_path.replace("\\", "/")
    # find the anchor 'SOCOFing/' and rebuild
    m = re.search(r"/SOCOFing/(.*)$", p, flags=re.IGNORECASE)
    if m:
        return str(Path(config.DATA_DIR) / m.group(1))
    # if no anchor, assume it's already relative to DATA_DIR
    if not os.path.isabs(p):
        return str(Path(config.DATA_DIR) / p)
    return p  # as-is (last resort)

def parse_from_label_and_filename(label: str, filename: str):
    """
    label looks like '115__F' or '115__M'
    filename e.g. '115__F_Left_ring_finger.BMP'
    """
    # subject id: digits before the double underscore
    m = re.match(r"(\d+)__([MF])", label.strip())
    if m:
        subject_id, gender = m.group(1), m.group(2)
    else:
        # fallback: pull leading digits
        subject_id = re.sub(r"\D", "", label) or "000"
        gender = "M" if "_M" in label.upper() else ("F" if "_F" in label.upper() else "U")

    # hand & finger from filename tokens
    fname = os.path.basename(filename)
    # Left/Right
    hand = "Left" if re.search(r"(?i)_left_", fname) else ("Right" if re.search(r"(?i)_right_", fname) else "UNK")
    # finger
    # match '_<finger>_finger' or just the plain token
    m2 = re.search(r"_(thumb|index|middle|ring|little|pinky)(?:_finger)?\.", fname, flags=re.IGNORECASE)
    finger_raw = m2.group(1).lower() if m2 else "UNK"
    finger = FINGER_MAP.get(finger_raw, "UNK")

    return subject_id, gender, hand, finger

def build_from_csv(csv_path: str) -> pd.DataFrame:
    df_in = pd.read_csv(csv_path)
    needed = {"ID", "Label", "ImagePath"}
    missing = needed - set(df_in.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

    rows = []
    for _, r in df_in.iterrows():
        raw_path = str(r["ImagePath"])
        local_path = normalize_to_local(raw_path)
        sid, gender, hand, finger = parse_from_label_and_filename(str(r["Label"]), local_path)
        subset = infer_subset(local_path)
        rows.append(dict(
            path=local_path,
            subject_id=str(sid),
            gender=gender,
            hand=hand,
            finger=finger,
            subset=subset
        ))
    return pd.DataFrame(rows)

def build_by_crawl() -> pd.DataFrame:
    base = config.DATA_DIR
    if not os.path.isdir(base):
        raise SystemExit(f"Expected dataset at {base}. Place SOCOFing there or use --csv.")
    exts = ("*.BMP","*.bmp","*.png","*.jpg","*.jpeg","*.tif")
    paths = []
    for pat in exts:
        paths += glob.glob(os.path.join(base,"**",pat), recursive=True)
    rows = []
    for p in sorted(paths):
        # derive label from filename for backwards compat
        fname = os.path.basename(p)
        # attempt to extract '123__M' style
        m = re.match(r"(\d+)__([MF])", fname)
        label = f"{m.group(1)}__{m.group(2)}" if m else fname
        sid, gender, hand, finger = parse_from_label_and_filename(label, p)
        subset = infer_subset(p)
        rows.append(dict(path=p, subject_id=str(sid), gender=gender, hand=hand, finger=finger, subset=subset))
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to SOCOFing_Full_Organised.csv (optional).")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)

    if args.csv:
        df = build_from_csv(args.csv)
    else:
        df = build_by_crawl()

    # keep only rows with known gender (M/F)
    df = df[df["gender"].isin(["M","F"])].reset_index(drop=True)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=["path"], keep="first").reset_index(drop=True)
    
    # (NEW) Binary label: Real (0) vs Altered (1) - leak-free, widely-studied task
    df["is_altered"] = (df["subset"] != "Real").astype(int)

    # warn for any paths that don't exist locally
    not_found = (~df["path"].apply(os.path.exists)).sum()
    if not_found:
        print(f"[WARN] {not_found} files listed in metadata do not exist at the normalized path. "
              f"Check config.DATA_DIR or use --csv to normalize properly.")

    out = "reports/metadata.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows")
    print(f"   Real: {(df['is_altered']==0).sum()}, Altered: {(df['is_altered']==1).sum()}")

if __name__ == "__main__":
    main()
