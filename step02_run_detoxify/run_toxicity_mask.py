import argparse
from pathlib import Path
import pandas as pd

import toxicity_mask as tm
from pipeline_utils import list_parquet_files, save_summary, info

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates toxicity subsets based on thresholds and outputs a JSON report."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to step02's own output directory (contains review_lang=<lang> subfolders)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Path to write the toxicity report JSON to",
    )
    parser.add_argument(
        "--lang", action="append", dest="languages", default=None,
        help="Language to check (repeat for multiple). Defaults to pt and en.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    languages = args.languages or ["pt", "en"]

    summaries = []
    all_frames = []
    
    for lang in languages:
        lang_dir = args.input / f"review_lang={lang}"
        files = list_parquet_files(lang_dir) if lang_dir.exists() else []
        frames = [pd.read_parquet(f, columns=["perspective_score", "detoxify_score"]) for f in files]
        if frames:
            df = pd.concat(frames, ignore_index=True)
            all_frames.append(df)
            summaries.append(tm.summarize_toxicity(df, lang))
        else:
            info(f"[{lang}] No files found in {lang_dir}")

    if all_frames:
        total_df = pd.concat(all_frames, ignore_index=True)
        summaries.append(tm.summarize_toxicity(total_df, "total"))
        
    save_summary({"toxicity_sets": summaries}, args.output)

if __name__ == "__main__":
    main()
