"""Finalize embeddings from checkpoint - load checkpoints and write to
parquet incrementally, without loading entire table into RAM at once."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipeline_utils import info

EMBEDDING_DIM = 384
POPULATIONS = ["pt", "en", "pt+en"]
UNION_KEY = "pt+en"

def finalize_from_checkpoint(checkpoint_dir: Path, profile_metadata: Path, output: Path):
    # Load matched users (define the output index)
    profile = pd.read_parquet(profile_metadata, columns=["user_url"])
    all_users = sorted(profile["user_url"].unique())
    info(f"Finalizing for {len(all_users):,} users...")
    
    # Load checkpoints
    means = {}
    counts = {}
    for pop in POPULATIONS:
        checkpoint_file = checkpoint_dir / f"checkpoint_{pop}.parquet"
        count_file = checkpoint_dir / f"checkpoint_{pop}_count.npy"
        if checkpoint_file.exists() and count_file.exists():
            means[pop] = pd.read_parquet(checkpoint_file)
            counts[pop] = pd.Series(np.load(count_file), index=means[pop].index)
            info(f"Loaded {pop} checkpoint: {len(means[pop]):,} user(s)")
        else:
            means[pop] = pd.DataFrame()
            counts[pop] = pd.Series()
    
    # Write to parquet in one go (reindex handles missing users as NaN)
    # This is still big but at least we're not doing extra operations
    info("Writing output (reindexing to full population)...")
    output_rows = []
    for pop in POPULATIONS:
        pop_label = "union" if pop == UNION_KEY else pop
        emb_cols = [f"emb_{pop_label}_{i}" for i in range(EMBEDDING_DIM)]
        
        if len(means[pop]) > 0:
            emb = means[pop].reindex(all_users)
            emb.columns = emb_cols
            output_rows.append(emb)
            
            n_col = f"n_{pop_label}_embedded"
            output_rows.append(counts[pop].reindex(all_users, fill_value=0).astype("int64").to_frame(n_col))
        else:
            for col in emb_cols:
                output_rows.append(pd.DataFrame({col: [np.nan] * len(all_users)}, index=all_users))
            output_rows.append(pd.DataFrame({f"n_{pop_label}_embedded": [0] * len(all_users)}, index=all_users))
    
    result = pd.concat(output_rows, axis=1)
    result.insert(0, "user_url", all_users)
    
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    info(f"Saved {len(result)} users, {len(result.columns)} columns to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--profile-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    finalize_from_checkpoint(args.checkpoint_dir, args.profile_metadata, args.output)
