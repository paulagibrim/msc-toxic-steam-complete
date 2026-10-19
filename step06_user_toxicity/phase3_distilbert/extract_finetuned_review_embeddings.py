"""Extracts per-review embeddings from a phase3 fine-tuning checkpoint's
DistilBERT ENCODER - not the frozen pretrained one phase4/step07 uses.
Lets pooling_ablation.py be rerun COMPLETELY UNMODIFIED on top of a
fine-tuned encoder instead of the frozen pretrained one, to test whether
the fine-tuning pipeline's mean-only pooling (vs mean+max) is what makes
it so much weaker than the frozen-encoder Linha C results, or whether the
weakness is the fine-tuning process itself.

WHY A SEPARATE SCRIPT, not importing 02_train_mil_distilbert.py: that
file's name starts with a digit, so it isn't importable as a normal Python
module - this project's established convention (see
phase4_text_only_fullpop/01_build_distilbert_embeddings.py) is to
duplicate the small pieces needed rather than fight that. MILToxicityClassifier's
definition and the MODEL_NAME/MAX_SEQ_LENGTH constants below are copied
VERBATIM from 02_train_mil_distilbert.py - any change to that file's
encoder architecture must be mirrored here, or the checkpoint's
state_dict will fail to load.

CHECKPOINT CAVEATS - read before trusting the result:
  1. An in-progress checkpoint (.foldN_inprogress.pt) is a snapshot from
     PARTWAY THROUGH training, not that fold's finished/best state - it
     only exists on disk because 02_train_mil_distilbert.py deletes it
     the instant a fold finishes. This is a proof-of-concept probe of the
     pooling hypothesis, not a substitute for the fold's real result.
  2. torch.save() in that script writes DIRECTLY to the checkpoint path
     (no atomic temp-file rename) - copying the file while the training
     job is mid-write can yield a torn/corrupt read. COPY THE CHECKPOINT
     FIRST (cp), then point --checkpoint at the copy, never at the live
     path a running job still owns. If torch.load() raises here, just
     re-copy (the job's next periodic save will have moved past whatever
     mid-write state broke the copy).
  3. Only the ENCODER weights are reused (`encoder.*` state_dict keys) -
     the checkpoint's `classifier.*` (a linear head trained specifically
     for mean-pooled input) is loaded but never called here;
     pooling_ablation.py fits its own fresh classical heads on top of
     whatever pooling strategy is being tested, exactly as it does for
     the frozen pretrained encoder.

Output format matches step07/02b_compute_embeddings_distilbert.py's
exactly (float32 .npy, row order aligned to --review-dataset), so
pooling_ablation.py needs ZERO code changes to consume it - just point
--embeddings at this script's output instead of the frozen encoder's.

Usage:
    cp ~/steam-data/step06-output/phase3_distilbert/.phase3_en_fullpop_lto_oof.fold3_inprogress.pt \\
       ~/steam-data/step06-output/phase3_distilbert/fold3_checkpoint_copy.pt
    python extract_finetuned_review_embeddings.py \\
        --checkpoint ~/steam-data/step06-output/phase3_distilbert/fold3_checkpoint_copy.pt \\
        --review-dataset ~/steam-data/step07-output/review_dataset_en.parquet \\
        --output ~/steam-data/step07-output/review_embeddings_finetuned_en_fold3.npy \\
        --batch-size 256
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-multilingual-cased"  # MUST match 02_train_mil_distilbert.py
MAX_SEQ_LENGTH = 256                                # MUST match 02_train_mil_distilbert.py
EMBEDDING_DIM = 768


class MILToxicityClassifier(nn.Module):
    """Verbatim copy of 02_train_mil_distilbert.py's class - only
    `.encoder` is used here; `.classifier` just needs to exist so the
    checkpoint's state_dict loads without missing/unexpected keys."""
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path, help="a COPY of a phase3 .foldN_inprogress.pt checkpoint")
    p.add_argument("--review-dataset", required=True, type=Path, help="review_dataset_<lang>.parquet (step07)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    device = get_device()

    print(f"Loading tokenizer/encoder ({MODEL_NAME}) and checkpoint {args.checkpoint} on device={device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MILToxicityClassifier(MODEL_NAME)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[checkpoint] epoch={ckpt.get('epoch')} batches_done={ckpt.get('batches_done')} "
          f"batch_size={ckpt.get('batch_size')} - REMINDER: this is a snapshot mid-training, not the fold's final state")
    model.to(device)
    model.eval()

    df = pd.read_parquet(args.review_dataset, columns=["review_text_clean"])
    texts = df["review_text_clean"].tolist()
    n_total = len(texts)
    print(f"Encoding {n_total:,} review(s)...")

    # Resumable like step07/02b_compute_embeddings_distilbert.py - a checkpoint
    # this expensive (25M reviews x MAX_SEQ_LENGTH=256, no --time extension
    # guaranteed) losing 12h to a wall-time kill with zero resume would be a
    # real waste. Progress is keyed on n_done only, not on WHICH checkpoint
    # produced it - deliberately: re-running the SAME --checkpoint file should
    # resume; pointing --output at a stale run from a DIFFERENT --checkpoint
    # is a usage error the CLI doesn't try to guard against (this is a
    # one-off probe script, not a production pipeline stage).
    checkpoint_meta = args.output.with_suffix(".progress.json")
    n_done = 0
    if args.output.exists() and checkpoint_meta.exists():
        n_done = json.loads(checkpoint_meta.read_text())["n_done"]
        print(f"[resume] {n_done:,} of {n_total:,} already encoded")

    out = np.lib.format.open_memmap(
        args.output, mode="r+" if n_done else "w+", dtype="float32", shape=(n_total, EMBEDDING_DIM)
    )
    bar = tqdm(total=n_total, initial=n_done, unit="review")
    with torch.no_grad():
        for start in range(n_done, n_total, args.batch_size):
            end = min(start + args.batch_size, n_total)
            batch = texts[start:end]
            encoded = tokenizer(batch, padding="max_length", truncation=True,
                                 max_length=MAX_SEQ_LENGTH, return_tensors="pt")
            outputs = model.encoder(
                input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
            )
            cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
            out[start:end] = cls_vectors
            if (start // args.batch_size) % 200 == 0:
                out.flush()
                checkpoint_meta.write_text(json.dumps({"n_done": end}))
            bar.update(end - start)
    out.flush()
    checkpoint_meta.write_text(json.dumps({"n_done": n_total}))
    bar.close()
    print(f"Salvo em {args.output}")
    checkpoint_meta.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
