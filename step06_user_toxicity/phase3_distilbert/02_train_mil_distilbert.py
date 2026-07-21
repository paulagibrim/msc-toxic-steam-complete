"""Phase 3, Step 2: fine-tunes DistilBERT end-to-end as a per-user toxicity
classifier using Multiple Instance Learning (MIL) with mean pooling -
Track B of step06's Phase 3 (see this project's planning conversation for
why MIL over concatenation: users have 1 to ~12,780 reviews, so
concatenating and truncating to a fixed token budget would silently drop
almost all of a prolific user's content; MIL processes every review
individually and only pools the resulting vectors, so no review is
dropped by a token-length limit).

ARCHITECTURE: each user is a "bag" of review "instances".
  1. Every review in a training batch is tokenized and passed through
     DistilBERT independently, taking the [CLS] token's final hidden state
     as that review's vector (768-dim).
  2. A user's review vectors are mean-pooled into ONE vector per user
     (implemented via index_add_, not torch_scatter, to avoid an extra
     dependency - see MILToxicityClassifier.forward).
  3. A linear classification head (768 -> 1) predicts toxicity from the
     pooled vector.
  4. Loss (binary cross-entropy) is computed ONCE PER USER, not per
     review - gradients flow back through the mean-pool into every review
     that contributed, and from there into DistilBERT's shared weights.
     This is genuine end-to-end fine-tuning, unlike Phase 2's classical ML
     (which used this same sentence-transformer-family model, but frozen,
     only as a feature extractor).

WHY k-FOLD, NOT LEAVE-ONE-OUT (unlike Phase 2): fine-tuning a transformer
takes far longer per fit than Logistic Regression - repeating it once per
positive user (75-1,316 times, as Phase 2 did) is computationally
infeasible. Uses StratifiedKFold (k=5) instead - a real trade-off of
statistical precision for feasibility, explicitly acknowledged: in pt (75
positives), this leaves only ~15 positives per test fold, a real
limitation worth flagging when reporting results, not hiding.

WHY A REVIEW CAP DURING TRAINING (--max-reviews-per-user-train, default
10): a single training step must bound its compute, and one very prolific
user (up to ~12,780 reviews) would otherwise dominate a batch's cost.
10 covers the vast majority of TOXIC users without truncation (mean 4.16,
median 3 reviews - measured on this corpus's pt population; negatives
skew even lower, mean 1.52/median 1), while keeping the fixed per-batch
cost (see make_collate_fn) modest. Applied ONLY during training -
evaluation always uses every review a test user has, uncapped, so
reported metrics aren't affected by this efficiency measure.

EVERY TRAINING BATCH IS PADDED TO A FIXED SHAPE, not just a capped one:
make_collate_fn pads each user's reviews up to EXACTLY
max_reviews_per_user_train slots (masked out of the mean-pool, not
silently averaged in) so every batch is IDENTICALLY
(batch_size * max_reviews_per_user_train, seq_len). Fixing sequence
length alone (see CLEANING/padding below) was not sufficient - the
number of reviews per batch still varied batch-to-batch (the balanced
sampler draws ~50% toxic users, who have ~3x more reviews on average
than non-toxic ones), and that alone kept fragmenting MPS's (Apple GPU)
caching allocator until a fresh OOM at ~20GB allocated, 5,479 batches
into a resumed run - well past where the sequence-length fix alone had
gotten. The trade-off is real: every batch now costs as much as the
worst case would have, not just occasionally - a deliberate stability-
over-speed choice after two separate OOM crashes from shape variability.

CLASS IMBALANCE: a WeightedRandomSampler on the training DataLoader, NOT
BCEWithLogitsLoss's pos_weight (see train_one_fold's comments for why a
~2,387x loss multiplier - pt's actual negative:positive ratio - destabilizes
gradient-based training instead of helping, unlike Phase 2's
class_weight='balanced', which works fine there because Logistic
Regression is convex). The sampler instead draws positives and negatives
with equal expected frequency per batch, resampling the rare positives
with replacement across an epoch.

--leave-toxic-out: mirrors Phase 2's circularity control for this MIL
architecture - for POSITIVE users, individually-toxic reviews (is_toxic_
review column from 01_prepare_review_texts.py) are excluded from their bag
before pooling, in both training and evaluation. Positive users left with
zero reviews after this exclusion are dropped from the run entirely (no
valid bag to construct), same as Phase 2.

Usage:
    python 02_train_mil_distilbert.py \\
        --review-texts ../../../../steam-data/step06-output/phase3_distilbert/review_texts.parquet \\
        --feature-table ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/phase1_feature_table.parquet \\
        --population pt \\
        --output ../../../../steam-data/step06-output/phase3_distilbert/phase3_pt_oof_predictions.parquet \\
        --n-folds 5 --epochs 3 --batch-size 8
"""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pipeline_utils import info, save_summary

def autocast_ctx(device: str):
    """bf16 mixed precision on CUDA (A100 Tensor Cores get ~2-4x throughput
    in bf16 vs fp32, no GradScaler needed unlike fp16) - a no-op on MPS/CPU,
    so the Mac/consumer-GPU runs already tuned are unaffected."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_SEQ_LENGTH = 256
RANDOM_STATE = 42
CHECKPOINT_EVERY_FOLD = True  # this script checkpoints at fold granularity, not batch granularity - see main()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tunes DistilBERT as a per-user MIL (mean-pooling) toxicity classifier."
    )
    parser.add_argument("--review-texts", required=True, type=Path, help="Path to 01_prepare_review_texts.py's output")
    parser.add_argument("--feature-table", required=True, type=Path, help="Path to phase1_feature_table.parquet (labels)")
    parser.add_argument("--population", required=True, choices=["pt", "en", "union"], help="Which population to run")
    parser.add_argument("--output", required=True, type=Path, help="Path to write out-of-fold predictions parquet to")
    parser.add_argument("--n-folds", type=int, default=5, help="Stratified k-fold count (default 5)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per fold (default 3)")
    parser.add_argument("--batch-size", type=int, default=8, help="Users per training batch (default 8)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate (default 2e-5)")
    parser.add_argument(
        "--max-reviews-per-user-train", type=int, default=10,
        help="Cap on reviews sampled per user DURING TRAINING ONLY (default 10, covers the vast majority of "
        "toxic users - mean 4.16/median 3 reviews - without truncation) - evaluation always uses all reviews. "
        "Every training batch is now padded to EXACTLY batch_size*this value (see make_collate_fn), so raising "
        "this significantly increases the fixed per-batch compute cost, not just the worst case.",
    )
    parser.add_argument(
        "--leave-toxic-out", action="store_true",
        help="Circularity control: excludes positive users' individually-toxic reviews from their bag",
    )
    parser.add_argument(
        "--limit-users", type=int, default=None,
        help="VALIDATION ONLY: cap the number of users used (stratified), instead of the full population",
    )
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU (M1/M2/M3/M4) - PyTorch's Metal backend, not CUDA
    return "cpu"


def load_user_bags(review_texts_path: Path, feature_table_path: Path, population: str, leave_toxic_out: bool) -> tuple:
    """Returns (bags, y, user_urls): bags[i] is the list of review texts for
    user_urls[i], y[i] is their toxic label (0/1). Reviews are filtered to
    the population's language (pt/en) or kept all (union)."""
    label_suffix = "pt+en" if population == "union" else population
    table = pd.read_parquet(feature_table_path, columns=["user_url", f"eligible_{label_suffix}", f"is_toxic_{label_suffix}"])
    eligible = table[f"eligible_{label_suffix}"].fillna(False)
    labels = table.loc[eligible, ["user_url", f"is_toxic_{label_suffix}"]].rename(
        columns={f"is_toxic_{label_suffix}": "is_toxic"}
    )
    labels["is_toxic"] = labels["is_toxic"].fillna(False)
    info(f"[{population}] {len(labels):,} eligible user(s), {int(labels['is_toxic'].sum()):,} positive (toxic)")

    reviews = pd.read_parquet(review_texts_path)
    reviews = reviews[reviews["user_url"].isin(set(labels["user_url"]))]
    if population != "union":
        reviews = reviews[reviews["review_lang"] == population]

    labels_by_user = labels.set_index("user_url")["is_toxic"]

    if leave_toxic_out:
        # Only exclude toxic reviews from POSITIVE users - negatives are untouched.
        positive_users = set(labels.loc[labels["is_toxic"], "user_url"])
        is_positive_review_owner = reviews["user_url"].isin(positive_users)
        n_before = len(reviews)
        reviews = reviews[~(is_positive_review_owner & reviews["is_toxic_review"])]
        info(f"[leave-toxic-out] Excluded {n_before - len(reviews):,} individually-toxic review(s) from positive users' bags")

    grouped = reviews.groupby("user_url")["review_text_clean"].apply(list)

    user_urls = []
    bags = []
    y = []
    n_dropped_empty_bag = 0
    for user_url, is_toxic in labels_by_user.items():
        texts = grouped.get(user_url)
        if texts is None or len(texts) == 0:
            n_dropped_empty_bag += 1
            continue
        user_urls.append(user_url)
        bags.append(texts)
        y.append(int(is_toxic))

    if n_dropped_empty_bag:
        info(
            f"[{population}] {n_dropped_empty_bag:,} user(s) dropped - no review text survives in this "
            f"population/mode (0-review bag would be undefined)"
        )

    return bags, np.array(y), np.array(user_urls)


class BagDataset(Dataset):
    def __init__(self, bags: list, y: np.ndarray, max_reviews: int = None, seed: int = 0):
        self.bags = bags
        self.y = y
        self.max_reviews = max_reviews  # None at eval time - use every review
        # A local, seeded generator (not the unseeded global np.random used
        # by an earlier version of this class) - without this, which
        # reviews get sampled for an over-cap user was different every run
        # (and after every checkpoint resume), making results
        # non-reproducible. Advances statefully across __getitem__ calls,
        # same as any other RNG use in a data pipeline - fine with
        # num_workers=0 (this project's default), where items are always
        # fetched sequentially in one process.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        texts = self.bags[idx]
        if self.max_reviews is not None and len(texts) > self.max_reviews:
            chosen = self.rng.choice(len(texts), size=self.max_reviews, replace=False)
            texts = [texts[i] for i in chosen]
        return texts, float(self.y[idx])


def _bucket_size(actual_max: int, cap: int) -> int:
    """Rounds actual_max UP to the next power-of-2 bucket, capped at `cap`
    (e.g. cap=10 -> buckets are 1, 2, 4, 8, 10) - see make_collate_fn's
    docstring for why: bounds the number of DISTINCT batch shapes MPS/CUDA's
    caching allocator ever sees across a whole run to a handful, instead of
    either always padding to the global cap (correctness-safe but ~3.5x
    slower on average, since most batches don't need it) or letting shape
    vary continuously batch-to-batch (fragments the allocator - the OOM
    this project hit twice)."""
    bucket = 1
    while bucket < actual_max:
        bucket *= 2
    return min(bucket, cap)


def make_collate_fn(tokenizer, fixed_reviews_per_user: int = None):
    """fixed_reviews_per_user: if given (training only), every user in the
    batch is padded with empty placeholder reviews up to a BUCKET size
    (see _bucket_size) chosen from this batch's own actual max review
    count, capped at fixed_reviews_per_user - not always padded to the
    full cap. A review_valid mask marks which slots are real vs. padding.
    This keeps the total number of distinct shapes MPS/CUDA ever sees
    small (bounded by log2(cap)+1 buckets) - enough for the allocator to
    reuse blocks and stop fragmenting - while a batch of mostly 1-2-review
    users (the common case) still only pays for a small bucket, not the
    full cap every time. Not just sequence length (already fixed via
    padding="max_length"): the NUMBER of reviews per batch was still
    varying batch-to-batch (a user's real review count, capped but not
    padded up at all), and MPS's caching allocator fragmented on THAT
    dimension instead - a fresh OOM was hit ~5,479 batches into a resumed
    run, at 20GB allocated, well past where the sequence-length fix alone
    had gotten. An earlier version of this function padded every batch to
    the FULL cap unconditionally - it fixed the crash but made every batch
    cost as much as the worst case, an observed ~5x slowdown in practice.
    Padding rows are masked out of the mean-pool in
    MILToxicityClassifier.forward, not silently averaged in - mean pooling
    with padding included would change the result, not just its shape.

    None at eval time (evaluate_bags doesn't use this collate_fn at all -
    it processes one user's uncapped bag in its own chunks instead)."""
    def collate_fn(batch):
        """Flattens every review across the whole batch into one token
        batch for the encoder, plus a user_index array mapping each
        flattened review back to its position (0..batch_size-1) in this
        batch - used by the model to mean-pool per user afterward."""
        all_texts = []
        user_index = []
        review_valid = []
        labels = []

        n_slots = None
        if fixed_reviews_per_user is not None:
            actual_max = max(len(texts) for texts, _ in batch)
            n_slots = _bucket_size(actual_max, fixed_reviews_per_user)

        for i, (texts, label) in enumerate(batch):
            if n_slots is not None:
                n_real = len(texts)
                n_pad = n_slots - n_real
                all_texts.extend(texts + [""] * n_pad)
                user_index.extend([i] * n_slots)
                review_valid.extend([1.0] * n_real + [0.0] * n_pad)
            else:
                all_texts.extend(texts)
                user_index.extend([i] * len(texts))
                review_valid.extend([1.0] * len(texts))
            labels.append(label)

        # Fixed-length padding (not dynamic padding=True): every review
        # produces an identically-shaped token sequence regardless of how
        # long it is. Dynamic padding wastes less compute on short
        # sequences, but was found to fragment MPS's (Apple GPU) caching
        # allocator over a long training run - each differently-shaped
        # batch needs a differently-sized memory block, and blocks freed
        # from one shape can't be reused for another, so usage crept up
        # until it OOM'd (crashed at batch 72/17,905 of fold 1 on a 32GB
        # M4 Pro). Fixed shape lets the allocator reuse the exact same
        # blocks batch after batch.
        encoded = tokenizer(
            all_texts, padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "user_index": torch.tensor(user_index, dtype=torch.long),
            "review_valid": torch.tensor(review_valid, dtype=torch.float32),
            "n_users": len(batch),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    return collate_fn


class MILToxicityClassifier(nn.Module):
    """DistilBERT encoder + mean-pool-by-user (Multiple Instance Learning)
    + a linear classification head. See module docstring for the full
    architecture rationale."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, user_index, n_users, review_valid=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        review_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] token per review

        hidden_size = review_vectors.shape[1]
        device = review_vectors.device

        if review_valid is not None:
            # Padding rows (from make_collate_fn's fixed_reviews_per_user)
            # still get a forward pass - DistilBERT has no notion of "this
            # whole review doesn't exist" - but must NOT contribute to the
            # pooled average. Zeroing them here and excluding them from the
            # per-user count makes the result IDENTICAL to pooling only the
            # real reviews (mean pooling is just a sum/count; padding rows
            # contribute 0/0 to both), not an approximation.
            review_vectors = review_vectors * review_valid.unsqueeze(1)
            valid_per_review = review_valid
        else:
            valid_per_review = torch.ones(review_vectors.shape[0], device=device)

        sums = torch.zeros(n_users, hidden_size, device=device)
        sums.index_add_(0, user_index, review_vectors)
        counts = torch.zeros(n_users, device=device)
        counts.index_add_(0, user_index, valid_per_review)
        pooled = sums / counts.unsqueeze(1).clamp(min=1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits


EVAL_CHUNK_SIZE = 32  # reviews per forward pass during evaluation - see evaluate_bags


def evaluate_bags(model, bags: list, tokenizer, device) -> np.ndarray:
    """Evaluates every user's FULL bag, uncapped - by design (see module
    docstring: reported metrics must reflect every review a test user has,
    never a training-time efficiency cap).

    NOT implemented via the same batched-across-users DataLoader path as
    training: batching several test users together, uncapped, means the
    single largest bag in that batch sets the tensor size for everyone in
    it - and this corpus has users with up to ~12,780 reviews. One such
    user sharing a batch would force a forward pass over ~12,780 x 256
    tokens at once, an OOM risk this project already hit once for a
    similar reason (see the fixed-padding fix earlier in this file).

    Instead, evaluates users ONE AT A TIME, and for a user whose bag
    exceeds EVAL_CHUNK_SIZE, processes their reviews in bounded chunks -
    accumulating a running sum (and count) of per-review [CLS] vectors
    across chunks, then dividing once at the end. This is mathematically
    identical to pooling every review in a single pass (mean pooling is
    associative/commutative - grouping the sum differently doesn't change
    the result), just bounded in peak memory regardless of how many
    reviews any single user has. Slower than multi-user batching (most
    users have only 1-2 reviews, so batch_size effectively collapses to
    ~1-2 most of the time) - an accepted trade-off since evaluation runs
    once per fold, not thousands of times like a training step."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for texts in tqdm(bags, desc="  evaluating", unit="user", smoothing=0):
            vector_sum = None
            for chunk_start in range(0, len(texts), EVAL_CHUNK_SIZE):
                chunk = texts[chunk_start : chunk_start + EVAL_CHUNK_SIZE]
                encoded = tokenizer(
                    chunk, padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
                )
                with autocast_ctx(device):
                    outputs = model.encoder(
                        input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
                    )
                chunk_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] per review in this chunk
                chunk_sum = chunk_vectors.sum(dim=0)
                vector_sum = chunk_sum if vector_sum is None else vector_sum + chunk_sum

            pooled = (vector_sum / len(texts)).unsqueeze(0)
            logits = model.classifier(pooled).squeeze(-1)
            predictions.append(torch.sigmoid(logits).item())

    return np.array(predictions)


def save_inprogress_checkpoint(
    path: Path, model, optimizer, epoch: int, batches_done: int, batch_size: int, max_reviews_per_user_train: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "batches_done": batches_done,
            # Recorded so a resumed run can detect a hyperparameter change
            # (see train_one_fold's loading logic) - "batch 600" only means
            # the same thing on resume if the batch composition (batch_size,
            # review cap) that produced it is unchanged. A real run hit
            # this: batch_size was bumped from 64->96 mid-fold, and this
            # checkpoint (with no such check at the time) silently resumed
            # as if 600 batches of size 96 had already run, when they were
            # actually 600 batches of size 64 - a different, incompatible
            # amount of training progress, not caught until noticed by eye.
            "batch_size": batch_size,
            "max_reviews_per_user_train": max_reviews_per_user_train,
        },
        path,
    )


def train_one_fold(
    train_bags, train_y, test_bags, test_y, tokenizer, device,
    epochs, batch_size, learning_rate, max_reviews_per_user_train,
    fold_checkpoint_path: Path = None, fold_seed: int = 0,
) -> np.ndarray:
    """Trains a fresh MILToxicityClassifier on this fold's training bags,
    returns predicted probabilities for the test bags (evaluated with
    every review a test user has, uncapped).

    If fold_checkpoint_path is given, periodically saves model/optimizer
    state DURING training (not just once per completed fold, as
    main()'s outer checkpoint already does) - a single fold over the full
    population is tens of thousands of batches (was measured at 17,905 for
    pt's fold 1 alone), so without this, any interruption mid-fold loses
    everything trained in that fold so far. fold_seed fixes the
    DataLoader's shuffle order so a resumed run can deterministically skip
    the batches already completed in the epoch it was interrupted during."""
    model = MILToxicityClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    batches_to_skip = 0
    if fold_checkpoint_path and fold_checkpoint_path.exists():
        ckpt = torch.load(fold_checkpoint_path, map_location=device)
        ckpt_batch_size = ckpt.get("batch_size")
        ckpt_max_reviews = ckpt.get("max_reviews_per_user_train")
        if ckpt_batch_size != batch_size or ckpt_max_reviews != max_reviews_per_user_train:
            info(
                f"  [checkpoint] IGNORING stale in-progress checkpoint - saved with "
                f"batch_size={ckpt_batch_size}, max_reviews_per_user_train={ckpt_max_reviews}, "
                f"current run uses batch_size={batch_size}, max_reviews_per_user_train={max_reviews_per_user_train}. "
                f"'batch N' means a different amount of progress under different settings - restarting this fold from scratch."
            )
        else:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"]
            batches_to_skip = ckpt["batches_done"]
            info(f"  [checkpoint] Resuming fold from epoch {start_epoch + 1}, batch {batches_to_skip}")

    # NOT a pos_weight-scaled loss (an earlier version of this script used
    # BCEWithLogitsLoss(pos_weight=n_neg/n_pos), which computed to ~2,387
    # for pt - fine for a convex model like Phase 2's Logistic Regression,
    # but destructive for gradient-based fine-tuning: with batch_size=8
    # users and ~0.04% positive prevalence, almost every batch contains
    # ZERO positive users, so the network gets essentially no gradient
    # signal for most of training - and on the rare batch that does
    # contain one, a ~2,387x loss multiplier produces a gradient spike
    # large enough to destabilize training rather than teach anything.
    # Measured result: predicted scores collapsed to ~0.30 for positives
    # and ~0.30 for negatives alike (AUC-PR 0.001, barely above the
    # 0.0004 baseline) - the model had not learned to discriminate at all.
    #
    # Fixed by balancing at the SAMPLING level instead: a
    # WeightedRandomSampler draws positive and negative users with equal
    # expected frequency (positives resampled with replacement across an
    # epoch, since there are far fewer of them) - every batch now reliably
    # contains a healthy mix of both classes, and the loss itself can stay
    # unweighted (pos_weight=1), avoiding the gradient-spike problem
    # entirely.
    loss_fn = nn.BCEWithLogitsLoss()

    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    sample_weights = np.where(train_y == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    generator = torch.Generator().manual_seed(fold_seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_y), replacement=True, generator=generator,
    )

    # fixed_reviews_per_user=max_reviews_per_user_train: every training
    # batch gets padded/masked to EXACTLY batch_size * max_reviews_per_user_train
    # review-slots (see make_collate_fn's docstring) - the number of
    # reviews per batch was still variable even after fixing sequence
    # length, and that variability alone was enough to keep fragmenting
    # MPS's allocator until it OOM'd again ~5,479 batches into a resumed
    # run.
    collate_fn = make_collate_fn(tokenizer, fixed_reviews_per_user=max_reviews_per_user_train)
    train_dataset = BagDataset(train_bags, train_y, max_reviews=max_reviews_per_user_train, seed=fold_seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    CACHE_CLEAR_EVERY = 20  # batches
    CHECKPOINT_EVERY_BATCHES = 200

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        skip_remaining = batches_to_skip if epoch == start_epoch else 0
        # smoothing=0: plain cumulative average (total batches / total time
        # so far), not tqdm's default exponential weighting toward recent
        # batches. With bucketed padding, per-batch time varies a lot
        # (bucket 1 vs bucket 10 batches differ ~10x in compute) - the
        # default smoothing made the ETA swing wildly (e.g. 6h to 25h)
        # depending on which bucket sizes were hit most recently, instead
        # of reflecting the true running average.
        bar = tqdm(train_loader, desc=f"  epoch {epoch + 1}/{epochs}", unit="batch", smoothing=0)
        for step, batch in enumerate(bar):
            if step < skip_remaining:
                continue  # deterministic replay of the same shuffle order - cheap (no forward/backward)

            optimizer.zero_grad()
            with autocast_ctx(device):
                logits = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["user_index"].to(device), batch["n_users"],
                    review_valid=batch["review_valid"].to(device),
                )
                loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            bar.set_postfix(loss=epoch_loss / (bar.n + 1 - skip_remaining))

            # Periodic cache release (not just once at the very end of the
            # fold, as before) - PyTorch's MPS/CUDA caching allocators hold
            # freed blocks for reuse rather than returning them to the OS,
            # and were observed to still creep upward over many steps even
            # with fixed-shape batches - this bounds how much accumulates
            # before being released.
            if device in ("cuda", "mps") and (step + 1) % CACHE_CLEAR_EVERY == 0:
                (torch.cuda if device == "cuda" else torch.mps).empty_cache()

            if fold_checkpoint_path and (step + 1) % CHECKPOINT_EVERY_BATCHES == 0:
                save_inprogress_checkpoint(
                    fold_checkpoint_path, model, optimizer, epoch, step + 1, batch_size, max_reviews_per_user_train,
                )

        info(f"  epoch {epoch + 1}/{epochs} - mean loss: {epoch_loss / max(len(train_loader) - skip_remaining, 1):.4f}")
        if fold_checkpoint_path:
            save_inprogress_checkpoint(
                fold_checkpoint_path, model, optimizer, epoch + 1, 0, batch_size, max_reviews_per_user_train,
            )

    predictions = evaluate_bags(model, test_bags, tokenizer, device)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return np.array(predictions)


def main():
    args = parse_args()

    device = get_device()
    info(f"Using device={device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bags, y, user_urls = load_user_bags(args.review_texts, args.feature_table, args.population, args.leave_toxic_out)
    info(f"[{args.population}] {len(bags):,} user(s) with a non-empty bag, {int(y.sum()):,} positive")

    if args.limit_users:
        rng = np.random.default_rng(RANDOM_STATE)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos_sample = min(len(pos_idx), max(1, args.limit_users // 2))
        n_neg_sample = min(len(neg_idx), args.limit_users - n_pos_sample)
        sampled = np.concatenate([
            rng.choice(pos_idx, size=n_pos_sample, replace=False),
            rng.choice(neg_idx, size=n_neg_sample, replace=False),
        ])
        bags = [bags[i] for i in sampled]
        y = y[sampled]
        user_urls = user_urls[sampled]
        info(f"[VALIDATION MODE] Limited to {len(bags)} user(s) ({int(y.sum())} positive)")

    checkpoint_dir = args.output.parent / f".{args.output.stem}_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_results_path = checkpoint_dir / "fold_predictions.parquet"
    config_path = checkpoint_dir / "run_config.json"

    # A checkpoint is only safe to resume if it was produced by an
    # IDENTICAL invocation - a fold's train/test split membership depends
    # on --n-folds, its training depends on --epochs/--batch-size/
    # --learning-rate/--max-reviews-per-user-train, and its population
    # depends on --leave-toxic-out. Reusing --output with any of these
    # changed would otherwise silently mix fold predictions computed under
    # different configurations into one "result" with no error at all.
    current_config = {
        "population": args.population,
        "leave_toxic_out": args.leave_toxic_out,
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_reviews_per_user_train": args.max_reviews_per_user_train,
        "limit_users": args.limit_users,
    }

    completed_folds = {}
    if fold_results_path.exists():
        if config_path.exists():
            saved_config = json.loads(config_path.read_text())
            if saved_config != current_config:
                mismatched = {
                    k: (saved_config.get(k), current_config[k])
                    for k in current_config if saved_config.get(k) != current_config[k]
                }
                raise SystemExit(
                    f"Checkpoint at {checkpoint_dir} was produced with different arguments than this run "
                    f"(saved -> current): {mismatched}. Resuming would silently mix incompatible fold "
                    f"predictions. Either match the original arguments, or remove the checkpoint directory "
                    f"to start fresh."
                )
        existing = pd.read_parquet(fold_results_path)
        completed_folds = {int(f): sub for f, sub in existing.groupby("fold")}
        info(f"[checkpoint] Resuming: {len(completed_folds)} fold(s) already completed")

    config_path.write_text(json.dumps(current_config, indent=2))

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=RANDOM_STATE)
    all_fold_predictions = []

    for fold_idx, (train_pos, test_pos) in enumerate(skf.split(np.zeros(len(y)), y)):
        if fold_idx in completed_folds:
            all_fold_predictions.append(completed_folds[fold_idx])
            continue

        info(f"[{args.population}] Fold {fold_idx + 1}/{args.n_folds}: {len(train_pos)} train, {len(test_pos)} test user(s)")
        train_bags = [bags[i] for i in train_pos]
        train_y = y[train_pos]
        test_bags = [bags[i] for i in test_pos]
        test_y = y[test_pos]

        fold_checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_inprogress.pt"
        predictions = train_one_fold(
            train_bags, train_y, test_bags, test_y, tokenizer, device,
            args.epochs, args.batch_size, args.learning_rate, args.max_reviews_per_user_train,
            fold_checkpoint_path=fold_checkpoint_path, fold_seed=RANDOM_STATE + fold_idx,
        )
        fold_checkpoint_path.unlink(missing_ok=True)  # fold finished - the in-progress checkpoint is no longer needed

        fold_df = pd.DataFrame({
            "fold": fold_idx,
            "user_url": user_urls[test_pos],
            "y_true": test_y,
            "y_score": predictions,
        })
        all_fold_predictions.append(fold_df)

        combined_so_far = pd.concat(all_fold_predictions, ignore_index=True)
        combined_so_far.to_parquet(fold_results_path, index=False)
        info(f"[checkpoint] Saved after fold {fold_idx + 1}/{args.n_folds}")

    oof = pd.concat(all_fold_predictions, ignore_index=True)
    auc_pr = average_precision_score(oof["y_true"], oof["y_score"])
    info(f"[{args.population}] AUC-PR (out-of-fold, {args.n_folds}-fold): {auc_pr:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.output, index=False)
    info(f"Saved out-of-fold predictions ({len(oof)} rows) to: {args.output}")

    if not args.limit_users and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        info(f"Cleaned up checkpoint directory: {checkpoint_dir}")

    save_summary(
        {
            "population": args.population,
            "n_users": int(len(bags)),
            "n_positive": int(y.sum()),
            "n_folds": args.n_folds,
            "epochs": args.epochs,
            "auc_pr": round(float(auc_pr), 6),
            "leave_toxic_out_control": args.leave_toxic_out,
            "validation_mode": args.limit_users is not None,
        },
        args.output.with_suffix(".summary.json"),
    )


if __name__ == "__main__":
    main()
