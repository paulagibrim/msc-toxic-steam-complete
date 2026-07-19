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
20): a single training step must bound its compute, and one very prolific
user (up to ~12,780 reviews) would otherwise dominate a batch's cost.
Since the median user has only 1-2 reviews, this cap affects only the long
tail. Applied ONLY during training - evaluation always uses every review a
test user has, uncapped, so reported metrics aren't affected by this
efficiency measure.

CLASS IMBALANCE: BCEWithLogitsLoss's pos_weight parameter (ratio of
negatives to positives in the training fold) - the neural-network
equivalent of Phase 2's class_weight='balanced' in scikit-learn.

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
        "--max-reviews-per-user-train", type=int, default=20,
        help="Cap on reviews sampled per user DURING TRAINING ONLY (default 20) - evaluation always uses all reviews",
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
    def __init__(self, bags: list, y: np.ndarray, max_reviews: int = None):
        self.bags = bags
        self.y = y
        self.max_reviews = max_reviews  # None at eval time - use every review

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        texts = self.bags[idx]
        if self.max_reviews is not None and len(texts) > self.max_reviews:
            chosen = np.random.choice(len(texts), size=self.max_reviews, replace=False)
            texts = [texts[i] for i in chosen]
        return texts, float(self.y[idx])


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        """Flattens every review across the whole batch into one token
        batch for the encoder, plus a user_index array mapping each
        flattened review back to its position (0..batch_size-1) in this
        batch - used by the model to mean-pool per user afterward."""
        all_texts = []
        user_index = []
        labels = []
        for i, (texts, label) in enumerate(batch):
            all_texts.extend(texts)
            user_index.extend([i] * len(texts))
            labels.append(label)

        encoded = tokenizer(
            all_texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "user_index": torch.tensor(user_index, dtype=torch.long),
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

    def forward(self, input_ids, attention_mask, user_index, n_users):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        review_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] token per review

        hidden_size = review_vectors.shape[1]
        device = review_vectors.device
        sums = torch.zeros(n_users, hidden_size, device=device)
        sums.index_add_(0, user_index, review_vectors)
        counts = torch.zeros(n_users, device=device)
        counts.index_add_(0, user_index, torch.ones_like(user_index, dtype=torch.float32))
        pooled = sums / counts.unsqueeze(1).clamp(min=1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits


def train_one_fold(
    train_bags, train_y, test_bags, test_y, tokenizer, device,
    epochs, batch_size, learning_rate, max_reviews_per_user_train,
) -> np.ndarray:
    """Trains a fresh MILToxicityClassifier on this fold's training bags,
    returns predicted probabilities for the test bags (evaluated with
    every review a test user has, uncapped)."""
    model = MILToxicityClassifier(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    collate_fn = make_collate_fn(tokenizer)
    train_dataset = BagDataset(train_bags, train_y, max_reviews=max_reviews_per_user_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        bar = tqdm(train_loader, desc=f"  epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in bar:
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["user_index"].to(device), batch["n_users"],
            )
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            bar.set_postfix(loss=epoch_loss / (bar.n + 1))
        info(f"  epoch {epoch + 1}/{epochs} - mean loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    test_dataset = BagDataset(test_bags, test_y, max_reviews=None)  # uncapped at eval time
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  evaluating", unit="batch"):
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["user_index"].to(device), batch["n_users"],
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs.tolist())

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

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

    completed_folds = {}
    if fold_results_path.exists():
        existing = pd.read_parquet(fold_results_path)
        completed_folds = {int(f): sub for f, sub in existing.groupby("fold")}
        info(f"[checkpoint] Resuming: {len(completed_folds)} fold(s) already completed")

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

        predictions = train_one_fold(
            train_bags, train_y, test_bags, test_y, tokenizer, device,
            args.epochs, args.batch_size, args.learning_rate, args.max_reviews_per_user_train,
        )

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
