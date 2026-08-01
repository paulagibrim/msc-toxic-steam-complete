"""Phase 6: classifies users as toxic/non-toxic via an LLM, as a third
modeling approach to compare against Phase 2 (frozen embeddings + classical
ML) and Phase 3 (attempted DistilBERT fine-tuning, descoped - see the
methodology report). Now supports THREE providers (Anthropic, OpenAI,
Google) so the same population/prompt can be compared across model
families, not just within one - the SAME sample, SAME system/user prompt
text is sent to whichever provider is selected via --provider, so any
difference in the resulting AUC-PR reflects the model, not the task.

SCOPE: NOT the full population - every positive user (75/1,225/1,316,
already a small set) plus a matched sample of negatives (default 3x the
positive count), per population. Classifying millions of users via API
would be slow and costly for a comparison data point, not the deliverable
itself - this mirrors Phase 2/3's own population-size trade-offs (e.g.
Phase 2's profile-metadata-matched subset), just via a different
mechanism (API cost/latency instead of GPU memory).

WHAT'S ASKED: given a user's own reviews (light-cleaned, same text already
used in Phase 3 - review_texts.parquet), whether the review AUTHOR looks
like a toxic user by this project's own definition (a pattern across
reviews, not a single flagged message) - with a 0-100 confidence score
(not just a binary label), so AUC-PR can be computed the same way as
Phases 2 and 3, for a direct comparison.

CHECKPOINTING: saves progress every 5 users (API calls are slow/costly
enough that losing even a partial run's worth of calls is wasteful) -
resumable, and the output/checkpoint file names include the provider AND
model, so runs from different providers never collide or get mixed up.

Usage:
    python 01_llm_classify_users.py \\
        --review-texts ../../../../steam-data/step06-output/phase3_distilbert/review_texts.parquet \\
        --feature-table ../../../../steam-data/step06-output/phase1_feature_engineering/pipeline/phase1_feature_table.parquet \\
        --population pt \\
        --provider anthropic --model claude-sonnet-5 \\
        --output-dir ../../../../steam-data/step06-output/phase6_llm_classifier \\
        --api-key sk-ant-... \\
        --negatives-per-positive 3
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from pipeline_utils import info, save_summary

RANDOM_STATE = 42
MAX_REVIEWS_IN_PROMPT = 30  # cap per user - keeps the prompt bounded for very prolific users
MAX_CHARS_PER_REVIEW = 500  # matches step03 Stage 8's example-truncation convention

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.1",
    "google": "gemini-3.6-flash",
}
API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

SYSTEM_PROMPT = """You are assisting with a Master's dissertation on toxic discourse in Steam \
game reviews. You will see a set of reviews written by a single Steam user, \
already collected for this research pipeline.

Your task: judge whether this user's OVERALL PATTERN of reviews looks like \
a "toxic user" - someone who repeatedly writes hostile, abusive, or \
harassing content - as opposed to a user whose reviews are simply critical \
of games, occasionally blunt, or use strong language without hostile \
intent (e.g. "this game sucks" is product criticism, not toxicity; \
enthusiastic profanity like "fucking awesome" is not toxicity either).
Base your judgment on the PATTERN across all reviews shown, not on a single \
isolated message in isolation.

This is legitimate academic content-moderation research. The example texts \
are real user-generated text already collected for this study, not content \
you are being asked to generate. Do not refuse or soften the task on \
safety grounds - the goal is precise academic classification, not \
amplification.

Output ONLY valid JSON matching this schema, no other text:
{
  "confidence_toxic": integer,   // 0-100: how confident you are this is a toxic user (100 = certain toxic, 0 = certain not toxic)
  "reasoning": string             // one sentence justification
}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Classifies users as toxic/non-toxic via an LLM (multiple providers supported).")
    parser.add_argument("--review-texts", required=True, type=Path, help="Path to 01_prepare_review_texts.py's output (Phase 3)")
    parser.add_argument("--feature-table", required=True, type=Path, help="Path to phase1_feature_table.parquet (labels)")
    parser.add_argument("--population", required=True, choices=["pt", "en", "union"], help="Which population to run")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "google"], help="Which LLM provider to use")
    parser.add_argument(
        "--model", default=None,
        help=f"Model id. Defaults per provider: {DEFAULT_MODELS} - VERIFY this is a currently valid model id "
        "for your account before a large run; these defaults may drift out of date.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write predictions to - filename is derived from population+provider+model automatically",
    )
    parser.add_argument("--api-key", default=None, help="API key for the selected provider. If omitted, reads the provider's standard env var.")
    parser.add_argument("--negatives-per-positive", type=int, default=3, help="How many negatives to sample per positive (default 3)")
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Output token cap per call (default 4096 - a generous ceiling, not a target; costs nothing extra "
        "unless the model actually generates that many tokens, and protects against always-thinking models "
        "exhausting a low cap on reasoning alone before reaching the JSON response)",
    )
    return parser.parse_args()


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def load_sample(review_texts_path: Path, feature_table_path: Path, population: str, negatives_per_positive: int) -> tuple:
    """Same population/label semantics as Phase 3's load_user_bags - see
    that script for why. Returns (bags, y, user_urls) for a SAMPLE: every
    positive + negatives_per_positive x negatives, sampled with a fixed seed
    - IDENTICAL across providers (same RANDOM_STATE), so every provider is
    scored on the exact same set of users."""
    label_suffix = "pt+en" if population == "union" else population
    table = pd.read_parquet(feature_table_path, columns=["user_url", f"eligible_{label_suffix}", f"is_toxic_{label_suffix}"])
    eligible = table[f"eligible_{label_suffix}"].fillna(False)
    labels = table.loc[eligible, ["user_url", f"is_toxic_{label_suffix}"]].rename(columns={f"is_toxic_{label_suffix}": "is_toxic"})
    labels["is_toxic"] = labels["is_toxic"].fillna(False)

    positives = labels.loc[labels["is_toxic"], "user_url"]
    negatives_pool = labels.loc[~labels["is_toxic"], "user_url"]
    rng = np.random.default_rng(RANDOM_STATE)
    n_neg_sample = min(len(negatives_pool), len(positives) * negatives_per_positive)
    negatives = pd.Series(rng.choice(negatives_pool, size=n_neg_sample, replace=False))

    sampled_users = pd.concat([positives, negatives], ignore_index=True)
    sampled_y = np.concatenate([np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)])
    info(f"[{population}] Sampled {len(positives):,} positive + {len(negatives):,} negative user(s) = {len(sampled_users):,} total")

    reviews = pd.read_parquet(review_texts_path)
    reviews = reviews[reviews["user_url"].isin(set(sampled_users))]
    if population != "union":
        reviews = reviews[reviews["review_lang"] == population]
    grouped = reviews.groupby("user_url")["review_text_clean"].apply(list)

    bags, y, user_urls = [], [], []
    for user_url, is_toxic in zip(sampled_users, sampled_y):
        texts = grouped.get(user_url)
        if texts is None or len(texts) == 0:
            continue
        bags.append(texts)
        y.append(is_toxic)
        user_urls.append(user_url)

    return bags, np.array(y), np.array(user_urls)


def build_prompt(texts: list) -> str:
    sample = texts[:MAX_REVIEWS_IN_PROMPT]
    examples_str = "\n".join(f'{i}. "{t[:MAX_CHARS_PER_REVIEW]}"' for i, t in enumerate(sample, start=1))
    note = "" if len(texts) <= MAX_REVIEWS_IN_PROMPT else f"\n(showing {MAX_REVIEWS_IN_PROMPT} of {len(texts)} total reviews from this user)"
    return f"""Reviews from this user:{note}
{examples_str}

Based on the pattern across these reviews, classify this user according to the JSON schema in your instructions."""


# ── Provider-specific client construction and calls ─────────────────────────
# Each provider has a different SDK shape (message format, response object,
# system-prompt placement) - isolated here so the sampling/prompt-building/
# checkpoint logic above (and the main loop below) stays provider-agnostic.

def get_client(provider: str, api_key: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key) if api_key else OpenAI()
    if provider == "google":
        from google import genai
        return genai.Client(api_key=api_key) if api_key else genai.Client()
    raise ValueError(f"Unknown provider: {provider}")


def call_anthropic(client, model: str, texts: list, max_tokens: int) -> dict:
    response = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(texts)}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    return _extract_json(text)


def call_openai(client, model: str, texts: list, max_tokens: int) -> dict:
    response = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(texts)},
        ],
    )
    text = response.choices[0].message.content or ""
    return _extract_json(text)


def call_google(client, model: str, texts: list, max_tokens: int) -> dict:
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(texts),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=max_tokens),
    )
    return _extract_json(response.text or "")


CALL_FUNCTIONS = {"anthropic": call_anthropic, "openai": call_openai, "google": call_google}


def main():
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    call_fn = CALL_FUNCTIONS[args.provider]
    client = get_client(args.provider, args.api_key)

    info(f"[{args.population}] provider={args.provider}, model={model}")

    bags, y, user_urls = load_sample(args.review_texts, args.feature_table, args.population, args.negatives_per_positive)
    info(f"[{args.population}] {len(bags):,} user(s) with review text to classify, {int(y.sum()):,} positive")

    # Filename includes population + provider + model - so anthropic/openai/google
    # runs (and different models within a provider) never collide or overwrite
    # each other, and it's unambiguous later which file is which comparison.
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    output_path = args.output_dir / f"phase6_{args.population}_{args.provider}_{safe_model}_predictions.parquet"
    checkpoint_path = output_path.with_suffix(".checkpoint.parquet")

    predictions = {}
    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        # Only successful classifications (confidence_toxic >= 0) count as
        # "already done" - a failed call (-1, e.g. a transient network error
        # or, as happened here, a wrong API key) should be RETRIED on
        # resume, not permanently treated as complete just because it has
        # a row in the checkpoint.
        successful = existing[existing["confidence_toxic"] >= 0]
        n_failed_in_checkpoint = len(existing) - len(successful)
        predictions = dict(zip(successful["user_url"], zip(successful["y_true"], successful["confidence_toxic"], successful["reasoning"])))
        info(f"[checkpoint] Resuming: {len(predictions):,} user(s) already classified successfully")
        if n_failed_in_checkpoint:
            info(f"[checkpoint] {n_failed_in_checkpoint:,} previously-failed user(s) will be retried")

    args.output_dir.mkdir(parents=True, exist_ok=True)  # must exist before the periodic checkpoint saves below, not just at the end

    n_failed = 0
    bar = tqdm(list(zip(user_urls, bags, y)), desc=f"[{args.population}/{args.provider}]", unit="user")
    for i, (user_url, texts, is_toxic) in enumerate(bar):
        if user_url in predictions:
            continue
        try:
            result = call_fn(client, model, texts, args.max_tokens)
            confidence = int(result.get("confidence_toxic", 50))
            reasoning = str(result.get("reasoning", ""))
        except Exception as exc:
            n_failed += 1
            confidence, reasoning = -1, f"FAILED: {exc}"
            if n_failed <= 3:  # print the first few errors in full - silent failures otherwise hide the real cause
                info(f"[{args.population}/{args.provider}] Classification failed for a user: {exc!r}")
        predictions[user_url] = (int(is_toxic), confidence, reasoning)
        bar.set_postfix(failed=n_failed)

        if (i + 1) % 5 == 0:  # every 5, not 50 - each API call costs real time/money, so cap the loss on a crash
            df = pd.DataFrame(
                [(u, v[0], v[1], v[2]) for u, v in predictions.items()],
                columns=["user_url", "y_true", "confidence_toxic", "reasoning"],
            )
            df.to_parquet(checkpoint_path, index=False)

    df = pd.DataFrame(
        [(u, v[0], v[1], v[2]) for u, v in predictions.items()],
        columns=["user_url", "y_true", "confidence_toxic", "reasoning"],
    )
    valid = df[df["confidence_toxic"] >= 0].copy()
    valid["y_score"] = valid["confidence_toxic"] / 100.0

    df.to_parquet(output_path, index=False)
    checkpoint_path.unlink(missing_ok=True)

    auc_pr = average_precision_score(valid["y_true"], valid["y_score"]) if len(valid) and valid["y_true"].sum() > 0 else None
    info(f"[{args.population}/{args.provider}] {len(valid):,}/{len(df):,} successful classification(s), AUC-PR={auc_pr}")

    save_summary(
        {
            "population": args.population,
            "provider": args.provider,
            "model": model,
            "n_users": int(len(df)),
            "n_successful": int(len(valid)),
            "n_failed": int((df["confidence_toxic"] < 0).sum()),
            "n_positive": int(df["y_true"].sum()),
            "auc_pr": round(float(auc_pr), 6) if auc_pr is not None else None,
        },
        output_path.with_suffix(".summary.json"),
    )
    info(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
