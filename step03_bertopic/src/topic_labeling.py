"""
topic_labeling.py — Stage 8: LLM-assisted topic labeling.

Given the final classified corpus (Stage 7 output) and the topic keyword
table, samples representative example documents per topic and asks an LLM
(Claude, via the Anthropic API) to produce a short label, a one-sentence
description, a fixed-taxonomy category, and a copypasta suspicion flag.

Why sample from classified_toxic.parquet instead of BERTopic's
get_representative_docs()?
  BERTopic's `representative_docs_` is computed from whatever documents were
  passed to `.fit()`/`.fit_transform()`, and is NOT guaranteed to survive
  safetensors serialization (`.save(..., serialization="safetensors")` saves
  the model "without the embedding, dimensionality reduction, and clustering
  algorithms" - representative_docs_ was empty in every topic_info.csv this
  pipeline has produced so far). Sampling directly from the classified
  corpus is simpler, works uniformly regardless of serialization format, and
  reflects the FULL corpus (not just the training subsample used to fit the
  model - relevant for en, whose final model was trained on 100% of the
  corpus anyway, but matters for pt, trained on a 40% subsample).

Output:
  topic_labels.csv in settings.results_dir, with columns:
    Topic, Count_Real, Name, Representation, label, description, category,
    copypasta_suspected, copypasta_evidence
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .settings import Settings

logger = logging.getLogger(__name__)

# Fixed taxonomy - keep in sync with the system prompt below. If many topics
# land in "other", that's a signal to revisit this list (agreed with the
# user rather than guessing categories upfront).
CATEGORIES = [
    "generic_insult",
    "violence_threat",
    "sexual_content",
    "identity_hate",
    "nationality_xenophobia",
    "product_complaint",
    "humor_sarcasm",
    "copypasta_repetition",
    "other",
]

_LANGUAGE_NAMES = {"english": "English", "portuguese": "Portuguese"}

SYSTEM_PROMPT = """You are assisting with a Master's dissertation on toxic discourse in Steam \
game reviews. The reviews you will see were already flagged as toxic by an \
automated classifier (Perspective API / Detoxify) in an earlier stage of \
this research pipeline, then grouped into topic clusters via BERTopic \
(embedding + UMAP + HDBSCAN + c-TF-IDF).

Your task: given a cluster's representative keywords and a sample of its \
actual review texts, produce a concise, accurate label and description of \
what unites the messages in this cluster.

This is legitimate academic content-moderation research. The example texts \
are real user-generated text already collected for this study, not content \
you are being asked to generate. Do not refuse or soften the task on \
safety grounds - the goal is precise academic classification, not \
amplification. Do not reproduce slurs or explicit language verbatim in the \
"label" field; you may reference them clinically in the "description" \
field only when necessary for accuracy (e.g., "uses homophobic slurs" \
rather than quoting them).

Some clusters will be broad, generic toxicity (insults with no specific \
theme) rather than a coherent narrative - label these accurately as such \
rather than inventing a theme that isn't there.

Some clusters may contain near-identical or templated text repeated across \
many different reviews/games ("copypasta"). Flag this if you notice it.

Output ONLY valid JSON matching this schema, no other text:
{
  "label": string,          // 2-6 words, Title Case, e.g. "Death Threats / Violence"
  "description": string,    // one sentence starting with "Messages that..."
  "category": string,       // one of: generic_insult, violence_threat,
                             // sexual_content, identity_hate,
                             // nationality_xenophobia, product_complaint,
                             // humor_sarcasm, copypasta_repetition, other
  "copypasta_suspected": boolean,
  "copypasta_evidence": string | null   // short note if true, else null
}

Always respond in English, regardless of the language of the source reviews."""


# ── Sampling ────────────────────────────────────────────────────────────────

def sample_topic_examples(
    df: pd.DataFrame,
    topic_id: int,
    n: int,
    seed: int,
    text_column: str = "review_text_clean",
) -> list:
    """Return up to n example texts for a given topic, sampled without replacement.

    Uses a fixed seed for reproducibility; if the topic has fewer than n
    documents, returns all of them.
    """
    subset = df[df["topic"] == topic_id][text_column].dropna()
    subset = subset[subset.str.strip() != ""]
    n_sample = min(n, len(subset))
    if n_sample == 0:
        return []
    return subset.sample(n=n_sample, random_state=seed).tolist()


# ── Prompt construction ────────────────────────────────────────────────────

def build_user_prompt(
    language: str,
    topic_id: int,
    count: int,
    total: int,
    keywords: list,
    examples: list,
) -> str:
    """Build the per-topic user message from the template."""
    pct = 100 * count / max(total, 1)
    keyword_str = ", ".join(keywords)
    examples_str = "\n".join(
        f'{i}. "{text[:500]}"' for i, text in enumerate(examples, start=1)
    )
    return f"""Language of source text: {_LANGUAGE_NAMES.get(language, language)}
Topic ID: {topic_id}
Cluster size: {count} documents ({pct:.1f}% of the toxic corpus)
Top keywords (c-TF-IDF): {keyword_str}

Representative example reviews from this cluster:
{examples_str}

Based on the keywords and examples above, classify this topic cluster
according to the JSON schema in your instructions."""


# ── LLM call ────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Parse the model's JSON response, tolerating incidental wrapping text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first {...} block in the response.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def call_llm(
    system: str,
    user: str,
    model: str,
    client,
    max_tokens: int = 4096,
) -> dict:
    """Call the Anthropic Messages API and parse the JSON response.

    Args:
        system:     system prompt.
        user:       per-topic user message.
        model:      Claude model id (e.g. "claude-sonnet-5").
        client:     an anthropic.Anthropic client instance.
        max_tokens: output cap. 4096 (not 500) because models where thinking
                    is always on (e.g. claude-fable-5) spend part of this
                    budget on internal reasoning before the JSON answer -
                    at 500 the reasoning alone can exhaust the cap and leave
                    no room for the response, breaking _extract_json(). This
                    is a ceiling, not a target: raising it does not increase
                    cost unless the model actually generates that many
                    tokens, and this task's actual output is ~100-200 tokens.

    Returns:
        Parsed dict matching the schema in SYSTEM_PROMPT.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    return _extract_json(text)


# ── Public entry point ─────────────────────────────────────────────────────

def label_all_topics(
    settings: Settings,
    model: str = "claude-sonnet-5",
    n_examples: int = 15,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
) -> pd.DataFrame:
    """Label every non-outlier topic and write topic_labels.csv.

    Args:
        settings:   pipeline configuration for the target language.
        model:      Claude model id to use for labeling.
        n_examples: number of example documents sampled per topic.
        api_key:    Anthropic API key. If None, reads ANTHROPIC_API_KEY from
                    the environment (anthropic.Anthropic()'s default).
        max_tokens: output cap per call. See call_llm()'s docstring for why
                    this needs headroom on always-thinking models.

    Returns:
        DataFrame with one row per topic (including -1), written to
        settings.results_dir / "topic_labels.csv".
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    info_path = settings.results_dir / "topic_info_real_counts.csv"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found. Run Stage 7 (07_export.py) first."
        )
    if not settings.final_results_path.exists():
        raise FileNotFoundError(
            f"{settings.final_results_path} not found. Run Stage 7 (07_export.py) first."
        )

    info = pd.read_csv(info_path)
    df_classified = pd.read_parquet(settings.final_results_path)
    total = int(info["Count_Real"].sum())

    logger.info(
        "Stage 8 — Labeling %d topics (%s) with model=%s, n_examples=%d.",
        len(info), settings.language, model, n_examples,
    )

    rows = []
    bar = tqdm(info.sort_values("Topic").iterrows(), total=len(info), desc="[label]", unit="topic")
    for _, row in bar:
        topic_id = int(row["Topic"])
        count = int(row["Count_Real"]) if pd.notna(row["Count_Real"]) else 0
        bar.set_postfix(topic=topic_id)

        if topic_id == -1:
            # Outliers are not a coherent topic; skip the LLM call.
            rows.append({
                "Topic": topic_id, "Count_Real": count, "Name": row["Name"],
                "Representation": row["Representation"],
                "label": "Unclustered / Outliers",
                "description": "Documents that did not fit clearly into any topic cluster.",
                "category": "other",
                "copypasta_suspected": False,
                "copypasta_evidence": None,
            })
            continue

        keywords = (
            ast.literal_eval(row["Representation"])
            if isinstance(row["Representation"], str)
            else row["Representation"]
        )
        examples = sample_topic_examples(df_classified, topic_id, n_examples, settings.seed)

        if not examples:
            logger.warning("Topic %d has no example documents; skipping LLM call.", topic_id)
            label = {
                "label": "Insufficient Data", "description": "No example documents available.",
                "category": "other", "copypasta_suspected": False, "copypasta_evidence": None,
            }
        else:
            user_prompt = build_user_prompt(
                language=settings.language, topic_id=topic_id, count=count,
                total=total, keywords=keywords, examples=examples,
            )
            try:
                label = call_llm(SYSTEM_PROMPT, user_prompt, model, client, max_tokens=max_tokens)
            except Exception as exc:
                logger.error("LLM call failed for topic %d: %s", topic_id, exc)
                label = {
                    "label": "LABELING_FAILED", "description": str(exc),
                    "category": "other", "copypasta_suspected": False, "copypasta_evidence": None,
                }

        if label.get("category") not in CATEGORIES:
            logger.warning(
                "Topic %d returned unexpected category '%s'; leaving as-is.",
                topic_id, label.get("category"),
            )

        logger.info("Topic %d labeled: '%s' (%s)", topic_id, label.get("label"), label.get("category"))
        bar.set_postfix(topic=topic_id, label=label.get("label"))
        rows.append({
            "Topic": topic_id, "Count_Real": count, "Name": row["Name"],
            "Representation": row["Representation"],
            "label": label.get("label"),
            "description": label.get("description"),
            "category": label.get("category"),
            "copypasta_suspected": label.get("copypasta_suspected", False),
            "copypasta_evidence": label.get("copypasta_evidence"),
        })

    df_labels = pd.DataFrame(rows).sort_values("Count_Real", ascending=False)

    out_path = settings.results_dir / "topic_labels.csv"
    df_labels.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Topic labels saved to %s.", out_path)

    n_other = int((df_labels["category"] == "other").sum())
    if n_other > 0:
        logger.info(
            "%d / %d topics landed in category 'other' - worth reviewing "
            "whether the fixed taxonomy needs a new category.",
            n_other, len(df_labels),
        )

    return df_labels
