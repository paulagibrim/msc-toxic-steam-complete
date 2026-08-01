#!/bin/bash
# Embeddings DistilBERT por REVIEW (nao por usuario) - equivalente ao
# 02_compute_embeddings.py (MiniLM) do step07, so trocando o encoder.
# Objetivo: alimentar phase4_text_only_fullpop/pooling_ablation.py com
# EMBEDDING_DIM=768, pra comparar MiniLM vs DistilBERT como representacao
# base, mantendo pooling (mean/mean+max/topk) e classificador fixos.
#
# pt: 2.089.904 reviews | en: 25.341.249 reviews
# Estimativa (baseada no embed por-usuario do pt, ~694k reviews em ~10min
# numa A100): pt ~30min, en ~6h. Ha checkpoint (.progress.json) -> se cair,
# reenvie o mesmo sbatch que retoma.
#
# Submeta a populacao:
#   sbatch --export=ALL,POP=pt rodar_embed_review_distilbert.slurm
#   sbatch --export=ALL,POP=en rodar_embed_review_distilbert.slurm
#SBATCH --job-name=embed_review_distilbert
#SBATCH --partition=scientific
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load CUDA/12.6.0
source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

: "${POP:?defina a populacao, ex.: sbatch --export=ALL,POP=pt rodar_embed_review_distilbert.slurm}"

cd "$HOME/msc-toxic-steam-complete/step07_toxicity_replication"

DATA="$HOME/steam-data/step07-output"

python 02b_compute_embeddings_distilbert.py \
  --review-dataset "$DATA/review_dataset_${POP}.parquet" \
  --output         "$DATA/review_embeddings_distilbert_${POP}.npy" \
  --batch-size 256
