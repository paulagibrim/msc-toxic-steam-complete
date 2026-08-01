#!/bin/bash
# Passo 2 do experimento DistilBERT-congelado: o MESMO LOO aninhado da Fase 2, em
# --text-only, sobre os embeddings DistilBERT (768-dim). Reusa 01_phase2_nested_cv.py
# sem modificacao, so passando EMBEDDING_DIM=768 (o codigo le do ambiente; default 384).
#
# Submeta a populacao:
#   sbatch --export=ALL,POP=pt rodar_loo_distilbert.slurm
#
# MEMORIA: 768-dim = ~2x o phase4 (384). pt e leve (poucas dobras LOO); en/union sao
# pesados -> se for rodar en, suba --mem e/ou reduza N_JOBS (ha checkpoint a cada 20).
#SBATCH --job-name=distilbert_loo
#SBATCH --partition=scientific
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err

ulimit -d unlimited
export EMBEDDING_DIM=768      # DistilBERT tem 768 dims (default do script e 384/MiniLM)

source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

: "${POP:?defina a populacao, ex.: sbatch --export=ALL,POP=pt rodar_loo_distilbert.slurm}"
N_JOBS=${N_JOBS:-8}

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase2_classical_ml"

DATA="$HOME/steam-data/step06-output/phase4_text_only_fullpop"

python 01_phase2_nested_cv.py \
  --feature-table "$DATA/distilbert_${POP}.parquet" \
  --population    "$POP" \
  --text-only \
  --output        "$DATA/phase4_distilbert_${POP}_oof_predictions.parquet" \
  --n-jobs "$N_JOBS"
