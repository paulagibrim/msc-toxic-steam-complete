#!/bin/bash
# Regera review_texts.parquet (entrada do fine-tuning) para a POPULACAO
# COMPLETA (toxic_user_labels.parquet: pt 490.470/en 5.111.196 elegiveis),
# em vez da populacao restrita a perfis casados (phase1_feature_table.parquet,
# 2.375.760 usuarios) que o arquivo atual usa. So CPU (varre step02-dir e
# limpa texto - nao carrega nenhum modelo).
#SBATCH --job-name=prep_review_texts_fullpop
#SBATCH --partition=scientific
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase3_distilbert"

LABELS_FULLPOP="$HOME/steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet"
DATA="$HOME/steam-data/step06-output/phase3_distilbert"

python 01_prepare_review_texts.py \
  --step02-dir "$HOME/steam-data/step02-output" \
  --feature-table "$LABELS_FULLPOP" \
  --output "$DATA/review_texts_fullpop.parquet"
