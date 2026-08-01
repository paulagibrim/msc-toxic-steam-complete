#!/bin/bash
# Fine-tuning phase3 'en', POPULACAO COMPLETA (5.111.196 elegiveis, 2.479
# positivos - vs a rodada restrita anterior, 2.191.286/1.225). 5 folds em
# PARALELO nas A100 (limite de 4 GPUs/usuario -> %4).
# PRE-REQUISITO: rodar_prepare_review_texts_fullpop.slurm precisa ter terminado.
# ATENCAO GPU: se ja houver outros jobs de GPU seus rodando (embeds, en_lto
# antigo etc.), o total pode estourar o teto de 4 GPU/usuario do QOS - confira
# squeue -u es90366 -r antes de submeter.
# Depois que os 5 folds terminarem, junte com merge_folds.py (comando no fim).
#SBATCH --job-name=phase3_en_fullpop
#SBATCH --partition=scientific
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4%4
#SBATCH --output=%x_fold%a_%A.out
#SBATCH --error=%x_fold%a_%A.err

module load CUDA/12.6.0
source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase3_distilbert"

DATA="$HOME/steam-data/step06-output/phase3_distilbert"
LABELS_FULLPOP="$HOME/steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet"

python 02_train_mil_distilbert.py \
  --review-texts  "$DATA/review_texts_fullpop.parquet" \
  --feature-table "$LABELS_FULLPOP" \
  --population en \
  --output        "$DATA/phase3_en_fullpop_oof.parquet" \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --n-folds 5 --epochs 3 --batch-size 32

# Depois que os 5 folds terminarem (squeue vazio), junte tudo no login node:
#   cd ~/msc-toxic-steam-complete/step06_user_toxicity/phase3_distilbert
#   source ~/msc-toxic-steam-complete/.venv/bin/activate
#   python merge_folds.py \
#     --fold-glob "$HOME/steam-data/step06-output/phase3_distilbert/phase3_en_fullpop_oof.fold*.parquet" \
#     --output    "$HOME/steam-data/step06-output/phase3_distilbert/phase3_en_fullpop_oof.parquet" \
#     --population en --expected-folds 5
