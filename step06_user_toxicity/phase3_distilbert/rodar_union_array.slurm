#!/bin/bash
# Treina os 5 folds do phase3 da populacao 'union' (pt+en, ~2,38M usuarios) em PARALELO.
# Igual ao en: --array=0-4%4 (5 folds, ate 4 GPUs por vez). Cada tarefa roda 1 fold.
# Depois, junte com merge_folds.py (comando no fim deste arquivo).
# OBS: precisa do labels_slim.parquet COM as colunas eligible_pt+en/is_toxic_pt+en
#      (o codigo mapeia --population union -> sufixo 'pt+en').
#SBATCH --job-name=phase3_union
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

python 02_train_mil_distilbert.py \
  --review-texts  "$DATA/review_texts.parquet" \
  --feature-table "$DATA/labels_slim.parquet" \
  --population union \
  --output        "$DATA/phase3_union_oof.parquet" \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --n-folds 5 --epochs 3 --batch-size 32

# Depois que os 5 folds terminarem (squeue vazio), junte no login node:
#   source ~/msc-toxic-steam-complete/.venv/bin/activate
#   python ~/msc-toxic-steam-complete/step06_user_toxicity/phase3_distilbert/merge_folds.py \
#     --fold-glob "$HOME/steam-data/step06-output/phase3_distilbert/phase3_union_oof.fold*.parquet" \
#     --output    "$HOME/steam-data/step06-output/phase3_distilbert/phase3_union_oof.parquet" \
#     --population union --expected-folds 5
