#!/bin/bash
# Treina os 5 folds do phase3 (populacao 'en') em PARALELO nas A100.
# --array=0-4%4  -> 5 folds, no maximo 4 rodando ao mesmo tempo (limite de 4 GPUs/usuario).
# Cada tarefa roda 1 fold e grava phase3_en_oof.fold<N>.parquet.
# Depois que todas terminarem, junte com merge_folds.py (ver o comando no fim deste arquivo).
#SBATCH --job-name=phase3_en
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
  --population en \
  --output        "$DATA/phase3_en_oof.parquet" \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --n-folds 5 --epochs 3 --batch-size 32

# Depois que os 5 folds terminarem (squeue vazio), junte tudo no login node:
#   cd ~/msc-toxic-steam-complete/step06_user_toxicity/phase3_distilbert
#   source ~/msc-toxic-steam-complete/.venv/bin/activate
#   python merge_folds.py \
#     --fold-glob "$HOME/steam-data/step06-output/phase3_distilbert/phase3_en_oof.fold*.parquet" \
#     --output    "$HOME/steam-data/step06-output/phase3_distilbert/phase3_en_oof.parquet" \
#     --population en --expected-folds 5
