#!/bin/bash
# Treina o phase3 da populacao 'pt' num job UNICO (5 folds em sequencia, 1 A100).
# pt e pequeno (179k usuarios, 75 positivos) -> nao vale a pena paralelizar por fold:
# roda em poucas horas e ja escreve phase3_pt_oof.parquet + AUC-PR (SEM precisar de merge).
# Se passar do --time, o checkpoint retoma de onde parou ao re-submeter.
#SBATCH --job-name=phase3_pt
#SBATCH --partition=scientific
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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
  --population pt \
  --output        "$DATA/phase3_pt_oof.parquet" \
  --n-folds 5 --epochs 3 --batch-size 32
