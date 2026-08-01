#!/bin/bash
# Fine-tuning phase3 'pt', POPULACAO COMPLETA (490.470 elegiveis, 160
# positivos - vs a rodada restrita anterior, 179.042/75). Job unico (5 folds
# em sequencia, 1 A100) - pt fullpop ainda e pequeno o bastante pra nao valer
# paralelizar por fold. Checkpoint retoma se cair.
# PRE-REQUISITO: rodar_prepare_review_texts_fullpop.slurm precisa ter terminado.
#SBATCH --job-name=phase3_pt_fullpop
#SBATCH --partition=scientific
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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
  --population pt \
  --output        "$DATA/phase3_pt_fullpop_oof.parquet" \
  --n-folds 5 --epochs 3 --batch-size 32
