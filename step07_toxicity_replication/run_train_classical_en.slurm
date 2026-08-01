#!/bin/bash
# Treina os modelos classicos (LogReg + HistGB) do step07 pra 'en', so CPU -
# nao pede GPU, entao nao compete pela fila QOSGrpGRES que trava os jobs
# de GPU (phase3, step07 embed).
#SBATCH --job-name=step07_train_en
#SBATCH --partition=scientific
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

cd "$HOME/msc-toxic-steam-complete/step07_toxicity_replication"

DATA="$HOME/steam-data/step07-output"

python 04_train_classical_models.py \
  --review-dataset "$DATA/review_dataset_en.parquet" \
  --embeddings     "$DATA/review_embeddings_en.npy" \
  --language en \
  --use-pca no \
  --output "$DATA/review_classical_en_oof.parquet" \
  --n-jobs 32
