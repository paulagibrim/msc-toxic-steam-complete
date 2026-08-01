#!/bin/bash
# Computa 1 embedding por review (en, step07) numa A100 do cluster da UFV.
# Sem array - e um unico passe de encoding, nao um treino em folds.
#SBATCH --job-name=step07_embed_en
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

cd "$HOME/msc-toxic-steam-complete/step07_toxicity_replication"

DATA="$HOME/steam-data/step07-output"

python 02_compute_embeddings.py \
  --review-dataset "$DATA/review_dataset_en.parquet" \
  --output         "$DATA/review_embeddings_en.npy" \
  --batch-size 256
