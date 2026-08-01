#!/bin/bash
# Embeddings ESTATICOS (congelados) do DistilBERT-base, mean-pooling por usuario,
# populacao pt completa. Mesma logica do phase4 (rodar_embed_array.slurm), so troca
# o encoder MiniLM (384-dim) por DistilBERT-base-multilingual-cased (768-dim).
# Objetivo: comparar "encoder congelado vs fine-tunado" com o MESMO encoder do phase3.
# Ha checkpoint a cada 50 arquivos -> se cair, reenvie o mesmo sbatch que retoma.
#SBATCH --job-name=distilbert_embed_pt
#SBATCH --partition=scientific
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load CUDA/12.6.0
source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase4_text_only_fullpop"

P1="$HOME/steam-data/step06-output/phase1_feature_engineering/pipeline"
OUT="$HOME/steam-data/step06-output/phase4_text_only_fullpop"

python 01_build_distilbert_embeddings.py \
  --step02-dir  "$HOME/steam-data/step02-output" \
  --labels      "$P1/toxic_user_labels.parquet" \
  --population  pt \
  --output      "$OUT/distilbert_pt.parquet" \
  --batch-size 256 \
  --checkpoint-every 50
