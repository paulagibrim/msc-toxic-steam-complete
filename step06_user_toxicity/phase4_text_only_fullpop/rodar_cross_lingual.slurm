#!/bin/bash
# Transferencia cross-lingual: treina no idioma inteiro (pt OU en, populacao
# completa), avalia no OUTRO idioma inteiro. As duas direcoes (pt->en e
# en->pt) x as duas condicoes (circular/leave_toxic_out) rodam num job so -
# so 4 combinacoes, cada uma SEM k-fold (1 treino, 1 teste), entao mais
# barato que uma rodada da ablation normal (que faz 5-fold x 2 condicoes).
#
# USA DISTILBERT (fullpop) - os mesmos embeddings/labels da ablation Linha C
# ja rodada, nada novo pra reencodar.
#
# Pre-requisitos (ja existem): review_embeddings_distilbert_pt.npy,
# review_embeddings_distilbert_en.npy, review_dataset_pt/en.parquet,
# toxic_user_labels.parquet.
#SBATCH --job-name=cross_lingual
#SBATCH --partition=scientific
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

ulimit -d unlimited
ulimit -t unlimited
export EMBEDDING_DIM=768      # DistilBERT

source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase4_text_only_fullpop"
DATA7="$HOME/steam-data/step07-output"
DATA6="$HOME/steam-data/step06-output/phase3_distilbert"
LABELS_FULLPOP="$HOME/steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet"

python cross_lingual_transfer.py \
  --embeddings-pt "$DATA7/review_embeddings_distilbert_pt.npy" \
  --embeddings-en "$DATA7/review_embeddings_distilbert_en.npy" \
  --review-dataset-pt "$DATA7/review_dataset_pt.parquet" \
  --review-dataset-en "$DATA7/review_dataset_en.parquet" \
  --labels "$LABELS_FULLPOP" \
  --models logreg,svm_linear,svm_rbf,xgboost --topk 3 \
  --output "$DATA6/cross_lingual_transfer.json"
