#!/bin/bash
# pooling_ablation.py sobre embeddings DistilBERT (768-dim) por review, em vez
# de MiniLM (384-dim) - mesma ablation (mean / mean+max / mean+max+topk x
# logreg/svm_linear/svm_rbf/xgboost x circular/leave_toxic_out), so trocando o
# encoder base. Objetivo: comparar MiniLM vs DistilBERT como representacao,
# mantendo pooling e classificador fixos. SEM metadados (--with-metadata nao
# usado de proposito: ja provamos que metadados nao ajudam e tinham vazamento).
#
# POPULACAO COMPLETA (nao a restrita a perfis casados): usa
# toxic_user_labels.parquet diretamente (eligible_<pop>/is_toxic_<pop> ja vem
# pronto de la), a MESMA populacao da linha B/fullpop - pt 490.470 usuarios/160
# positivos, en maior/2.479 positivos - em vez do labels_slim.parquet restrito
# (75/1.225 positivos) usado na primeira rodada da ablation.
#
# Pre-requisito: review_embeddings_distilbert_${POP}.npy precisa existir -
# gerado por step07_toxicity_replication/rodar_embed_review_distilbert.slurm
# (ja cobre qualquer populacao: e por-review, sem filtro de elegibilidade)
#
# Submeta a populacao:
#   sbatch --export=ALL,POP=pt rodar_ablation_distilbert.slurm
#   sbatch --export=ALL,POP=en rodar_ablation_distilbert.slurm
#SBATCH --job-name=ablation_distilbert
#SBATCH --partition=scientific
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# mem/time subiram frente a rodada anterior (250G/12h, populacao restrita
# 2,19M usuarios, MiniLM 384-dim): populacao completa do en tem 5,1M usuarios
# elegiveis (vs 2,19M) sobre um pool de 9,7M usuarios com review, e DistilBERT
# e 768-dim (2x MiniLM) - tanto mais reviews retidas pro top-k quanto os
# arrays mean/max maiores. pt e bem mais leve (490k usuarios); os limites
# generosos so custam fila, nao rodada, se sobrar.

ulimit -d unlimited
ulimit -t unlimited           # sem isso, o job do 4722 morreu em "CPU time limit exceeded" as 21h23
export EMBEDDING_DIM=768      # DistilBERT tem 768 dims (default do script e 384/MiniLM)

source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

: "${POP:?defina a populacao, ex.: sbatch --export=ALL,POP=pt rodar_ablation_distilbert.slurm}"

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase4_text_only_fullpop"
DATA7="$HOME/steam-data/step07-output"
DATA6="$HOME/steam-data/step06-output/phase3_distilbert"
LABELS_FULLPOP="$HOME/steam-data/step06-output/phase1_feature_engineering/pipeline/toxic_user_labels.parquet"

python pooling_ablation.py --language "$POP" \
  --embeddings "$DATA7/review_embeddings_distilbert_${POP}.npy" \
  --review-dataset "$DATA7/review_dataset_${POP}.parquet" \
  --labels "$LABELS_FULLPOP" \
  --models logreg,svm_linear,svm_rbf,xgboost --topk 3 \
  --output "$DATA6/pooling_${POP}_distilbert_fullpop.json"
