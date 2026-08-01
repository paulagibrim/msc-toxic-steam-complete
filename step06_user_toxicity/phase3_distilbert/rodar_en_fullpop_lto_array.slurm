#!/bin/bash
# Fine-tuning phase3 'en', POPULACAO COMPLETA, condicao HONESTA (leave-toxic-out).
# PRE-REQUISITO: rodar_prepare_review_texts_fullpop.slurm ja terminou
# (review_texts_fullpop.parquet existe).
# 5 folds em PARALELO nas A100 (%4 = teto de 4 GPU/usuario do QOS).
#
# ATENCAO TEMPO: o en_lto da populacao RESTRITA (2,19M usuarios) levava ~17h
# por fold; fullpop tem 5,11M usuarios (2,3x) - um fold pode passar do teto de
# 48h da fila. Se o job morrer por tempo, RESUBMETA O MESMO sbatch (mesmo
# --array=N) - o checkpoint por epoca/batch retoma de onde parou, nao perde
# o progresso.
#
# ATENCAO GPU: confira squeue -u es90366 -r antes de submeter - com o fullpop
# circular (phase3_pt_fullpop/phase3_en_fullpop) rodando, pode faltar vaga de
# GPU (teto 4/usuario) e esses folds ficam PD ate abrir espaco - normal.
#
# Depois que os 5 folds terminarem, junte com merge_folds.py (comando no fim).
#SBATCH --job-name=phase3_en_fullpop_lto
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
  --leave-toxic-out \
  --output        "$DATA/phase3_en_fullpop_lto_oof.parquet" \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --n-folds 5 --epochs 3 --batch-size 32

# Depois que os 5 folds terminarem (squeue vazio), junte tudo no login node:
#   cd ~/msc-toxic-steam-complete/step06_user_toxicity/phase3_distilbert
#   source ~/msc-toxic-steam-complete/.venv/bin/activate
#   python merge_folds.py \
#     --fold-glob "$HOME/steam-data/step06-output/phase3_distilbert/phase3_en_fullpop_lto_oof.fold*.parquet" \
#     --output    "$HOME/steam-data/step06-output/phase3_distilbert/phase3_en_fullpop_lto_oof.parquet" \
#     --population en --expected-folds 5
