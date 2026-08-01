#!/bin/bash
# Fase 4, passo 2: o mesmo LOO aninhado da Fase 2, em modo --text-only, sobre a
# tabela slim da populacao completa. Roda o script da Fase 2 sem modificacao -
# e exatamente esse reuso que garante que o resultado seja comparavel com o da
# Fase 2 (mesmos folds, mesma PCA por fold, mesmo tuning interno).
#
# Passe a populacao ao submeter:
#   sbatch --export=POP=pt    rodar_loo.slurm
#   sbatch --export=POP=en    rodar_loo.slurm
#   sbatch --export=POP=union rodar_loo.slurm
#
# MEMORIA - este e o recurso que limita, nao a CPU. Por fold, o worker copia a
# fatia de treino do array de embeddings e a padroniza, ou seja ~2x:
#   pt      160 folds |  0,6GB por worker  -> cabe em qualquer no
#   en    2.479 folds | 12,6GB por worker  -> pede no de memoria grande
#   union 2.668 folds | 13,8GB por worker  -> idem
# Com --mem=128G e N_JOBS=8: ~8,6GB compartilhado + 8x13,8GB = ~119GB (en/union).
# Se o no tiver menos memoria, REDUZA N_JOBS na proporcao - estourar a memoria
# mata o job inteiro, enquanto menos workers so deixa mais lento (ha checkpoint
# a cada 20 folds, entao um job morto retoma de onde parou).
#SBATCH --job-name=phase4_loo
#SBATCH --partition=scientific
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err

source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

: "${POP:?defina a populacao, ex.: sbatch --export=POP=en rodar_loo.slurm}"
N_JOBS=${N_JOBS:-8}

# O script vive na pasta da Fase 2 de proposito: usar o MESMO codigo mantem a
# comparabilidade com os numeros ja publicados da Fase 2.
cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase2_classical_ml"

DATA="$HOME/steam-data/step06-output/phase4_text_only_fullpop"

python 01_phase2_nested_cv.py \
  --feature-table "$DATA/fullpop_${POP}.parquet" \
  --population    "$POP" \
  --text-only \
  --output        "$DATA/phase4_${POP}_fullpop_oof_predictions.parquet" \
  --n-jobs "$N_JOBS"
