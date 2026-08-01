#!/bin/bash
# Fase 4, passo 1: embeddings da populacao COMPLETA (sem o filtro de perfil).
# --array=0-2 -> uma tarefa por populacao (0=pt, 1=en, 2=union).
#
# Cada tarefa reaproveita os vetores que a Fase 1 ja calculou (mesmo modelo,
# mesma limpeza) via --existing-embeddings, entao so codifica quem ficou de
# fora do join de perfil:
#   pt     490.470 elegiveis - 179.042 reaproveitados =   311.428 a codificar
#   en   5.111.196 elegiveis - 2.191.286 reaproveitados = 2.919.910 a codificar
#   union 5.615.629 elegiveis - 2.375.760 reaproveitados = 3.239.869 a codificar
#
# Memoria: o acumulador dense e n_usuarios x 384 x 4 bytes (pt 0,8GB / en 7,9GB /
# union 8,6GB) + a tabela reaproveitada (~3,7GB) + o modelo. 32G cobre o pior caso.
#SBATCH --job-name=phase4_embed
#SBATCH --partition=scientific
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=%x_%a_%A.out
#SBATCH --error=%x_%a_%A.err

module load CUDA/12.6.0
source "$HOME/msc-toxic-steam-complete/.venv/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

POPS=(pt en union)
POP=${POPS[$SLURM_ARRAY_TASK_ID]}

cd "$HOME/msc-toxic-steam-complete/step06_user_toxicity/phase4_text_only_fullpop"

P1="$HOME/steam-data/step06-output/phase1_feature_engineering/pipeline"
OUT="$HOME/steam-data/step06-output/phase4_text_only_fullpop"

# SEM --existing-embeddings de proposito: reaproveitar os vetores da Fase 1
# exigiria subir user_text_embeddings.parquet (8,4GB) para poupar ~40% de um
# job que na A100 leva ~1h. Nao compensa a transferencia. Se o arquivo ja
# estiver no cluster por outro motivo, acrescente:
#   --existing-embeddings "$P1/user_text_embeddings.parquet"
python 01_build_fullpop_embeddings.py \
  --step02-dir  "$HOME/steam-data/step02-output" \
  --labels      "$P1/toxic_user_labels.parquet" \
  --population  "$POP" \
  --output      "$OUT/fullpop_${POP}.parquet" \
  --batch-size 256 \
  --checkpoint-every 50

# Ha checkpoint a cada 50 arquivos (validado: retomar produz resultado
# identico bit a bit). Se o job morrer por preempcao/QOS, basta reenviar o
# mesmo sbatch - ele retoma de onde parou.
#
# ATENCAO: requer step02-output no cluster (5,9GB, e a fonte dos textos).
# Se nao estiver la, rode este passo localmente - o script detecta MPS no Mac
# automaticamente - e suba so o fullpop_<pop>.parquet resultante
# (pt ~0,8GB / en ~7,9GB / union ~8,6GB).
