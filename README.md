# Toxic Steam — pipeline, código e dados

Código e dados da dissertação *Toxic Steam: Towards Understanding and Mitigating
Toxicity in Digital Gaming Platforms* (Paula Teresa Mota Gibrim, PPGCC/UFV,
orientação de Julio Cesar Soares dos Reis).

O estudo caracteriza toxicidade em reviews da Steam de forma **cross-lingual
(inglês e português)** e testa se é possível **predizer usuários tóxicos** a
partir das reviews que eles escrevem.

---

## Onde ficam as coisas

| | Caminho |
|---|---|
| Código (este repositório) | `~/Documents/dev/msc-toxic-steam-complete/` |
| Dados, local | `~/Documents/dev/steam-data/` |
| Dados, cluster (ClusterGPU/UFV) | `/home/es90366/steam-data/` |

As duas árvores de dados têm a mesma estrutura interna, então os scripts só
mudam o prefixo. **Atenção:** localmente várias pastas de `steam-data` estão
descarregadas no iCloud (ícone de nuvem no Finder) — abrir um arquivo dispara
download e pode demorar. Antes da defesa, vale forçar o download das pastas que
você for precisar mostrar.

## Mapa: código → saída

| Pasta do código | Pasta de saída | O que faz |
|---|---|---|
| `step01_cleaning_and_language_detection/` | `step01-output/` | limpeza, deduplicação, detecção de idioma |
| `step02_run_detoxify/` | `step02-output/` | pontuação Detoxify, rótulo de toxicidade |
| `step03_bertopic/` | `step03-output/` | modelagem de tópicos (análise semântica) |
| `step04_sentiment_analysis/` | `step04-output/` | sentimento via BERT multilíngue |
| `step05_tfidf_analysis/` | `step05-output*/` | análise lexical (TF-IDF) |
| `step06_user_toxicity/` | `step06-output/` | predição de usuários tóxicos (6 fases) |
| `step07_toxicity_replication/` | `step07-output/` | replicação no nível da review |
| `annotation_agreement/` | `annotation-output/` | concordância entre anotadores |
| `review_examples/` | `examples/` | extração de exemplos de reviews |

---

## Entrada

### `raw/`  — dados coletados, sem tratamento

A coleta bruta da Steam, como veio do scraping.

- `games/todos_jogos.json` — catálogo de títulos
- `reviews/comentarios{1..25}_limpos_toxicidade.parquet` — 25 arquivos de reviews.
  O sufixo `_toxicidade` é enganoso: a única pontuação presente aqui é a do
  **Perspective API**, que veio junto da coleta. Detoxify ainda não existe nesta
  etapa.
- `users/todos_usuarios{1..10}_completo.parquet` — perfis de usuário
- `annotations/{en,pt}/bins_{en,pt}_{1,2}.xlsx` — planilhas de anotação manual,
  usadas no cálculo de concordância

---

## O pipeline

### `step01-output/` — limpeza e detecção de idioma

Transforma a coleta bruta no corpus de trabalho.

- `reviews_deduped.parquet/` — 200 buckets, passo intermediário da deduplicação
- `reviews_by_lang/reviews_cleaned.parquet/` — **o corpus limpo** (200 partições,
  11 colunas), com `review_lang` e `detection_confidence` de `langdetect`
- `users/all_users.parquet` — perfis limpos (20 colunas)
- `games/games.parquet` — catálogo limpo (26 colunas)
- `ascii_art_reviews.csv` — reviews de ASCII art isoladas na limpeza
- `*_report.json` — auditoria de cada corte

Números (em `dedup_report.json`, `users_report.json`, `games_report.json`):

| | Carregado | Final |
|---|---|---|
| Reviews | 97.530.533 | **93.015.854** |
| Usuários | 21.513.415 | **21.464.517** |
| Jogos | 169.141 | **167.246** |

Descartes nas reviews: 11.450 sem campo crítico, 4.503.229 duplicatas.

`und_breakdown_report.json` explica os 31.510.523 "undetermined" (33,88% do
corpus): 99,96% são `gated_model_never_called`, isto é, texto curto demais para
o detector ser chamado — **não** é falha de detecção.

### `step02-output/` — Detoxify e rótulo de toxicidade

Onde a coluna `detoxify_score` é criada. Só existem duas línguas aqui, e é por
um motivo de método: **o Detoxify só foi rodado em inglês e português**, então a
regra de união que define o rótulo não é computável nos outros 54 idiomas.

- `review_lang=en/part.{0..199}.parquet` — 200 partições
- `review_lang=pt/part.{0..199}.parquet` — 200 partições
- `toxicity_report.json`, `validity_report.json`, `score_correlation_report.json`

Esquema: o do step01 + `detoxify_score`. Ambas as colunas de pontuação têm 0% de
nulos.

Rótulo adotado: `Detoxify ≥ 0.9 ∪ Perspective ≥ 0.7`.

| | Subconjunto de trabalho | Tóxicas | % |
|---|---|---|---|
| EN | 25.341.249 | **480.729** | 1,90% |
| PT | 2.087.904 | **33.685** | 1,61% |

O subconjunto de trabalho é menor que a contagem de idioma do step01
(EN: 32.130.304) porque exige concordância entre o idioma detectado e o
declarado ao Perspective — 78,87% em inglês, 79,46% em português.

**Esta é a pasta que os steps 03 a 07 consomem.**

### `step03-output/` — análise semântica (BERTopic)

Uma árvore por idioma, `en/` e `pt/`:

- `cleaned/` — 200 partições de texto pré-processado para tópicos
- `embeddings/` — `embeddings_raw.npy`, `embeddings_pca.npy`, `pca_model.joblib`,
  `toxic_index.parquet`
- `optuna/` — `study.db` e `best_params.json` da busca de hiperparâmetros
- `models/final_model` — modelo BERTopic treinado
- `mlruns/` — rastreamento MLflow
- `stability/stability_report.json` — estabilidade dos tópicos entre execuções
- `results/` — `topic_info.csv`, `topic_labels.csv`, `classified_toxic.parquet`,
  `llm_prompts/`, `batches/`

A rotulação semântica dos tópicos foi assistida por LLM (Claude via API
Anthropic), com título, descrição e categoria de uma taxonomia fechada. Clusters
outlier (tópico −1) ficaram fora dessa inferência.

Resultado: 23 tópicos em inglês (4,03% outliers), 15 em português (4,23%).

### `step04-output/` — análise de sentimento

Acrescenta `sentiment_score` (escala contínua de 1 a 5) ao esquema do step02,
via `nlptown/bert-base-multilingual-uncased-sentiment`.

- `review_lang=en/`, `review_lang=pt/` — 200 partições cada

É a coluna que sustenta a validação do rótulo de usuário tóxico no step06.

### `step05-output*/` — análise lexical (TF-IDF)

Quatro pastas, porque foram três recortes diferentes:

- `step05-output-en/`, `step05-output-pt/` — o recorte principal:
  `tfidf_lexicon_{en,pt}.csv`, `reviews_cleaned_labeled_{en,pt}.parquet`,
  `tfidf_report_{en,pt}.json`
- `step05-output-venn/` — o mesmo cálculo por região do diagrama de Venn
  (Perspective only / Detoxify only / ambos)
- `step05-output/` — as tabelas consolidadas que vão para o texto:
  `top_terms.csv` e `top_terms_diferenca.csv`

### `step06-output/` — predição de usuários tóxicos

O capítulo mais extenso. Seis fases:

- **`phase1_feature_engineering/`** — constrói os rótulos e as features.
  `pipeline/toxic_user_labels.parquet` é o rótulo de usuário tóxico (teste
  binomial unicaudal, α = 0,01, só usuários com ≥90% das reviews no idioma);
  `phase1_feature_table.parquet`, `user_sentiment_table.parquet`,
  `user_profile_metadata.parquet`, `user_text_embeddings.parquet`.
  `diagnostics/` traz os relatórios de validação por sentimento, inclusive na
  condição *leave-toxic-out*.
- **`phase2_classical_ml/`** — classificadores clássicos sobre features
  agregadas; OOF por idioma, com e sem LTO.
- **`phase3_distilbert/`** — DistilBERT com fine-tuning. Predições OOF por fold
  (`*.fold{0..4}.parquet`), mescladas por `merge_folds.py`. Também
  `cross_lingual_transfer.json`.
- **`phase4_text_only_fullpop/`** — encoder congelado + cabeça clássica sobre a
  população completa; é a abordagem com melhor resultado, e a base do
  experimento de transferência cross-lingual.
- **`phase5_significance/significance_report.json`** — testes de significância.
- **`phase6_llm_classifier/`** — LLMs comerciais como classificadores de usuário
  (Claude, GPT, Gemini), em condição circular apenas.

Rótulo resultante: **2.479 usuários em inglês (0,0485%)** e **160 em português
(0,0326%)**.

Duas condições de avaliação atravessam as fases e **não devem ser confundidas**:

- **circular** — o modelo vê as mesmas reviews que geraram o rótulo; infla tudo
- **LTO (*leave-toxic-out*)** — as reviews sinalizadas são removidas antes do
  pooling; é a condição honesta

A distância entre as duas é o resultado central do capítulo.

### `step07-output/` — replicação no nível da review

Repete a predição com a review (não o usuário) como unidade, para comparação.

- `review_dataset_{en,pt}.parquet` — datasets achatados com o rótulo
- `review_embeddings_{en,pt}.npy` — embeddings DistilBERT (o arquivo em inglês
  tem ~39 GB)
- `review_classical_{en,pt}_oof.parquet` — predições out-of-fold

Predizer *a review* é muito mais fácil que predizer *o usuário*: AUC-PR de 0,256
(regressão logística) e 0,281 (HistGB) em português, contra ordens de grandeza
menos no nível de usuário.

---

## Pastas auxiliares

### `annotation-output/`
`agreement_table.json` — concordância entre os três anotadores humanos. Reporta
kappa de Randolph (medida principal, já que os anotadores não trabalharam com
cota e a prevalência de cada bin vem da estratificação do estudo) e kappa de
Fleiss para comparabilidade.

### `reports/`
`step06_relatorio_fases_1_2.md` — relatório de metodologia das duas primeiras
fases do step06.

### `examples/`
~80 CSVs de exemplos de reviews, extraídos por `review_examples/`. O nome
codifica o recorte: `{toxic,nontoxic}_{en,pt}_{termo}_seed{N}_n{N}_light.csv`,
além dos recortes por região do Venn (`perspective_only_*`, `detoxify_only_*`).
São a origem dos exemplos citados no texto e nos slides.

### `visualizations/`
Figuras finais, em PNG e PDF:

- `venn/` — diagramas de Venn da sobreposição entre as ferramentas (EN, PT, total)
- `languages/` — distribuição de idiomas (`top_languages`, `language_counts.csv`)
- `sentiment_by_toxicity_set/` — sentimento por região do Venn (violino, boxplot, CDF)

---

## Convenções

- Seed global **42** em tudo que amostra ou particiona.
- Reviews particionadas em **200 partes** (`part.0` … `part.199`) em todas as
  etapas; a partição é estável entre os steps.
- Todo passo que filtra escreve um `*_report.json` ou `*.summary.json` ao lado da
  saída, com as contagens antes e depois. **Todo número da dissertação é
  rastreável a um desses arquivos.**
- Rótulos de figura sempre em CAIXA ALTA (eixos, ticks, legenda, categorias),
  aplicada no desenho e não nos dados.
