# Relatório de Metodologia — Modelagem de Tópicos (BERTopic) sobre Reviews Tóxicas

**Data:** 20/07/2026 (atualizado com Stage 8 — rotulagem via LLM)
**Escopo:** Corpus de reviews tóxicas do Steam, em inglês (EN) e português (PT)

---

## 1. Objetivo

Descobrir, de forma não supervisionada, quais narrativas/temas estruturam o discurso tóxico nos dois subcorpora de reviews (previamente rotuladas como tóxicas em etapa anterior do trabalho), usando BERTopic como técnica de modelagem de tópicos.

---

## 2. Arquitetura do Pipeline

O pipeline foi estruturado em 7 estágios independentes, cada um com um script de entrada em `run/` e a lógica correspondente em `src/`:

| Estágio | Script | Função |
|---|---|---|
| 1 | `01_clean.py` | Limpeza de texto em lote (minúsculas, remoção de URLs, normalização Unicode, remoção de caracteres não alfabéticos) |
| 2 | `02_embed.py` | Geração de embeddings + redução via PCA |
| 3 | `03_search.py` | Busca de hiperparâmetros (Optuna) para UMAP + HDBSCAN |
| 4 | `04_stability.py` | Análise empírica de estabilidade por tamanho de amostra |
| 5 | `05_train.py` | Treino do modelo BERTopic final |
| 6 | `06_infer.py` | Classificação de toda a base tóxica via `.transform()` |
| 7 | `07_export.py` | Consolidação dos resultados e exportação das tabelas finais |
| 8 | `08_label_topics.py` | Rotulagem semântica dos tópicos assistida por LLM (Claude) |

Cada idioma tem seu próprio arquivo de configuração (`config_en.yaml` / `config_pt.yaml`), permitindo hiperparâmetros, caminhos e stop words distintos por idioma, mas usando o mesmo código-fonte.

---

## 3. Decisões Metodológicas e Justificativas

### 3.1 Modelo de embedding
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — escolhido por produzir um espaço vetorial multilíngue compartilhado, permitindo aplicar a mesma arquitetura a EN e PT sem trocar de modelo.

### 3.2 Redução de dimensionalidade em duas etapas
1. **PCA → 50 dimensões**: redução grosseira, preservando a maior parte da variância global antes da etapa não-linear.
2. **UMAP** sobre a saída do PCA, com métrica de cosseno, para a redução final antes da clusterização.

Motivo: clusterização direta sobre embeddings de alta dimensão é computacionalmente cara e HDBSCAN degrada em espaços de alta dimensão (maldição da dimensionalidade). O PCA como pré-filtro é uma prática padrão recomendada para acelerar o UMAP sem perda relevante de estrutura.

### 3.3 Clusterização com HDBSCAN
Escolhido por (a) não forçar todo documento a pertencer a algum cluster (permite outliers/ruído, comum em corpora de reviews curtas), e (b) não exigir o número de clusters a priori.

### 3.4 Busca de hiperparâmetros com Optuna
- Espaço de busca: `n_neighbors`, `n_components`, `min_dist` (UMAP) e `min_cluster_size`, `min_samples` (HDBSCAN).
- 60 trials por idioma.
- Amostra de busca: **30% do corpus tóxico de cada idioma** (não uma contagem fixa — ver §4.1).
- Função objetivo: `outlier_rate - coherence_weight × c_npmi`, com uma penalidade adicional quando o número de tópicos fica abaixo de um piso mínimo (`min_topics = 10`).
- Coerência semântica medida via `c_npmi` (gensim), sobre as 10 palavras mais representativas de cada tópico.

### 3.5 Vetorização c-TF-IDF
BERTopic extrai o vocabulário representativo de cada tópico concatenando os documentos do cluster em um "super-documento" e computando TF-IDF entre clusters (não entre documentos individuais). A vetorização usa `CountVectorizer` do scikit-learn com stop words do NLTK por idioma, ampliadas com termos estruturais do domínio (`game/steam/play/playing/games` em EN; `jogo/jogar/jogando/jogos/steam` em PT).

### 3.6 Reprodutibilidade
Seed global fixada em 42 em todas as etapas estocásticas (embedding, UMAP, HDBSCAN, Optuna). `UMAP(n_jobs=1)` e `HDBSCAN(core_dist_n_jobs=1)` são obrigatórios para reprodutibilidade bit-a-bit (paralelismo introduziria não-determinismo).

### 3.7 Rotulagem semântica assistida por LLM (Stage 8)

Os tópicos descobertos pelo BERTopic são identificados apenas por um conjunto de 10 palavras-chave (c-TF-IDF) — sem um rótulo interpretativo nem uma descrição legível. O Stage 8 usa a API da Anthropic (Claude) para gerar, por tópico: um rótulo curto, uma descrição de uma frase, uma categoria de uma taxonomia fixa, e uma sinalização de suspeita de *copypasta* (texto repetido/templado entre reviews).

**Fonte dos exemplos por tópico:** em vez de usar `BERTopic.get_representative_docs()`, os exemplos são amostrados diretamente do `classified_toxic.parquet` (15 documentos aleatórios por tópico, seed fixa). Motivo: a serialização `safetensors` do BERTopic não preserva `representative_docs_` — essa coluna já estava sempre vazia nos CSVs exportados pelo Stage 7, mesmo antes da rotulagem via LLM. Amostrar diretamente do resultado classificado também tem a vantagem de refletir a base completa, não só a amostra de treino.

**Taxonomia fixa de categorias:** `generic_insult`, `violence_threat`, `sexual_content`, `identity_hate`, `nationality_xenophobia`, `product_complaint`, `humor_sarcasm`, `copypasta_repetition`, `other`. Decisão consciente de usar uma lista fechada (em vez de deixar a LLM inventar categorias livremente), para permitir agregação e contagem por categoria depois. Combinado explicitamente que a taxonomia seria revisada caso uma proporção alta de tópicos caísse em `other` — o que não se confirmou (ver §7).

**Enquadramento do prompt:** o system prompt deixa explícito que se trata de pesquisa acadêmica de moderação de conteúdo, que os textos são dados já coletados (não conteúdo a ser gerado), e instrui o modelo a não recusar nem suavizar a tarefa por questões de segurança — mas também a não reproduzir slurs/palavrões literalmente no campo de rótulo (só de forma clínica na descrição, quando necessário).

**Escolha do modelo — validação em duas etapas:**
1. Primeiro teste com `claude-haiku-4-5` (modelo mais barato, ~$1/$5 por milhão de tokens) para validar o pipeline (parsing de JSON, taxonomia, ausência de erros) sem gastar com o modelo mais caro.
2. Após validação sem erros nos dois idiomas, execução final com `claude-fable-5` (modelo mais capaz da Anthropic, indicado para tarefas de raciocínio mais exigentes) para o resultado que compõe este relatório.

**Nota técnica:** o `max_tokens` por chamada foi elevado de um valor inicial de 500 para 4096. Modelos com raciocínio sempre ativado (caso do Fable 5, que não pode ser desligado) consomem parte do `max_tokens` com "pensamento" interno antes de gerar a resposta; com um teto baixo, esse raciocínio podia consumir todo o orçamento e não deixar espaço para o JSON de resposta. Elevar o teto não aumenta o custo real (que depende só do que o modelo efetivamente gera), apenas evita truncamento.

---

## 4. Problemas Identificados e Correções Aplicadas

Ao longo da execução, seis problemas metodológicos/técnicos distintos foram identificados e corrigidos. Documentados aqui porque cada um teve impacto direto nos resultados finais.

### 4.1 Espaço de busca do HDBSCAN desproporcional entre idiomas
**Sintoma:** nenhum trial de PT conseguia ultrapassar 6 tópicos (o piso mínimo era 10), mesmo depois de 60 trials.
**Causa:** `min_cluster_size` era buscado em um intervalo absoluto fixo (`[100, 3000]`), calibrado implicitamente para o corpus EN (480.729 documentos tóxicos). Para o corpus PT (33.685 — **14× menor**), esse mesmo intervalo absoluto representa uma fração proporcionalmente muito maior do corpus, tornando 10+ tópicos estruturalmente inatingíveis.
**Correção:** o espaço de busca passou a ser expresso como **fração do corpus total** (`[0.000208, 0.006241]`), resolvida para uma contagem absoluta em tempo de execução, por idioma. As frações foram calibradas para reproduzir exatamente o intervalo absoluto antigo quando aplicadas ao corpus EN — ou seja, o comportamento de busca do EN não mudou; só o do PT passou a ser proporcionalmente equivalente.

### 4.2 `min_cluster_size` reaproveitado sem reescala entre tamanhos de amostra diferentes
**Sintoma:** mesmo após o fix acima, a análise de estabilidade (Stage 4) e o treino final (Stage 5) mostravam resultados instáveis e sem relação clara com o tamanho da amostra.
**Causa:** o `min_cluster_size` encontrado pelo Optuna é calibrado sobre a amostra de busca (30% do corpus). Reaplicá-lo, sem ajuste, a um tamanho de treino diferente (qualquer degrau da escada de estabilidade, ou o corpus completo no treino final) distorce silenciosamente o que essa contagem absoluta representa proporcionalmente.
**Correção:** `best_params.json` passou a registrar `_search_sample_size` (o tamanho exato da amostra em que o valor foi calibrado). Uma função utilitária (`scale_min_cluster_size`) reaplica a mesma proporção `min_cluster_size / search_sample_size` a qualquer novo tamanho de treino.

### 4.3 `stability_score` sempre nulo
**Sintoma:** a análise de estabilidade nunca conseguia confirmar convergência — sempre recomendava treinar com 100% do corpus, mesmo quando isso não era necessário.
**Causa:** o vocabulário do `CountVectorizer` era reconstruído do zero em cada tamanho de amostra testado, produzindo matrizes c-TF-IDF com número de colunas diferente a cada tamanho — impossibilitando a comparação de similaridade de cosseno entre elas.
**Correção:** um vocabulário único é ajustado uma vez sobre o corpus completo do idioma e reutilizado em todos os tamanhos testados na mesma execução, garantindo que todas as matrizes comparadas compartilhem o mesmo espaço de colunas.

### 4.4 Seleção de tamanho recomendado aceitando "estabilidade" degenerada
**Sintoma:** mesmo após o fix acima, o EN chegou a recomendar 96.146 documentos (20% do corpus) com um `stability_score` de 0,9866 — aparentemente ótimo — mas o modelo treinado nesse tamanho colapsou em apenas 2 tópicos.
**Causa:** a lógica de seleção aceitava o **primeiro** tamanho cuja similaridade superasse o limiar (0,85), sem verificar se o número de tópicos correspondente atingia o piso mínimo. Dois tamanhos consecutivos que colapsam para a mesma estrutura degenerada (ex: tudo em 2 blobs genéricos) produzem uma similaridade de cosseno artificialmente alta entre si — parecem "estáveis" por serem igualmente vazios de estrutura.
**Correção:** candidatos com número de tópicos abaixo do piso mínimo (`min_topics`) são descartados da seleção, mesmo que sua similaridade individual ultrapasse o limiar.

### 4.5 Reclassificação da base pulada silenciosamente (Stage 6)
**Sintoma:** após retreinar o modelo, os resultados exportados continuavam refletendo o modelo antigo.
**Causa:** o Stage 6 (`06_infer.py`) tem retomada (`--resume`) habilitada por padrão, e encontrava arquivos de batch já existentes (de execuções anteriores, com modelos antigos) com o mesmo nome — pulando a reclassificação inteira sem aviso visível (log mostrava `total classified: 0`).
**Correção:** nenhuma mudança de código foi necessária; a flag `--no-resume` deve ser usada explicitamente sempre que o modelo subjacente mudar entre execuções.

### 4.6 Falha de generalização do UMAP fora da amostra (EN)
**Sintoma:** ao classificar a base completa de EN (480.729 documentos) usando um modelo treinado em apenas 30% dela (144.219 documentos — a recomendação legítima da análise de estabilidade, já corrigida), **todos os 336.510 documentos não vistos no treino** foram classificados em apenas 2 dos 20 tópicos existentes; nenhum dos outros 18 tópicos recebeu um único documento novo.
**Causa:** verificado diretamente no código-fonte do BERTopic (`_bertopic.py`) que `.transform()`, com um `hdbscan_model` real (não um `BaseCluster` vazio), usa `umap_model.transform(embeddings)` seguido de `hdbscan.approximate_predict()` — ou seja, depende da capacidade do UMAP de extrapolar posições para pontos nunca vistos durante o ajuste. Essa extrapolação se mostrou pouco confiável quando o volume de dados a extrapolar (336.510) é uma ordem de grandeza maior que o volume usado no ajuste (144.219).
**Correção:** o modelo final de EN foi treinado sobre a **totalidade do corpus tóxico** (480.729 documentos), eliminando a necessidade de qualquer extrapolação do UMAP no Stage 6 — todo documento classificado já havia sido visto durante o treino.
**Nota metodológica:** essa decisão **não contradiz** a metodologia da análise de estabilidade — ela expõe uma limitação que a análise de estabilidade, por construção, nunca testava (ver §5).

---

## 5. Nota sobre a Assimetria Metodológica entre EN e PT

A análise de estabilidade (Stage 4) responde a uma pergunta: *a estrutura de tópicos converge conforme a amostra de treino cresce?* Ela **não** responde a uma segunda pergunta, igualmente necessária: *um modelo treinado numa amostra menor generaliza, via `.transform()`, para os documentos que ficaram de fora?*

- Para **PT**, ambas as perguntas tiveram resposta afirmativa: a estrutura convergiu a 40% do corpus (similaridade de cosseno 0,9079 entre os tamanhos de 30% e 40%), e a classificação da base completa confirmou que o modelo treinado nessa amostra generaliza bem (todos os 15 tópicos populados corretamente).
- Para **EN**, a primeira pergunta teve resposta afirmativa (convergência a 30%), mas a segunda, não — a extrapolação do UMAP falhou pela escala do que precisava ser extrapolado (336 mil documentos, mais que o dobro do corpus inteiro de PT). Por isso, o modelo final de EN foi treinado com o corpus completo, eliminando a necessidade de extrapolar.

A diferença de tratamento entre os dois idiomas é, portanto, **uma consequência validada empiricamente de suas diferenças de escala**, não uma inconsistência metodológica.

---

## 6. Resultados Finais

### 6.1 Hiperparâmetros selecionados (Optuna)

| Parâmetro | EN | PT |
|---|---|---|
| `n_neighbors` | 25 | 40 |
| `n_components` | 10 | 13 |
| `min_dist` | 0,0148 | 0,0166 |
| `min_cluster_size` (calibrado na busca) | 376 (sobre amostra de 144.219) | 110 (sobre amostra de 10.106) |
| `min_cluster_size` (reescalado p/ treino final) | 1.253 (sobre 480.729) | ~147 (sobre 13.474) |
| `min_samples` | 9 | 20 |

### 6.2 Treino final

| | EN | PT |
|---|---|---|
| Tamanho de treino | 480.729 (100% — extrapolação do UMAP não confiável na amostra recomendada) | 13.474 (40% — convergência validada empiricamente) |
| Tópicos encontrados | 23 | 15 |
| Outlier rate (no ajuste) | 25,0% | 29,5% |

### 6.3 Classificação da base completa (Stage 6/7)

| | EN (480.729 docs) | PT (33.685 docs) |
|---|---|---|
| Tópicos com documentos | 23 de 23 | 15 de 15 |
| **Outlier rate** | **4,03%** | **4,23%** |
| Menor tópico | 5.485 docs | 642 docs |

> **Nota:** a taxa de outliers na classificação da base completa (Stage 6/7, via `.transform()`) é sistematicamente menor que a taxa observada no ajuste do modelo (Stage 5, via `fit_transform()`). Isso é esperado: `hdbscan.approximate_predict()` (usado no `.transform()`) tende a atribuir documentos a clusters existentes de forma mais permissiva que a clusterização original por densidade. O número mais relevante para a seção de resultados é o da classificação da base completa, pois é ele que reflete a cobertura real do corpus tóxico inteiro.

### 6.4 Principais tópicos por volume

**EN (top 8 de 23):**

| Tópico | Documentos | % | Palavras-chave |
|---|---|---|---|
| 0 | 168.545 | 35,1% | like, sucks, ass, stupid, get, even, good, time, fun, buy |
| 1 | 41.936 | 8,7% | kill, shoot, people, death, die, killing, love, fun, gun |
| 2 | 38.578 | 8,0% | garbage, trash, crap, buy, money, stupid, piece, ass, sucks |
| -1 (outlier) | 19.379 | 4,0% | hate, ass, sex, terrorist, like, kill, get, russia, state |
| 7 | 15.976 | 3,3% | suck, sh, sucks, stupid, good, ass, dumb, ck, love, like |
| 14 | 14.394 | 3,0% | fun, silly, stupid, funny, dumb, goofy, suck, stupidly |
| 16 | 13.874 | 2,9% | ass, pain, butt, pizza, kicked, fat, poop, suck, eat, hurts |
| 3 | 13.697 | 2,8% | sex, penis, sexy, porn, cant, want, anal, add, please |

**PT (top 8 de 15):**

| Tópico | Documentos | % | Palavras-chave |
|---|---|---|---|
| 0 | 10.346 | 30,7% | nao, lixo, pra, so, caralho, bom, voce, porcaria, valve, vai |
| 2 | 2.491 | 7,4% | lixo, nao, porcaria, dinheiro, compra, pra, ruim, comprar, pessimo |
| 6 | 2.396 | 7,1% | caguei, pau, chorei, camisinha, caralho, confortavel, senhores |
| 4 | 2.295 | 6,8% | caralho, carlinhos, pra, pau, peixe, chupa, ai, porrada, vai |
| 5 | 2.251 | 6,7% | bom, recomendo, caralho, pra, adoro, gosto, gostei, porrada, legal |
| 1 | 2.081 | 6,2% | matar, matei, pessoas, mata, bom, morre, morrer, pra, morri |
| 10 | 1.695 | 5,0% | matar, vc, bom, morrer, mata, morre, pra, voce, vai, amigos |
| 3 | 1.662 | 4,9% | sexo, faco, gay, bonito, mulheres, porque, penis, fazer, amigas |

Tabelas completas (todos os tópicos, todas as colunas) disponíveis em:
- `steam-data/step03-output/en/results/topic_info_real_counts.csv`
- `steam-data/step03-output/pt/results/topic_info_real_counts.csv`

---

## 7. Rotulagem Semântica dos Tópicos (Stage 8 — resultado)

Rotulagem executada com `claude-fable-5` (após validação prévia sem erros com `claude-haiku-4-5`), sobre os tópicos finais de ambos os idiomas. Tabelas completas em:
- `steam-data/step03-output/en/results/topic_labels.csv`
- `steam-data/step03-output/pt/results/topic_labels.csv`

### 7.1 Distribuição por categoria

| Categoria | EN (de 24 tópicos) | PT (de 16 tópicos) |
|---|---|---|
| `product_complaint` | 6 | 1 |
| `humor_sarcasm` | 5 | 6 |
| `sexual_content` | 4 | 1 |
| `other` | 3 | 4 |
| `generic_insult` | 2 | 3 |
| `nationality_xenophobia` | 2 | 1 |
| `identity_hate` | 2 | 0 |
| `violence_threat` | 0 | 0 |
| `copypasta_repetition` | 0 | 0 |

> O tópico de nazistas/comunistas em PT (tópico 12) foi classificado como `humor_sarcasm`, não `violence_threat` — o LLM interpretou o registro como entusiasmo hiperbólico/jocoso sobre a mecânica de jogo, não como ameaça real. Nenhum tópico caiu em `violence_threat` em nenhum dos dois idiomas.

Nenhum idioma concentrou uma proporção alta de tópicos em `other` (3/24 no EN, 4/16 no PT), então a taxonomia fixa definida em §3.7 não precisou ser revisada — decisão tomada previamente de só reabrir essa discussão caso isso acontecesse.

### 7.2 Achado qualitativo: violência ficcional/de jogo como possível falso positivo

Um padrão semântico se repete de forma consistente **nos dois idiomas**, concentrado sobretudo nos tópicos que caíram em `other`: o LLM descreve reviews sobre matar zumbis, nazistas ou demônios *dentro do jogo* como prováveis falsos positivos do classificador de toxicidade original (Perspective/Detoxify), por reconhecer palavras de violência (kill, matar, shoot) sem que exista hostilidade real direcionada a alguém.

- **EN** — tópico 9 ("In-Game Nazi Killing Enthusiasm"): *"...using graphic violent language that likely triggered the toxicity classifier despite referring to fictional gameplay rather than real-world threats."*
- **EN** — tópico 15 ("In-Game Demon Killing Descriptions"): mesmo padrão, associado a Doom/Dark Souls/God of War.
- **PT** — tópicos 1, 7, 10 (matar/morrer, demônios): mesmo padrão, com o LLM descrevendo o registro como "hiperbólico ou humorístico de jogador" em vez de ameaça real.

Esse é um achado relevante para a discussão da dissertação sobre a distinção entre toxicidade detectada automaticamente e hostilidade genuína — o próprio processo de rotulagem semântica está sinalizando, de forma sistemática e nos dois idiomas, onde o classificador de toxicidade de origem provavelmente reagiu ao vocabulário de violência sem capturar o contexto ficcional/lúdico. Uma extensão natural da taxonomia (não implementada ainda) seria uma categoria dedicada, como `fictional_violence`, separada de `violence_threat` (reservada a ameaças/hostilidade real). Hoje nenhum tópico, em nenhum dos dois idiomas, caiu em `violence_threat` — mesmo o tópico de nazistas/comunistas em PT foi classificado como `humor_sarcasm`, reforçando que o próprio LLM já está absorvendo esse padrão de "violência de jogo ≠ ameaça real" em categorias adjacentes, sem uma categoria própria para isolá-lo explicitamente.

### 7.3 Copypasta confirmado com evidência textual

A suspeita de copypasta levantada na leitura das palavras-chave (relatório anterior, PT tópico 6) foi substituída por confirmação textual direta em outros tópicos, com trechos específicos apontados pelo LLM:

| Idioma | Tópico | Evidência |
|---|---|---|
| EN | 1 (kill/shoot) | Frase "made me kill my whole family" aparece verbatim em múltiplos reviews — piada-template repetida |
| EN | 6 (balls/sussy) | "sussy balls" — meme derivado de Among Us, templado em vários reviews |
| PT | 3 (sexo/gay) | Palavra "sexo" repetida ~100 vezes num único review; template repetido sobre "a dalva" em outro |
| PT | 8 (spam) | "charada eu te odeio" repetido ~17 vezes; "eu não sou louco" (meme do Coringa) repetido ~15 vezes |
| PT | 11 (tiro/porrada) | "tiro, porrada e bomba" — bordão de programa de TV brasileiro (Ratinho) — repetido quase verbatim em pelo menos 4 reviews |

Recomenda-se inspeção manual desses casos via `review_url` antes de citar como copypasta confirmado no texto final da dissertação — a sinalização do LLM é um indício forte, não uma verificação manual.

---

## 8. Próximos Passos Sugeridos

1. ~~Rotulagem semântica dos tópicos~~ — **concluído** (Stage 8, §7).
2. Inspeção manual dos casos de copypasta listados em §7.3, confirmando via os `review_url` originais.
3. Avaliar a criação da categoria `fictional_violence` na taxonomia do Stage 8 (§7.2) e re-rotular os tópicos afetados, caso o achado seja incorporado à dissertação.
4. Cruzamento dos tópicos com metadados de jogos (gênero, `game_id`) para checar se algum tópico se concentra em categorias específicas de jogos.
5. Análise de sentimento por tópico (ver Capítulo de Análise Semântica da dissertação, seção pendente sobre `sentiment_score`).
