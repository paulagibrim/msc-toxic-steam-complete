# Relatório de Resultados — Modelagem de Tópicos (BERTopic) sobre Reviews Tóxicas

**Data:** 23/07/2026
**Escopo:** Corpus de reviews tóxicas do Steam, em inglês (EN) e português (PT)
**Nota:** este documento apresenta apenas valores. Para metodologia, decisões e justificativas, ver `METHODOLOGY_REPORT.md`.

---

## 1. Corpus

| | EN | PT |
|---|---|---|
| Total de reviews tóxicas | 480.729 | 33.685 |
| Amostra de busca de hiperparâmetros (30%) | 144.219 | 10.106 |
| Amostra de treino do modelo final | 480.729 (100%) | 13.474 (40%) |

---

## 2. Busca de Hiperparâmetros (Stage 3 — Optuna)

- `n_trials`: 60 (ambos os idiomas)
- Amostra de busca: 30% do corpus tóxico de cada idioma
- Função objetivo: `outlier_rate - coherence_weight × c_npmi`, `coherence_weight = 1.0`, `min_topics = 10`
- Métrica de coerência: `c_npmi` (gensim), top-10 palavras por tópico

| Parâmetro | EN | PT |
|---|---|---|
| `n_neighbors` (UMAP) | 25 | 40 |
| `n_components` (UMAP) | 10 | 13 |
| `min_dist` (UMAP) | 0,014834779559245344 | 0,016572299261668113 |
| `min_cluster_size` (HDBSCAN, calibrado na busca) | 376 | 110 |
| `min_samples` (HDBSCAN) | 9 | 20 |
| `_search_sample_size` | 144.219 | 10.106 |
| Espaço de busca `min_cluster_size` (fração) | [0,000208 ; 0,006241] | [0,000208 ; 0,006241] |
| Espaço de busca `min_samples` | [1, 20] | [1, 20] |
| Espaço de busca `n_neighbors` | [5, 50] | [5, 50] |
| Espaço de busca `n_components` | [5, 15] | [5, 15] |
| Espaço de busca `min_dist` | [0,0 ; 0,5] | [0,0 ; 0,5] |
| Tópicos no trial vencedor | 20 | 12 |
| `c_npmi` no trial vencedor (sobre amostra de busca) | 0,0021 | 0,0114 |

Fonte: `steam-data/step03-output/{en,pt}/optuna/best_params.json`.

---

## 3. Análise de Estabilidade (Stage 4)

Limiar de estabilidade (`stability_threshold`): 0,85 (ambos os idiomas).

### 3.1 EN — todos os tamanhos testados

| Fração | Tamanho (docs) | Tópicos | Outlier rate | Menor tópico | `stability_score` |
|---|---|---|---|---|---|
| 10% | 48.073 | 2 | 0,67% | 22.358 | — (nulo, sem par anterior) |
| 20% | 96.146 | 2 | 0,82% | 44.864 | 0,9866 |
| **30%** | **144.219** | **20** | **20,74%** | **448** | **0,9128** |
| 40% | 192.292 | 23 | 22,69% | 507 | 0,8637 |

Tamanho recomendado: **144.219 (30%)**, `stability_score` = 0,9128 — primeiro tamanho acima do limiar cujo número de tópicos também ultrapassa o piso mínimo. O par de 20% (0,9866) foi descartado apesar do escore mais alto, por corresponder a apenas 2 tópicos.

### 3.2 PT — todos os tamanhos testados

| Fração | Tamanho (docs) | Tópicos | Outlier rate | Menor tópico | `stability_score` |
|---|---|---|---|---|---|
| 10% | 3.368 | 17 | 32,27% | 37 | — (nulo, sem par anterior) |
| 20% | 6.737 | 6 | 5,46% | 80 | 0,5333 |
| 30% | 10.106 | 12 | 21,82% | 111 | 0,6346 |
| **40%** | **13.474** | **15** | **29,50%** | **154** | **0,9079** |

Tamanho recomendado: **13.474 (40%)**, `stability_score` = 0,9079.

**Nota:** o tamanho de 30% recomendado pela análise de estabilidade do EN não foi o tamanho usado no treino final — ver §4 (Treino Final) e `METHODOLOGY_REPORT.md` §4.6/§5 para o motivo.

Fonte: `steam-data/step03-output/{en,pt}/stability/stability_report.json`.

---

## 4. Treino do Modelo Final (Stage 5)

| | EN | PT |
|---|---|---|
| Tamanho de treino usado | 480.729 (100%) | 13.474 (40%) |
| `min_cluster_size` reescalado para o treino final | 1.253 | 147 |
| `min_samples` | 9 | 20 |
| Tópicos encontrados | 23 | 15 |
| Outlier rate no ajuste (`fit_transform`) | 25,00% (0,2496) | 29,50% (0,2950) |

Fonte: experimento `bertopic_final_training` no MLflow de cada idioma; espelhado em `results/logs/05_train.log`.

---

## 5. Classificação da Base Completa (Stage 6/7)

| | EN | PT |
|---|---|---|
| Documentos classificados | 480.729 | 33.685 |
| Tópicos com documentos | 23 de 23 | 15 de 15 |
| Outlier rate (`.transform()`, base completa) | 4,03% | 4,23% |
| Menor tópico (documentos) | 5.485 | 642 |
| Maior tópico (documentos) | 168.545 | 10.346 |

### 5.1 Tabela completa de tópicos — EN (24 linhas, incl. outlier)

| Tópico | Documentos | % | Palavras-chave (c-TF-IDF) |
|---|---|---|---|
| 0 | 168.545 | 35,07% | like, sucks, ass, stupid, get, even, good, time, fun, buy |
| 1 | 41.936 | 8,73% | kill, shoot, people, death, die, killing, love, fun, gun, get |
| 2 | 38.578 | 8,03% | garbage, trash, crap, buy, money, stupid, piece, ass, sucks, even |
| -1 (outlier) | 19.379 | 4,03% | hate, ass, sex, terrorist, like, kill, get, russia, state, click |
| 7 | 15.976 | 3,32% | suck, sh, sucks, stupid, good, ass, dumb, ck, love, like |
| 14 | 14.394 | 3,00% | fun, silly, stupid, funny, dumb, goofy, suck, stupidly, really, hilarious |
| 16 | 13.874 | 2,89% | ass, pain, butt, pizza, kicked, fat, poop, suck, eat, hurts |
| 3 | 13.697 | 2,85% | sex, penis, sexy, porn, cant, want, anal, add, please, would |
| 5 | 13.422 | 2,79% | sex, penis, women, like, good, porn, would, sexy, big, girls |
| 11 | 13.321 | 2,77% | story, characters, character, puzzles, stupid, ending, damn, like, ass, good |
| 6 | 12.111 | 2,52% | balls, mucky, sussy, hurtmy, ball, among, love, us, hitler, remnant |
| 15 | 12.010 | 2,50% | demons, hell, demon, kill, go, satan, shoot, doom, killing, burn |
| 13 | 11.591 | 2,41% | rats, goat, monkey, rat, cat, pig, silly, spider, pigs, ass |
| 22 | 10.304 | 2,14% | silly, stupid, man, guy, jonkler, ass, little, hes, fat, dumb |
| 4 | 9.915 | 2,06% | russian, russians, russia, toxic, full, get, putin, good, kids, people |
| 9 | 9.377 | 1,95% | nazis, nazi, killing, kill, shoot, shooting, fun, good, germans, get |
| 18 | 8.897 | 1,85% | music, soundtrack, ass, sound, voice, damn, audio, sonic, like, song |
| 20 | 8.867 | 1,84% | cheaters, cheat, anti, valve, full, cheater, cheating, vac, many, trash |
| 10 | 8.710 | 1,81% | gay, gaythis, piss, like, gays, dont, ass, good, homosexual, people |
| 19 | 8.603 | 1,79% | zombies, zombie, kill, shoot, killing, die, fun, get, head, friends |
| 17 | 8.120 | 1,69% | boobs, tits, nipples, big, good, nice, great, like, boob, stayed |
| 8 | 7.469 | 1,55% | boobs, tits, big, anime, girl, girls, nipples, like, ass, nice |
| 12 | 6.148 | 1,28% | gay, sex, im, furry, gays, lesbian, like, homosexual, love, people |
| 21 | 5.485 | 1,14% | chinese, china, english, japanese, players, asian, people, translation, french, language |

### 5.2 Tabela completa de tópicos — PT (16 linhas, incl. outlier)

| Tópico | Documentos | % | Palavras-chave (c-TF-IDF) |
|---|---|---|---|
| 0 | 10.346 | 30,71% | nao, lixo, pra, so, caralho, bom, voce, porcaria, valve, vai |
| 2 | 2.491 | 7,39% | lixo, nao, porcaria, dinheiro, compra, pra, ruim, comprar, pessimo, horrivel |
| 6 | 2.396 | 7,11% | caguei, pau, chorei, camisinha, caralho, confortavel, senhores, ja, bunda, pra |
| 4 | 2.295 | 6,81% | caralho, carlinhos, pra, pau, peixe, chupa, ai, porrada, vai, boca |
| 5 | 2.251 | 6,68% | bom, recomendo, caralho, pra, adoro, gosto, gostei, porrada, legal, pau |
| 1 | 2.081 | 6,18% | matar, matei, pessoas, mata, bom, morre, morrer, pra, morri, vc |
| 10 | 1.695 | 5,03% | matar, vc, bom, morrer, mata, morre, pra, voce, vai, amigos |
| 3 | 1.662 | 4,93% | sexo, faco, gay, bonito, mulheres, porque, penis, fazer, amigas, nao |
| 8 | 1.563 | 4,64% | clica, sexoporno, patetico, porno, morra, macaco, lixo, novus, lixoum, completamente |
| -1 (outlier) | 1.425 | 4,23% | odeio, voce, nao, pra, fodamuito, vai, bunda, so, caralho, vc |
| 7 | 1.206 | 3,58% | demonios, matar, demonio, inferno, diabo, porrada, bom, pra, capeta, vampiro |
| 11 | 1.082 | 3,21% | tiro, bomba, porrada, bala, ninjas, cortadores, cebola, bom, tiros, atirar |
| 9 | 1.034 | 3,07% | kkkkk, kkkk, kkkkkkk, kkkkkk, kkkkkkkkk, kkk, bom, pra, tao, cara |
| 14 | 847 | 2,51% | amigos, amigo, pra, porrada, bom, cagar, burro, legal, recomendo, divertido |
| 12 | 669 | 1,99% | nazistas, nazista, matar, comunista, nazi, nazis, zumbi, bom, pra, mata |
| 13 | 642 | 1,91% | argentino, argentinos, brasil, brasileiro, nao, so, pra, brasileiros, portugues, vai |

Fonte: `steam-data/step03-output/{en,pt}/results/topic_info_real_counts.csv`.

---

## 6. Coerência Semântica (`c_npmi`)

| | EN | PT |
|---|---|---|
| `c_npmi` na busca (Optuna, amostra de 30%) | 0,0021 | 0,0114 |
| `c_npmi` do modelo final (corpus de treino real) | 0,0498 | -0,0638 |

---

## 7. Rotulagem Semântica dos Tópicos (Stage 8)

**Configuração da execução:** amostragem aleatória, `n_examples = 15` fixo por tópico, seed fixa, sem detecção de copypasta, modelo `claude-fable-5`, via API (`run/08_label_topics.py`).

### 7.1 Distribuição por categoria

| Categoria | EN (de 24 tópicos) | PT (de 16 tópicos) |
|---|---|---|
| `product_complaint` | 6 | 2 |
| `humor_sarcasm` | 5 | 8 |
| `sexual_content` | 4 | 1 |
| `other` | 3 | 2 |
| `generic_insult` | 2 | 2 |
| `nationality_xenophobia` | 2 | 1 |
| `identity_hate` | 2 | 0 |
| `violence_threat` | 0 | 0 |

### 7.2 Tabela completa de rótulos — EN

| Tópico | Documentos | Label | Categoria | Descrição |
|---|---|---|---|---|
| 0 | 168.545 | Profane Game Complaints | product_complaint | Reviews expressing frustration or disappointment with games (quality, matchmaking, bugs, monetization) using generic profanity and insults like 'trash', 'garbage', and 'sucks', without a specific targeted theme. |
| 1 | 41.936 | In-Game Violence Humor | humor_sarcasm | Reviews describing in-game killing, shooting, and violence, mostly as hyperbolic or darkly humorous praise of gameplay (e.g., joking that a game 'made me kill my whole family'), flagged as toxic due to violent language rather than genuine threats. |
| 2 | 38.578 | Vulgar Product Bashing | product_complaint | Reviews denigrating games as worthless purchases using profane derogatory terms (e.g., 'garbage', 'trash', 'crap'), often citing bugs, poor controls, or feeling ripped off, without targeting people or groups. |
| -1 | 19.379 | Unclustered / Outliers | other | Documents that did not fit clearly into any topic cluster. |
| 7 | 15.976 | Generic Profane Banter | generic_insult | A broad, thematically incoherent cluster of reviews united only by casual profanity and mild insults (e.g., 'sucks', 'stupid', 'dumb', 'ass'), used both to praise and criticize games rather than to express any specific targeted toxicity. |
| 14 | 14.394 | Playful Profane Praise | humor_sarcasm | Largely positive or humorous reviews describing games as fun, silly, or stupid in an affectionate way, flagged as toxic due to casual profanity and mock insults rather than genuine hostility. |
| 16 | 13.874 | Crude Body Humor | humor_sarcasm | Reviews united by juvenile scatological and body-part vulgarity (references to buttocks, poop, and 'getting one's ass kicked'), mostly used as crude humor or hyperbolic game commentary rather than targeted abuse, with occasional sexual or insulting remarks mixed in. |
| 3 | 13.697 | Crude Sexual Jokes | sexual_content | Reviews making crude sexual references and jokes, often humorously requesting or lamenting the ability to have sex with game characters or comparing gameplay to sex or pornography. |
| 5 | 13.422 | Crude Sexual Humor | sexual_content | Reviews united by crude sexual references and jokes, using vulgar anatomical and sexual language (references to genitalia, sex acts, and pornographic content) typically as shock humor or hyperbole while commenting on games, often those with adult or suggestive themes. |
| 11 | 13.321 | Profane Game Critiques | product_complaint | Standard game reviews discussing story, characters, puzzles, and endings that were flagged as toxic due to casual profanity and mild insults used both in criticism and enthusiastic praise, rather than targeted abuse. |
| 6 | 12.111 | Crude Testicle Humor | humor_sarcasm | Mostly positive or joking game reviews united by vulgar comedic references to testicles and other crude meme humor (e.g., 'sussy', Among Us references) rather than genuine hostility, likely flagged as toxic due to profane anatomical language. |
| 15 | 12.010 | In-Game Violence Talk | other | Reviews describing in-game combat and violent gameplay themes (killing demons, DOOM-style shooting, boss fights) using aggressive language that refers to fictional game content rather than real-world targets, likely flagged as toxic due to violent vocabulary. |
| 13 | 11.591 | Profane Animal Commentary | humor_sarcasm | Reviews united by references to animals and creatures in games (rats, goats, monkeys, dinosaurs, frogs), expressed through crude, profanity-laden humor and exaggerated remarks rather than targeted hostility. |
| 22 | 10.304 | Male Figure Insults | generic_insult | Insults and mocking remarks directed at male individuals—game characters, villains, developers, and fellow players—calling them stupid, dumb, or useless, often in a joking or crude tone, with one example using an ableist slur against a developer. |
| 4 | 9.915 | Anti-Russian Hostility | nationality_xenophobia | Reviews expressing xenophobic hostility toward Russian players, blaming Russians for cheating, hacking, and toxic behavior in multiplayer games, often mixed with insults referencing Russia, Putin, and 'Russian kids' as scapegoats for poor game experiences. |
| 9 | 9.377 | Nazi-Killing Gameplay Praise | other | Positive game reviews celebrating in-game violence against Nazis and other enemies (WWII shooters and similar titles), flagged as toxic due to graphic descriptions of killing rather than genuine hostility toward real people. |
| 18 | 8.897 | Profane Soundtrack Commentary | product_complaint | Reviews discussing a game's music, soundtrack, sound effects, or audio quality (both praising and criticizing) that were flagged due to casual profanity rather than targeted abuse. |
| 20 | 8.867 | Cheater Complaints | product_complaint | Angry reviews complaining that multiplayer games (especially Valve titles) are overrun with cheaters and hackers, insulting the developers and the ineffective anti-cheat systems (e.g., VAC). |
| 10 | 8.710 | Casual Homophobic Language | identity_hate | Profanity-laden game reviews united mainly by the casual use of 'gay' and 'homosexual' as pejoratives or throwaway remarks, mixed with generic vulgar insults about games and their communities rather than targeted hate speech. |
| 19 | 8.603 | Zombie Shooter Reviews | product_complaint | Reviews of zombie and killer-versus-survivor games that were flagged due to violent gameplay vocabulary (kill, shoot, die) and profanity-laden complaints or crude jokes about game mechanics, bots, and updates, rather than genuine interpersonal toxicity. |
| 17 | 8.120 | Sexualized Game Commentary | sexual_content | Crude, profanity-laden game reviews united by casual sexualized remarks, most prominently repeated references to female breasts and characters' bodies as a selling point, mixed with generic vulgar praise or criticism of games. |
| 8 | 7.469 | Sexualized Character Commentary | sexual_content | Reviews containing crude sexual remarks about female game characters and anime-style women, focusing on breasts, buttocks, and sexualized appearance, often framed as juvenile praise for a game's sexual content. |
| 12 | 6.148 | Homophobic Remarks | identity_hate | Reviews united by references to homosexuality, using 'gay' and related terms either as a pejorative insult toward games and players, in crude sexual jokes, or occasionally combined with racial slurs, mixing homophobic mockery with juvenile humor. |
| 21 | 5.485 | Nationality Based Hostility | nationality_xenophobia | Reviews expressing hostility or complaints tied to nationalities, countries, and languages—especially China and Chinese players—including blaming national player bases for toxicity, political rants (e.g., Taiwan, Steam China), and gripes about translations and regional versions, mixed with some generic game complaints. |

### 7.3 Tabela completa de rótulos — PT

| Tópico | Documentos | Label | Categoria | Descrição |
|---|---|---|---|---|
| 0 | 10.346 | Profane Game Opinions | product_complaint | A broad catch-all cluster of Brazilian Portuguese reviews expressing opinions about game quality — both harshly negative (calling games 'trash' or 'garbage') and enthusiastically positive — unified only by heavy use of profanity and crude sexual expressions rather than a coherent toxic theme. |
| 2 | 2.491 | Profane Game Bashing | product_complaint | Portuguese-language reviews denouncing games as worthless trash ("lixo", "porcaria") and a waste of money, urging others not to buy while venting with heavy profanity about poor optimization, bad servers, hackers, and unresponsive developers. |
| 6 | 2.396 | Crude Sexual Humor | humor_sarcasm | Brazilian Portuguese reviews using vulgar sexual and scatological slang in a joking, meme-like register to express (often positive) opinions about games, rather than to attack anyone. |
| 4 | 2.295 | Vulgar Portuguese Profanity | generic_insult | Brazilian Portuguese reviews united by heavy use of generic profanity and crude slang (e.g., 'caralho', 'pau', 'toma no c') applied indiscriminately to both praise and complaints about games, rather than any coherent targeted theme. |
| 5 | 2.251 | Profane Praise | humor_sarcasm | Enthusiastic positive Brazilian Portuguese reviews that recommend games using heavy profanity as intensifiers (e.g., vulgar equivalents of 'damn good'), occasionally mixed with crude sexual jokes and a stray homophobic slur, flagged as toxic due to swearing rather than hostile intent. |
| 1 | 2.081 | Gameplay Killing Descriptions | other | Reviews describing in-game violence, killing enemies, and repeatedly dying as part of normal gameplay, using violent vocabulary (matar, morrer) that triggered the toxicity classifier despite mostly expressing enjoyment or frustration with game mechanics rather than real-world threats. |
| 10 | 1.695 | In-Game Killing Banter | humor_sarcasm | Portuguese reviews using violent vocabulary (kill, die, exterminate) to hyperbolically describe gameplay experiences—killing enemies, dying repeatedly, or jokingly threatening developers/players—which likely triggered the toxicity classifier despite being mostly gaming banter rather than genuine threats. |
| 3 | 1.662 | Crude Sexual Commentary | sexual_content | Portuguese-language reviews united by explicit sexual content, including crude jokes about sex in games, demands for sexual/pornographic content, graphic descriptions of sex acts and genitalia, and repetitive spam of the word 'sexo'. |
| 8 | 1.563 | Incoherent Profane Rants | generic_insult | A thematically incoherent cluster of Brazilian Portuguese reviews consisting of profanity-laden outbursts, repeated-phrase spam, memetic slang, and scattered insults or death wishes with no unifying target or narrative. |
| -1 | 1.425 | Unclustered / Outliers | other | Documents that did not fit clearly into any topic cluster. |
| 7 | 1.206 | Demon Slaying Hyperbole | humor_sarcasm | Enthusiastic, profanity-laden Portuguese reviews praising games with demonic or supernatural themes (e.g., beating up the devil, killing demons, vampires), where crude and violent language expresses excitement and dark humor rather than genuine hostility. |
| 11 | 1.082 | Vulgar Shooter Commentary | humor_sarcasm | Profanity-laden Portuguese reviews of shooter and action games, united by the memetic slang phrase 'tiro, porrada e bomba' (shooting, brawling and bombs), mixing crude enthusiasm for combat mechanics with vulgar complaints about gunplay, hit registration and difficulty. |
| 9 | 1.034 | Brazilian Joke Reviews | humor_sarcasm | Brazilian Portuguese reviews united by the 'kkkk' laughter marker, using casual profanity and crude humor to jokingly praise or mock games rather than express targeted hostility. |
| 14 | 847 | Profane Friend Recommendations | humor_sarcasm | Mostly positive, humorous reviews recommending games to play with friends, flagged as toxic due to heavy Portuguese profanity and playful insults directed at friends (e.g., calling them idiots or stupid) rather than genuine hostility. |
| 12 | 669 | Nazi Killing Gameplay | humor_sarcasm | Mostly positive Portuguese game reviews (likely for WWII shooters) that were flagged for violent language, as users enthusiastically praise the games as simulators for killing Nazis, Germans, communists, or zombies, often with profanity and political banter about fascists and capitalists. |
| 13 | 642 | Lusophone Nationality Toxicity | nationality_xenophobia | Brazilian Portuguese reviews mixing profanity-laden game complaints with derogatory or xenophobic remarks targeting nationalities and regional groups, notably Argentinians, Spanish speakers, and Northeastern Brazilians. |

Fonte: `steam-data/step03-output/{en,pt}/results/topic_labels.csv`.

---

## 8. Índice de Fontes

| Seção | Arquivo-fonte |
|---|---|
| Corpus | `steam-data/step03-output/{en,pt}/results/topic_info_real_counts.csv`, `METHODOLOGY_REPORT.md` §4.1, §6.2 |
| Hiperparâmetros (Stage 3) | `steam-data/step03-output/{en,pt}/optuna/best_params.json` |
| Estabilidade (Stage 4) | `METHODOLOGY_REPORT.md` §4.3–§4.4, §5 |
| Treino final (Stage 5) | MLflow `bertopic_final_training`, `results/logs/05_train.log` |
| Classificação (Stage 6/7) | `steam-data/step03-output/{en,pt}/results/topic_info_real_counts.csv`, `classified_toxic.parquet` |
| Coerência | MLflow (busca) + recálculo pós-hoc (`METHODOLOGY_REPORT.md` §6.5) |
| Rotulagem (Stage 8) | `steam-data/step03-output/{en,pt}/results/topic_labels.csv` |
