"""Recupera perspective_score, detoxify_score e sentiment_score (nessa ordem)
das reviews usadas como exemplo no texto da dissertacao.

Nao re-pontua nada: os tres scores ja existem no corpus (step02 gerou
perspective/detoxify, step04 acrescentou sentiment ao mesmo arquivo), entao
isto e so uma varredura procurando cada exemplo pelo texto. Roda em CPU -
nao carrega nenhum modelo, nao precisa de GPU.

Como cada exemplo e localizado
------------------------------
Os exemplos aparecem no .tex censurados (`\\faHeart` no lugar de cada
caractere de palavrao) e com marcacao LaTeX (`[\\ldots]`, `{ }`), entao
casar o texto inteiro nao funciona. Cada entrada de EXAMPLES traz:

  - `key`: um trecho curto, sem acento e sem censura, usado como pre-filtro
    barato (regex case-insensitive direto na coluna review_text);
  - `anchor`: um trecho maior, tambem sem censura, conferido depois da
    normalizacao (minusculas, sem acentos, espacos colapsados) sobre as
    poucas linhas que passaram no pre-filtro;
  - `game`: o jogo do exemplo - vira um filtro por game_id (varias frases,
    tipo "this game makes me want to kill myself", aparecem em centenas de
    reviews de jogos diferentes; sem esse filtro o anchor sozinho nao
    identifica a review);
  - `full`: o texto completo, so para os exemplos que nao estao censurados -
    quando bate exatamente (ja descontado o boilerplate que os modelos
    tambem descartam) a linha e marcada como MATCH EXATO e as parciais sao
    omitidas.

O anchor nunca cobre a parte censurada da review - so o miolo legivel -,
o que basta para identificar a linha sem depender de adivinhar quais
caracteres estavam sob os `\\faHeart` (que no dado cru sao os `♥` com que a
propria Steam censura o texto).

Saida
-----
Imprime, por exemplo/\\label, quantas linhas casaram e os scores no mesmo
formato usado no .tex (`[persp][detox][sent]`, 4/4/2 casas), e compara com
o valor ja escrito no texto quando existe (`expected`), marcando divergencia
com "!= texto". Tambem grava um CSV com a linha completa de cada match.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from pipeline_utils import info, list_parquet_files
from show_review_examples import clean_review_text

# Os oito exemplos do texto. `expected` sao os scores que ja constam no .tex
# (None = e justamente o que falta descobrir); servem de conferencia para os
# que ja estao la.
EXAMPLES = [
    {
        "label": "ex:porcaria-critique",
        "lang": "pt",
        "game": "Counter-Strike 2",
        "key": "porcaria imunda",
        "anchor": "counter strike 2 e uma porcaria imunda",
        "full": None,  # censurado/abreviado no texto
        "expected": (0.5727, 0.9742, 1.37),
    },
    {
        "label": "ex:porcaria-hostile",
        "lang": "pt",
        "game": "Battlefield 1",
        "key": "porcaria da steam",
        "anchor": "porcaria da steam ta deixando 2 reais pendente na minha carteira",
        "full": (
            "Simplesmente vou perder a promoção pq a porcaria da steam TA DEIXANDO 2 REAIS "
            "PENDENTE NA MINHA CARTEIRA, VTMNC MORRA GORDO ARROMBADO"
        ),
        "expected": (0.8300, 0.9742, 1.37),
    },
    {
        "label": "ex:kill-gameplay",
        "lang": "en",
        "game": "Warhammer 40,000: Darktide",
        "key": "praise the emperor",
        "anchor": "praise the emperor. kill the heretics",
        "full": "Praise the Emperor. Kill the heretics.",
        "expected": None,
    },
    {
        "label": "ex:matar-gameplay",
        "lang": "pt",
        "game": "Counter-Strike 2",
        "key": "matar na zeus",
        "anchor": "gosto de matar na zeus",
        "full": "gosto de matar na zeus",
        "expected": None,
    },
    {
        "label": "ex:kill-ideation",
        "lang": "en",
        "game": "A Dance of Fire and Ice",
        "key": "wanna kill myself after the first real level",
        "anchor": "i wanna kill myself after the first real level",
        "full": (
            "I wanna kill myself after the first real level (there totally isn't a huge "
            "difficulity spike) i havent gotten past the first level after a while and i "
            "still wanna kill myself"
        ),
        "expected": None,
    },
    {
        "label": "ex:kill-ideation-2",
        "lang": "en",
        "game": "DARK SOULS: REMASTERED",
        "key": "makes me want to kill myself",
        "anchor": "this game makes me want to kill myself",
        "full": "This game makes me want to kill myself.",
        "expected": (0.6343, 0.4599, 4.62),
    },
    {
        "label": "ex:matar-idiom",
        "lang": "pt",
        "game": "DOOM + DOOM II",
        "key": "matar sdd",
        "anchor": "comprei pra matar sdd, muito bom",
        "full": "comprei pra matar sdd, muito bom.",
        "expected": (0.1992, 0.2788, 3.93),
    },
    {
        "label": "ex:matar-hostile",
        "lang": "pt",
        "game": "Inside the Backrooms",
        "key": "lei do retorno",
        "anchor": "lei do retorno eu vou matar qm fez esse codigo lixo",
        "full": None,  # censurado no texto
        "expected": (0.6408, 0.9073, 1.17),
    },
]

# step04 reescreve o arquivo do step02 com sentiment_score junto, entao ele
# sozinho ja tem os tres scores; step02 e o fallback (sentiment vazio) para
# quando o step04 ainda nao rodou para aquela lingua.
DEFAULT_STEP04_DIR = Path("../../steam-data/step04-output")
DEFAULT_STEP02_DIR = Path("../../steam-data/step02-output")
DEFAULT_GAMES_PATH = Path("../../steam-data/step01-output/games/games.parquet")

MAX_ROWS_PRINTED = 10  # quando nenhuma linha e identica ao exemplo; o CSV traz todas

WANTED_COLUMNS = [
    "review_url", "game_id", "review_text", "is_recommended", "review_lang",
    "perspective_declared_language", "perspective_score", "detoxify_score", "sentiment_score",
]


def normalize(text) -> str:
    """Minusculas, sem acentos, espacos colapsados - o mesmo tratamento
    aplicado nos dois lados da comparacao (anchor e review_text), para que
    diferenca de acentuacao/quebra de linha nao impeca o match."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title) -> str:
    """Como normalize, mas tambem descarta pontuacao - os titulos do
    games.parquet trazem simbolos que o texto da dissertacao nao repete
    ("Battlefield™ 1" vs "Battlefield 1", "Warhammer 40,000: Darktide")."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", normalize(title))).strip()


def resolve_game_ids(games: pd.DataFrame, examples: list) -> dict:
    """Mapeia cada exemplo -> conjunto de game_ids cujo titulo bate com o jogo
    citado no texto (comparacao normalizada, por substring nos dois sentidos,
    para tolerar sufixo de edicao no titulo do catalogo)."""
    titles = games["title"].map(normalize_title)
    game_ids = {}
    for ex in examples:
        wanted = normalize_title(ex["game"])
        mask = titles.apply(lambda t: bool(t) and (wanted in t or t in wanted))
        ids = set(games.loc[mask, "game_id"])
        if not ids:
            info(f"[{ex['label']}] Nenhum game_id para {ex['game']!r} - vai casar so pelo texto")
        game_ids[ex["label"]] = ids
    return game_ids


def _resolve_lang_source(base_dir: Path, lang: str) -> Path:
    """Mesma logica de show_review_examples._resolve_lang_source: o output do
    step02/step04 ja apareceu nos dois layouts (subpasta review_lang=<lang>/
    ou tudo junto com review_lang como coluna)."""
    subfolder = base_dir / f"review_lang={lang}"
    return subfolder if subfolder.is_dir() else base_dir


def _readable_columns(path: Path) -> list:
    import pyarrow.parquet as pq
    names = set(pq.read_schema(path).names)
    return [c for c in WANTED_COLUMNS if c in names]


def scan_language(base_dir: Path, lang: str, examples: list, game_ids: dict) -> pd.DataFrame:
    """Varre os parquets de uma lingua arquivo a arquivo (nunca segura o
    corpus inteiro em memoria) e devolve so as linhas que casam com algum
    anchor (dentro do jogo daquele exemplo), com as colunas `label` e
    `is_exact` acrescentadas."""
    source = _resolve_lang_source(base_dir, lang)
    files = list_parquet_files(source)
    if not files:
        info(f"[{lang}] Nenhum parquet em {source} - pulando")
        return pd.DataFrame()

    columns = _readable_columns(files[0])
    missing = [c for c in WANTED_COLUMNS if c not in columns]
    if missing:
        info(f"[{lang}] Colunas ausentes em {source.name}: {missing} (virao vazias)")

    # Pre-filtro barato: uma unica regex com as chaves de todos os exemplos
    # desta lingua, aplicada direto no texto cru antes de qualquer
    # normalizacao (que e cara em milhoes de linhas).
    coarse = "|".join(re.escape(ex["key"]) for ex in examples)

    hits = []
    for i, f in enumerate(files, start=1):
        df = pd.read_parquet(f, columns=columns)
        candidates = df[df["review_text"].str.contains(coarse, case=False, na=False, regex=True)]
        if candidates.empty:
            continue

        candidates = candidates.copy()
        candidates["_normalized"] = candidates["review_text"].map(normalize)
        # Igualdade exata e conferida contra o texto sem boilerplate: as
        # frases que o step02/03/04 descartam antes de pontuar ("Produto
        # recebido de graca", "Analise de acesso antecipado") vem coladas no
        # review_text cru e nunca aparecem no exemplo citado na dissertacao.
        candidates["_clean"] = candidates["review_text"].map(lambda t: normalize(clean_review_text(t)))

        for ex in examples:
            matched = candidates[candidates["_normalized"].str.contains(ex["anchor"], regex=False)]
            ids = game_ids.get(ex["label"]) or set()
            if ids:
                matched = matched[matched["game_id"].isin(ids)]
            if matched.empty:
                continue
            full = normalize(ex.get("full")) if ex.get("full") else None
            matched = matched.copy()
            matched["is_exact"] = False if full is None else (matched["_clean"] == full)
            matched = matched.drop(columns=["_normalized", "_clean"])
            matched["label"] = ex["label"]
            matched["source_file"] = f.name
            hits.append(matched)
        info(f"[{lang}] {i}/{len(files)} arquivos varridos - {sum(len(h) for h in hits)} match(es) ate agora")

    if not hits:
        return pd.DataFrame()
    return pd.concat(hits, ignore_index=True)


def _fmt(value, decimals: int) -> str:
    return "?" * 6 if pd.isna(value) else f"{value:.{decimals}f}"


def report(matches: pd.DataFrame, examples: list) -> None:
    """Imprime, por exemplo, os scores no formato do .tex e sinaliza os casos
    que precisam de decisao manual: nenhum match, mais de um match, ou score
    divergente do que ja esta escrito no texto."""
    print()
    print("=" * 100)
    print("SCORES POR EXEMPLO  -  formato [perspective][detoxify][sentiment], como no .tex")
    print("=" * 100)

    for ex in examples:
        rows = matches[matches["label"] == ex["label"]] if not matches.empty else pd.DataFrame()
        print()
        print(f"{ex['label']}  ({ex['lang']}, {ex['game']})")

        if rows.empty:
            print("  SEM MATCH - nenhuma review do corpus contem o anchor nesse jogo")
            print(f"  anchor: {ex['anchor']!r}")
            continue

        exact = rows[rows["is_exact"]] if "is_exact" in rows.columns else pd.DataFrame()
        if not exact.empty:
            partial = len(rows) - len(exact)
            print(f"  MATCH EXATO: {len(exact)} review(s)"
                  + (f" ({partial} match(es) parcial(is) omitido(s) - veja o CSV)" if partial else ""))
            rows = exact
        elif len(rows) > 1:
            print(f"  ATENCAO: {len(rows)} reviews casaram e nenhuma e identica ao texto citado "
                  f"- confira qual e a do exemplo")
            rows = rows.head(MAX_ROWS_PRINTED)

        for _, row in rows.iterrows():
            persp, detox, sent = row["perspective_score"], row["detoxify_score"], row.get("sentiment_score")
            latex = f"[{_fmt(persp, 4)}][{_fmt(detox, 4)}][{_fmt(sent, 2)}]"
            flag = ""
            if ex["expected"] is not None:
                e_persp, e_detox, e_sent = ex["expected"]
                same = (
                    not pd.isna(persp) and abs(persp - e_persp) < 5e-5
                    and not pd.isna(detox) and abs(detox - e_detox) < 5e-5
                    and not pd.isna(sent) and abs(sent - e_sent) < 5e-3
                )
                flag = "  (confere com o texto)" if same else (
                    f"  != texto [{e_persp:.4f}][{e_detox:.4f}][{e_sent:.2f}]"
                )
            print(f"  {latex}{flag}")
            print(f"    game_name: {row.get('game_name')}")
            print(f"    review_url: {row['review_url']}")
            print(f"    is_recommended: {row['is_recommended']}  "
                  f"review_lang: {row.get('review_lang')}  "
                  f"perspective_declared_language: {row.get('perspective_declared_language')}")
            text = row["review_text"]
            print(f"    review_text: {text if isinstance(text, str) else ''}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recupera perspective/detoxify/sentiment das reviews usadas como exemplo no texto."
    )
    parser.add_argument(
        "--scores-dir", type=Path, default=None,
        help=f"Diretorio com os scores (default: {DEFAULT_STEP04_DIR} se existir, senao {DEFAULT_STEP02_DIR})",
    )
    parser.add_argument(
        "--games", type=Path, default=DEFAULT_GAMES_PATH,
        help=f"games.parquet do step01, para o game_name de conferencia (default: {DEFAULT_GAMES_PATH})",
    )
    parser.add_argument(
        "--label", action="append", default=None,
        help="Roda so o(s) \\label indicado(s) (repetivel). Default: todos os oito.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("../../steam-data/examples/example_scores.csv"),
        help="CSV com a linha completa de cada match",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    scores_dir = args.scores_dir
    if scores_dir is None:
        scores_dir = DEFAULT_STEP04_DIR if DEFAULT_STEP04_DIR.exists() else DEFAULT_STEP02_DIR
    if not scores_dir.exists():
        raise SystemExit(f"Diretorio de scores nao encontrado: {scores_dir} (passe --scores-dir)")
    info(f"Lendo scores de: {scores_dir}")

    examples = EXAMPLES
    if args.label:
        wanted = set(args.label)
        examples = [ex for ex in EXAMPLES if ex["label"] in wanted]
        unknown = wanted - {ex["label"] for ex in EXAMPLES}
        if unknown:
            raise SystemExit(f"\\label desconhecido(s): {sorted(unknown)}")

    if args.games.exists():
        games = pd.read_parquet(args.games, columns=["game_id", "title"])
        game_ids = resolve_game_ids(games, examples)
    else:
        info(f"games.parquet nao encontrado em {args.games} - sem filtro por jogo "
             f"(frases curtas vao casar com muitas reviews) e sem game_name")
        games, game_ids = None, {}

    frames = []
    for lang in sorted({ex["lang"] for ex in examples}):
        lang_examples = [ex for ex in examples if ex["lang"] == lang]
        info(f"[{lang}] Procurando {len(lang_examples)} exemplo(s)")
        found = scan_language(scores_dir, lang, lang_examples, game_ids)
        if not found.empty:
            found["review_lang"] = found.get("review_lang", lang)
            frames.append(found)

    matches = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not matches.empty:
        if games is not None:
            matches = matches.merge(games.rename(columns={"title": "game_name"}), on="game_id", how="left")
        else:
            matches["game_name"] = pd.NA

    report(matches, examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print()
    info(f"{len(matches)} linha(s) salva(s) em: {args.output}")


if __name__ == "__main__":
    main()
