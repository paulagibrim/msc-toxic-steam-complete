"""The coded review table behind the four non-year literature-review charts.

Why this is a hand-written table rather than a parser over the Notion export
(`ETP03 - Systematic Review ...csv`): the export's `Dados de Dataset` column
is free prose, one paragraph per paper, in inconsistent shapes - "2.177.488
partidas", "1,46 milhões de jogadores", "Tamanho: 80 posts, 10
participantes". A regex over it can find numbers but cannot decide which
category each number belongs to, and that decision is the entire content of
these charts. It is written out once, here, where it can be read and
corrected.

The Notion export's `Citation Key` column is NOT trustworthy as an
identifier - several rows carry a key belonging to a different paper (the
row keyed `saleousUninstallNoobViews2023` holds Schulenberg's 31 interviews,
`kordyakaUnifiedTheoryToxic2020` holds Rubin's Something Awful thread, and
`blackburnSTFUNOOBPredicting2014` holds Hilvert-Bruce's survey). The keys
below are this project's own, from Selected-papers.bib, matched to the
export by title and DOI rather than by its key column.

VALIDATION. These numbers are not eyeballed off the reference figures; they
are read from the export and then *checked* against them. Feeding the five
lists below through matplotlib's boxplot reproduces the reference boxplot's
quartiles, whiskers and outlier positions for USERS, MATCHES and
PARTICIPANTS exactly - e.g. participants come out Q1 120 / median 267.5 /
Q3 469 / upper whisker 959 / lone outlier 1646, against roughly 120 / 270 /
450 / 950 / 1650 read off the reference. Group sizes likewise land on
16/14/10/5/4, the reference count chart's bars. That agreement is the
evidence the coding is right; if you edit a number here, re-check it.

The one known departure is documented at ENTREVISTAS below.

Sources of the per-chart counts:
  - PLATFORMS / GAMES: the reference figures' own bar values, which are the
    review's published numbers. The export's `Plataforma(s)` column
    reproduces them to within one paper (it yields Twitch 5 where the
    figure shows 4, and LoL 20 where the figure shows 19), so the figures
    win - they are what the dissertation already reports.
  - DATA TYPES / VOLUMES: the export's `Dados de Dataset` column, coded
    below.
"""

# ---------------------------------------------------------------------------
# Non-game platforms studied, from the reference figure's bar values.
# ---------------------------------------------------------------------------
PLATFORMS = [
    ("Twitter", 6),
    ("Reddit", 5),
    ("Twitch", 4),
    ("Metacritic", 2),
    ("Youtube", 2),
    ("Amino", 1),
    ("Discord", 1),
    ("Facebook", 1),
]

# ---------------------------------------------------------------------------
# Games studied, from the reference figure's bar values.
# ---------------------------------------------------------------------------
GAMES = [
    ("League Of Legends", 19),
    ("Dota 2", 10),
    ("Overwatch", 4),
    ("World Of Tanks", 3),
    ("For Honor", 2),
    ("Cs:Go", 2),
    ("Rainbow Six Siege", 2),
    ("The Last Of Us Part 2", 1),
]

# ---------------------------------------------------------------------------
# Dataset volumes, one entry per (paper, data type) pair.
#
# A paper appears once per category it reports, so the papers that published
# two different units appear twice - Yang 2023 contributes both 8,653 matches
# and 388,785 comments, Ribeiro 2024 both 1,017,504 comments and 333,643
# users. That is deliberate: the categories count *datasets*, not papers, and
# the count chart's bars sum to more than 49 for exactly this reason.
#
# Only explicitly stated totals are included. Papers reporting a volume in a
# unit no category covers are left out rather than converted - Laato's "600
# hours of gameplay", Irwin's "50 hours of structured observation", Wijkstra's
# 1,906 screened articles (a literature review, not a game dataset), and
# Kocielnik's "millions of players" with no stated total.
# ---------------------------------------------------------------------------

# Comments / messages / posts / chat lines / tweets. Rubin's 80 forum posts
# are deliberately absent: the reference boxplot's lower whisker sits at
# ~950, i.e. Siutila's 952 is the group minimum, so that paper was coded to
# its 10 participants only (see PARTICIPANTES).
COMENTARIOS = [
    ("siutila:2018", 952),            # Reddit comments
    ("deslauriers:2020", 1_848),      # r/deadbydaylight posts
    ("vandorpe:2023", 16_008),        # synthetic chat lines
    ("kou:2020", 40_133),             # r/leagueoflegends comments
    ("dreier:2023", 100_000),         # Twitch messages
    ("yang:2023", 388_785),           # in-game chat comments
    ("cornel:2019", 764_722),         # Dota + Ragnarok chat messages
    ("ribeiro:2024", 1_017_504),      # Metacritic reviews
    ("melo:2022", 1_046_040),         # tweets
    ("chatzakou:2017", 1_650_000),    # #GamerGate tweets
    ("cook:2020", 1_697_222),         # Tribunal chat messages
    ("sengun:2019", 2_000_000),       # in-game chat lines
    ("blackburn:2014", 10_000_000),   # Tribunal user reports
    ("kwak:2015b", 11_000_000),       # Tribunal reports
    ("yang:2024", 29_000_000),        # For Honor chat lines
    ("kwak:2015a", 57_000_000),       # Tribunal in-game messages
]

# Survey respondents / study participants.
PARTICIPANTES = [
    ("rubin:2013", 10),
    ("bernardes:2016", 10),
    ("turkay:2020", 19),
    ("kordyaka:2022", 116),
    ("reid:2022a", 132),
    ("trudgettklose:2024", 145),
    ("kordyaka:2023c", 230),
    ("kordyaka:2023a", 305),
    ("kordyaka:2020", 320),
    ("mattinen:2018", 373),
    ("mclean:2020", 501),
    # 816 and 1,646 were initially attributed to the wrong papers. In the
    # Notion export the `Title` column is shifted by one row against
    # `DOI`/`Citation Key`, so the row keyed hilvert-bruceImJustTrolling2020
    # is titled "Who Are the Cyka Blyat?" and the row keyed
    # blackburnSTFUNOOBPredicting2014 is titled "I'm just trolling". The
    # title is the reliable one: 816 is Lemercier-Dugarin's questionnaire,
    # and the 1,646/2,042 survey is Hilvert-Bruce's - Blackburn's STFU NOOB
    # is a 10M-report Tribunal study, not a survey at all.
    ("lemercierdugarin:2021", 816),
    ("poeller:2023", 959),
    ("hilvertbruce:2020", 1_646),
]

# Matches / battles / games analysed.
PARTIDAS = [
    ("reid:2022b", 48),
    ("mclean:2020", 56),
    ("yang:2023", 8_653),
    ("martens:2015", 12_923),
    ("esmaeili:2016", 23_750),
    ("sengun:2019", 60_000),
    ("balci:2014", 800_000),
    ("shen:2020", 1_300_000),
    ("kwak:2015a", 2_107_522),
    ("neto:2018", 2_177_488),
]

# Distinct users / players in the dataset.
USUARIOS = [
    ("jiang:2016", 401),
    ("balci:2014", 100_000),
    ("ribeiro:2024", 333_643),
    ("blackburn:2014", 1_460_000),
]

# Interviews conducted.
#
# Two entries here are reconstructions rather than straight reads, and both
# are forced by the reference boxplot rather than chosen. That figure puts
# this group's box at 20-31 with the median on 22 and a lone outlier near
# 2,000 - which is only reachable from {12, 20, 22, 31, 2042}. Irwin's 15
# interviews therefore were NOT counted here by the original review (with
# them in, Q1 falls to 15 and the box bottom drops a full step), and
# Hilvert-Bruce's "2042 respostas" WAS, despite being an online
# questionnaire. That double-counts Hilvert-Bruce, whose 1,646 valid
# participants also appear under PARTICIPANTES.
#
# Both look like coding slips in the original rather than decisions, but
# they are reproduced because the published figure is what the dissertation
# already reports. If the review is ever recoded, the defensible set is
# {12, 15, 20, 22, 31} - drop HILVERT_BRUCE_SURVEY below and restore
# ("irwin:2021", 15) - which changes this box and nothing else.
ENTREVISTAS = [
    ("laato:2024", 12),
    ("liu:2023", 20),
    ("kim:2024", 22),
    ("schulenberg:2023", 31),
    ("hilvertbruce:2020", 2_042),   # survey responses; see note above
]

# Display order for the count chart: descending by number of papers, which is
# the order the reference figure uses.
DATA_TYPES = [
    ("COMMENTS", COMENTARIOS),
    ("PARTICIPANTS", PARTICIPANTES),
    ("MATCHES", PARTIDAS),
    ("INTERVIEWS", ENTREVISTAS),
    ("USERS", USUARIOS),
]

# Display order for the boxplot: descending by median volume, matching the
# reference figure (comments, users, matches, participants, interviews) -
# a different order from DATA_TYPES, because the reference orders the count
# chart by paper count and the boxplot by magnitude.
VOLUME_ORDER = ["COMMENTS", "USERS", "MATCHES", "PARTICIPANTS", "INTERVIEWS"]

BY_NAME = {
    "COMMENTS": COMENTARIOS,
    "PARTICIPANTS": PARTICIPANTES,
    "MATCHES": PARTIDAS,
    "INTERVIEWS": ENTREVISTAS,
    "USERS": USUARIOS,
}


def values(name: str) -> list:
    """The bare volume numbers for a data-type group, dropping the keys."""
    return [n for _, n in BY_NAME[name]]
