"""Deterministic generator for a synthetic, internally-consistent genealogy corpus.

Why synthetic: the public repo stays free of real-person PII, and a generated
tree is internally consistent, so every gold answer is *derivable* from the tree
rather than asserted by hand. Drop a real GEDCOM in via `genealogy_rag.corpus`
to run the same pipeline over your own data locally.

Outputs (all deterministic; seeded):
  data/genealogy/documents.jsonl  -- the retrieval corpus (one source per line)
  data/genealogy/graph.json       -- persons, relationships, person<->doc edges
  eval/questions.jsonl            -- gold QA with computed relevant_doc_ids

Run: python data/generate_corpus.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx

SEED = 17
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "genealogy"
EVAL = ROOT / "eval"

# --------------------------------------------------------------------------- #
# 1. The family, defined as data. (id, name, sex, birth, death, place, occ)
# --------------------------------------------------------------------------- #
PEOPLE: dict[str, dict] = {
    # Generation 1
    "P01": dict(name="Josiah Calloway", sex="M", birth=1851, death=1922,
                place="Greene County, Georgia", occ="farmer"),
    "P02": dict(name="Martha Whitfield", sex="F", birth=1855, death=1931,
                place="Morgan County, Georgia", occ="keeping house"),
    "P03": dict(name="Henry Brandt", sex="M", birth=1848, death=1910,
                place="Charleston, South Carolina", occ="blacksmith"),
    "P04": dict(name="Catherine Voss", sex="F", birth=1852, death=1928,
                place="Savannah, Georgia", occ="seamstress"),
    # Generation 2
    "P05": dict(name="Thomas Calloway", sex="M", birth=1878, death=1944,
                place="Greene County, Georgia", occ="merchant"),
    "P06": dict(name="Susan Calloway", sex="F", birth=1881, death=1960,
                place="Greene County, Georgia", occ="schoolteacher"),
    "P07": dict(name="Margaret Brandt", sex="F", birth=1879, death=1962,
                place="Charleston, South Carolina", occ="keeping house"),
    "P08": dict(name="Wilhelm Brandt", sex="M", birth=1875, death=1951,
                place="Charleston, South Carolina", occ="machinist"),
    "P09": dict(name="Frank Mercer", sex="M", birth=1877, death=1949,
                place="Augusta, Georgia", occ="railroad clerk"),
    "P10": dict(name="Ingrid Halvorsen", sex="F", birth=1882, death=1965,
                place="Minneapolis, Minnesota", occ="keeping house"),
    # Generation 3
    "P11": dict(name="Robert Calloway", sex="M", birth=1903, death=1978,
                place="Atlanta, Georgia", occ="accountant"),
    "P12": dict(name="Alice Calloway", sex="F", birth=1906, death=1989,
                place="Atlanta, Georgia", occ="librarian"),
    "P13": dict(name="James Calloway", sex="M", birth=1909, death=1985,
                place="Atlanta, Georgia", occ="engineer"),
    "P14": dict(name="Clara Mercer", sex="F", birth=1905, death=1988,
                place="Augusta, Georgia", occ="nurse"),
    "P15": dict(name="Otto Brandt", sex="M", birth=1907, death=1990,
                place="Chicago, Illinois", occ="toolmaker"),
    "P16": dict(name="Helen Park", sex="F", birth=1908, death=1995,
                place="Knoxville, Tennessee", occ="keeping house"),
    "P17": dict(name="Charles Ainsworth", sex="M", birth=1904, death=1981,
                place="Macon, Georgia", occ="attorney"),
    "P18": dict(name="Dorothy Klein", sex="F", birth=1911, death=1998,
                place="Cincinnati, Ohio", occ="keeping house"),
    "P25": dict(name="Greta Lindqvist", sex="F", birth=1910, death=2001,
                place="Chicago, Illinois", occ="keeping house"),
    # Generation 4
    "P19": dict(name="David Calloway", sex="M", birth=1932, death=2010,
                place="Atlanta, Georgia", occ="physician"),
    "P20": dict(name="Ruth Calloway", sex="F", birth=1935, death=2018,
                place="Atlanta, Georgia", occ="architect"),
    "P21": dict(name="Nancy Ainsworth", sex="F", birth=1931, death=2015,
                place="Macon, Georgia", occ="journalist"),
    "P22": dict(name="Walter Calloway", sex="M", birth=1936, death=2012,
                place="Cincinnati, Ohio", occ="chemist"),
    "P23": dict(name="Grace Calloway", sex="F", birth=1939, death=2021,
                place="Cincinnati, Ohio", occ="teacher"),
    "P24": dict(name="Erik Brandt", sex="M", birth=1933, death=2008,
                place="Chicago, Illinois", occ="draftsman"),
}

# child_id -> (father_id, mother_id)
PARENTS: dict[str, tuple[str, str]] = {
    "P05": ("P01", "P02"), "P06": ("P01", "P02"),
    "P07": ("P03", "P04"), "P08": ("P03", "P04"),
    "P11": ("P05", "P07"), "P12": ("P05", "P07"), "P13": ("P05", "P07"),
    "P14": ("P09", "P06"),
    "P15": ("P08", "P10"),
    "P19": ("P11", "P16"), "P20": ("P11", "P16"),
    "P21": ("P17", "P12"),
    "P22": ("P13", "P18"), "P23": ("P13", "P18"),
    "P24": ("P15", "P25"),
}

# (spouse1, spouse2, year, place)  -- the single connecting union is P05 x P07
MARRIAGES: list[tuple[str, str, int, str]] = [
    ("P01", "P02", 1874, "Morgan County, Georgia"),
    ("P03", "P04", 1872, "Charleston, South Carolina"),
    ("P05", "P07", 1901, "Charleston, South Carolina"),
    ("P09", "P06", 1903, "Greene County, Georgia"),
    ("P08", "P10", 1904, "Minneapolis, Minnesota"),
    ("P11", "P16", 1930, "Knoxville, Tennessee"),
    ("P17", "P12", 1929, "Atlanta, Georgia"),
    ("P13", "P18", 1934, "Cincinnati, Ohio"),
    ("P15", "P25", 1932, "Chicago, Illinois"),
]

# --------------------------------------------------------------------------- #
# 2. Build the kinship graph (used both to template docs and to derive gold).
# --------------------------------------------------------------------------- #
def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for pid, attrs in PEOPLE.items():
        g.add_node(pid, **attrs)
    for child, (father, mother) in PARENTS.items():
        g.add_edge(child, father, rel="CHILD_OF")
        g.add_edge(child, mother, rel="CHILD_OF")
        g.add_edge(father, child, rel="PARENT_OF")
        g.add_edge(mother, child, rel="PARENT_OF")
    for a, b, *_ in MARRIAGES:
        g.add_edge(a, b, rel="SPOUSE_OF")
        g.add_edge(b, a, rel="SPOUSE_OF")
    return g


G = build_graph()


def name(pid: str) -> str:
    return PEOPLE[pid]["name"]


def father_of(pid: str) -> str | None:
    p = PARENTS.get(pid)
    return p[0] if p else None


def mother_of(pid: str) -> str | None:
    p = PARENTS.get(pid)
    return p[1] if p else None


def children_of(pid: str) -> list[str]:
    return [c for c, (f, m) in PARENTS.items() if pid in (f, m)]


# --------------------------------------------------------------------------- #
# 3. Citation templates -- realistic genealogical source citations.
# --------------------------------------------------------------------------- #
def cite_census(year: int, head: str, county_state: str, roll: int) -> str:
    ed = random.randint(5, 60)
    sheet = f"{random.randint(1, 30)}{random.choice('AB')}"
    dwelling = random.randint(10, 320)
    return (f"{year} U.S. Federal Census, {county_state}, population schedule, "
            f"ED {ed}, sheet {sheet}, dwelling {dwelling}, {head} household; "
            f"NARA microfilm publication T{600 + year % 100}, roll {roll:04d}.")


def cite_vital(kind: str, county_state: str, vol: int, page: int) -> str:
    series = {"birth": "Births", "death": "Deaths", "marriage": "Marriage Records"}[kind]
    return (f"{county_state.split(',')[1].strip()}, County {series}, "
            f"{county_state.split(',')[0].strip()}, vol. {vol}, p. {page}.")


def cite_obit(place: str, y: int, m: int, d: int, subject: str) -> str:
    papers = {"Georgia": "The Atlanta Constitution", "Tennessee": "Knoxville News-Sentinel",
              "Ohio": "The Cincinnati Enquirer", "Illinois": "Chicago Tribune",
              "Carolina": "The Charleston News and Courier"}
    paper = next((v for k, v in papers.items() if k in place), "The Atlanta Constitution")
    return f"{paper}, {y:04d}-{m:02d}-{d:02d}, obituary of {subject}."


def cite_dna(cm: int, segs: int, kit: str, hap: str) -> str:
    return (f"Autosomal DNA match report, shared {cm} cM across {segs} segments, "
            f"managed kit {kit}; Y-DNA Big Y-700 terminal haplogroup {hap}.")


# --------------------------------------------------------------------------- #
# 4. Generate documents. Each doc: id, type, title, text, persons[], citation,
#    facts{}. Person<->doc edges are recorded via `persons`.
# --------------------------------------------------------------------------- #
docs: list[dict] = []
_counter = 0


def add_doc(dtype: str, title: str, text: str, persons: list[str],
            citation: str, facts: dict) -> str:
    global _counter
    _counter += 1
    did = f"D{_counter:04d}"
    docs.append(dict(id=did, type=dtype, title=title, text=" ".join(text.split()),
                     persons=persons, citation=citation, facts=facts))
    return did


HAPS = ["R-M269 > R-FT12345", "I-M253 > I-Y6634", "R-U106 > R-Z18", "E-M35 > E-V13"]


def generate_documents() -> None:
    rolls = iter(range(101, 999))
    # 4a. Birth records (vital) -- name, date, place, parents.
    for pid, a in PEOPLE.items():
        f, m = father_of(pid), mother_of(pid)
        if f and m:
            par = f"son of {name(f)} and {name(m)}" if a["sex"] == "M" else \
                  f"daughter of {name(f)} and {name(m)}"
            ppl = [pid, f, m]
        else:
            par = "parents not recorded in this register"
            ppl = [pid]
        text = (f"Record of birth. {a['name']}, {a['sex']}, born in the year "
                f"{a['birth']} at {a['place']}, {par}.")
        add_doc("birth_record", f"Birth record of {a['name']}", text, ppl,
                cite_vital("birth", a["place"], random.randint(1, 9), random.randint(10, 400)),
                dict(person=pid, birth_year=a["birth"], birth_place=a["place"]))

    # 4b. Marriage records (vital) -- couple, year, place.
    for s1, s2, y, place in MARRIAGES:
        text = (f"Marriage record. {name(s1)} and {name(s2)} were joined in marriage "
                f"in the year {y} at {place}.")
        add_doc("marriage_record", f"Marriage record: {name(s1)} & {name(s2)}", text,
                [s1, s2], cite_vital("marriage", place, random.randint(2, 8),
                                     random.randint(100, 400)),
                dict(spouses=[s1, s2], marriage_year=y, marriage_place=place))

    # 4c. Census households -- couple + their then-minor (<18) children. Multi-person
    #     docs that create cross-links but never span three generations (clean hops).
    census_years = [1880, 1900, 1910, 1920, 1930, 1940]
    for s1, s2, my, place in MARRIAGES:
        head = name(s1) if PEOPLE[s1]["sex"] == "M" else name(s2)
        kids = sorted(set(children_of(s1)) & set(children_of(s2)))
        for cy in census_years:
            if cy < my:
                continue
            # both spouses alive and present
            if not (PEOPLE[s1]["birth"] <= cy <= PEOPLE[s1]["death"]
                    and PEOPLE[s2]["birth"] <= cy <= PEOPLE[s2]["death"]):
                continue
            minors = [k for k in kids if 0 <= cy - PEOPLE[k]["birth"] < 18]
            present = [s1, s2] + minors
            who = "; ".join(f"{name(p)}, age {cy - PEOPLE[p]['birth']}" for p in present)
            text = (f"In the {cy} census the household of {head} at {place} was "
                    f"enumerated as follows: {who}.")
            cs = PEOPLE[s1]["place"] if PEOPLE[s1]["sex"] == "M" else PEOPLE[s2]["place"]
            add_doc("census", f"{cy} census household of {head}", text, present,
                    cite_census(cy, head, cs, next(rolls)),
                    dict(year=cy, head=s1 if PEOPLE[s1]["sex"] == "M" else s2,
                         members=present))

    # 4d. Obituaries -- list surviving spouse + children only (no grandchildren).
    for pid, a in PEOPLE.items():
        if not a["death"]:
            continue
        spouse = next((s2 for s1, s2, *_ in MARRIAGES if s1 == pid), None) or \
                 next((s1 for s1, s2, *_ in MARRIAGES if s2 == pid), None)
        kids = children_of(pid)
        survivors = []
        if spouse and PEOPLE[spouse]["death"] > a["death"]:
            survivors.append(name(spouse))
        survivors += [name(k) for k in kids if PEOPLE[k]["death"] > a["death"]]
        surv = ("Survived by " + ", ".join(survivors) + "."
                if survivors else "No immediate survivors recorded.")
        text = (f"Obituary. {a['name']}, of {a['place']}, died in {a['death']} at the "
                f"age of {a['death'] - a['birth']}. {a['name']} worked as a "
                f"{a['occ']}. {surv}")
        ppl = [pid] + ([spouse] if spouse and spouse in survivors else []) \
            + [k for k in kids if name(k) in surv]
        mth, day = random.randint(1, 12), random.randint(1, 28)
        add_doc("obituary", f"Obituary of {a['name']}", text, list(dict.fromkeys(ppl)),
                cite_obit(a["place"], a["death"], mth, day, a["name"]),
                dict(person=pid, death_year=a["death"], occupation=a["occ"]))

    # 4e. Biographical sketches for a few notable people (occupation + place + era).
    for pid in ["P01", "P05", "P11", "P17", "P19", "P08"]:
        a = PEOPLE[pid]
        text = (f"Biographical sketch. {a['name']} ({a['birth']}-{a['death']}) was a "
                f"{a['occ']} associated with {a['place']}. Active in local civic life, "
                f"{a['name']} is documented across vital, census, and probate records "
                f"of the period.")
        add_doc("biography", f"Biographical sketch of {a['name']}", text, [pid],
                f"County historical society biographical files, {a['place']}, "
                f"folder {pid}.", dict(person=pid, occupation=a["occ"]))

    # 4f. DNA match notes -- connect living-descendant G4 pairs (the Big Y-700 flavor).
    g4_pairs = [("P19", "P21"), ("P22", "P24"), ("P20", "P22")]
    for i, (a, b) in enumerate(g4_pairs):
        cm = random.randint(40, 240)
        hap = HAPS[i % len(HAPS)]
        text = (f"DNA match note. Test-taker {name(a)} and match {name(b)} share "
                f"autosomal DNA consistent with a documented cousin relationship; "
                f"see kinship chart and shared ancestral lines.")
        add_doc("dna_match", f"DNA match: {name(a)} & {name(b)}", text, [a, b],
                cite_dna(cm, random.randint(1, 4), f"K{1000 + i}", hap),
                dict(test_takers=[a, b], shared_cm=cm, haplogroup=hap))


generate_documents()

# Build person -> docs index for gold-answer construction.
person_docs: dict[str, list[str]] = defaultdict(list)
for d in docs:
    for p in d["persons"]:
        person_docs[p].append(d["id"])


def identifying_docs(pid: str) -> list[str]:
    """Docs that identify a person on their own terms (birth + obituary + bio).

    Deliberately excludes census/marriage docs that co-mention other people, so
    that relational gold answers are reachable only by traversing the graph from
    the *named* person, not by lexical overlap with the question."""
    out = [d["id"] for d in docs
           if d["type"] in ("birth_record", "obituary", "biography")
           and d["facts"].get("person") == pid]
    return out


# --------------------------------------------------------------------------- #
# 5. Gold question set -- computed from the tree so answers are provably right.
# --------------------------------------------------------------------------- #
questions: list[dict] = []
_qc = 0


def add_q(cat: str, q: str, rel: list[str], ans_entity: str | None, ans_text: str) -> None:
    global _qc
    _qc += 1
    questions.append(dict(id=f"Q{_qc:03d}", category=cat, question=q,
                          relevant_doc_ids=sorted(set(rel)),
                          answer_entity=ans_entity, answer_text=ans_text))


def gen_questions() -> None:
    # lookup: birth year / birthplace / occupation (named entity -> lexical-friendly)
    for pid in ["P05", "P12", "P19", "P08", "P14", "P22", "P03", "P21"]:
        a = PEOPLE[pid]
        add_q("lookup", f"In what year was {a['name']} born?",
              [d["id"] for d in docs if d["facts"].get("person") == pid
               and d["type"] == "birth_record"], pid, str(a["birth"]))
        add_q("lookup", f"Where was {a['name']} born?",
              [d["id"] for d in docs if d["facts"].get("person") == pid
               and d["type"] in ("birth_record", "biography")], pid, a["place"])
    for pid in ["P01", "P11", "P17", "P13", "P06"]:
        a = PEOPLE[pid]
        add_q("lookup", f"What was the occupation of {a['name']}?",
              [d["id"] for d in docs if d["facts"].get("person") == pid
               and d["type"] in ("obituary", "biography")], pid, a["occ"])

    # provenance: which record documents the marriage of A and B?
    for s1, s2, y, _ in MARRIAGES[:6]:
        rel = [d["id"] for d in docs if d["type"] == "marriage_record"
               and set(d["facts"]["spouses"]) == {s1, s2}]
        add_q("provenance",
              f"Which source record documents the marriage of {name(s1)} and {name(s2)}?",
              rel, None, f"the {y} marriage record")

    # household: who were the children in a given census household?
    for s1, s2, _, _ in [MARRIAGES[2], MARRIAGES[0], MARRIAGES[7]]:
        head_pid = s1 if PEOPLE[s1]["sex"] == "M" else s2
        cdocs = [d for d in docs if d["type"] == "census"
                 and d["facts"]["head"] == head_pid]
        if not cdocs:
            continue
        d0 = max(cdocs, key=lambda d: len(d["facts"]["members"]))
        minors = [name(p) for p in d0["facts"]["members"][2:]]
        add_q("household",
              f"Which children were enumerated in the household of "
              f"{name(head_pid)} in the {d0['facts']['year']} census?",
              [d0["id"]], None, ", ".join(minors) or "no minor children recorded")

    # relational (multi-hop): grandparent / great-grandparent. The answer entity
    # is NOT named in the question -> only graph traversal surfaces the gold docs.
    g4 = ["P19", "P20", "P21", "P22", "P23", "P24"]
    for pid in g4:
        f, m = father_of(pid), mother_of(pid)
        # paternal grandfather
        if f and father_of(f):
            gpf = father_of(f)
            add_q("relational",
                  f"Who was the paternal grandfather of {name(pid)}?",
                  identifying_docs(gpf), gpf, name(gpf))
        # maternal grandfather
        if m and father_of(m):
            gmf = father_of(m)
            add_q("relational",
                  f"Who was the maternal grandfather of {name(pid)}?",
                  identifying_docs(gmf), gmf, name(gmf))
        # maternal grandmother
        if m and mother_of(m):
            gmm = mother_of(m)
            add_q("relational",
                  f"Who was the maternal grandmother of {name(pid)}?",
                  identifying_docs(gmm), gmm, name(gmm))


gen_questions()

# --------------------------------------------------------------------------- #
# 6. Write outputs.
# --------------------------------------------------------------------------- #
def write() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    with (DATA / "documents.jsonl").open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")

    relationships = []
    for child, (f, m) in PARENTS.items():
        for parent in (f, m):
            relationships.append(dict(src=child, dst=parent, rel="CHILD_OF"))
            relationships.append(dict(src=parent, dst=child, rel="PARENT_OF"))
    for a, b, y, place in MARRIAGES:
        relationships.append(dict(src=a, dst=b, rel="SPOUSE_OF", year=y, place=place))
        relationships.append(dict(src=b, dst=a, rel="SPOUSE_OF", year=y, place=place))
    graph = dict(
        persons=[dict(id=p, **a) for p, a in PEOPLE.items()],
        relationships=relationships,
        person_documents={p: sorted(ds) for p, ds in sorted(person_docs.items())},
    )
    (DATA / "graph.json").write_text(json.dumps(graph, indent=2))

    with (EVAL / "questions.jsonl").open("w") as fh:
        for q in questions:
            fh.write(json.dumps(q) + "\n")

    by_cat = defaultdict(int)
    for q in questions:
        by_cat[q["category"]] += 1
    print(f"persons          : {len(PEOPLE)}")
    print(f"relationships    : {len(relationships)}")
    print(f"documents        : {len(docs)}")
    dt = defaultdict(int)
    for d in docs:
        dt[d["type"]] += 1
    for k, v in sorted(dt.items()):
        print(f"  {k:16s}: {v}")
    print(f"gold questions   : {len(questions)}")
    for k, v in sorted(by_cat.items()):
        print(f"  {k:16s}: {v}")


if __name__ == "__main__":
    write()
