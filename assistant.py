"""Ask the Guide: a retrieval assistant that answers agent questions from an operations guide.

Run:  python assistant.py            evaluate, write the report and ROI
      python assistant.py "question" ask one question
"""
import hashlib
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
GUIDE = ROOT / "data" / "ops_guide.md"
QUESTIONS = ROOT / "data" / "questions.csv"
CONFIG = ROOT / "roi_config.json"
OUT = ROOT / "output"
SEED = 7
MAX_WRONG_RATE = 0.05  # at most 5% confident wrong answers allowed on the dev set

# ---------- loading and chunking ----------

def load_sections(path=GUIDE):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    sections = []
    for p in parts:
        head, _, body = p.partition("\n")
        sid, _, title = head.partition(" ")
        sections.append({"section": sid, "title": title.strip(), "text": body.strip()})
    return sections


def chunk(sections, strategy, size=40, overlap=10):
    """Two chunking strategies so they can be compared: whole sections or fixed word windows."""
    chunks = []
    for s in sections:
        if strategy == "section":
            chunks.append({"section": s["section"], "text": f'{s["title"]}. {s["text"]}'})
        else:
            words = s["text"].split()
            step = size - overlap
            for start in range(0, max(len(words) - overlap, 1), step):
                piece = " ".join(words[start:start + size])
                chunks.append({"section": s["section"], "text": f'{s["title"]}. {piece}'})
    return chunks


# ---------- guardrails ----------

PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?:\+44\s?|\b0)\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b")),
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I)),
]


def redact(text):
    """Remove personal data before a question is stored or sent to any model."""
    found = []
    for label, pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[{label}]", text)
    return text, found


# ---------- retrieval ----------

class Retriever:
    def __init__(self, sections, strategy="section"):
        self.sections = {s["section"]: s for s in sections}
        self.chunks = chunk(sections, strategy)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, question, k=3):
        sims = cosine_similarity(self.vec.transform([question.lower()]), self.matrix)[0]
        best = {}
        for c, score in zip(self.chunks, sims):
            best[c["section"]] = max(best.get(c["section"], 0.0), float(score))
        ranked = sorted(best.items(), key=lambda x: -x[1])
        return ranked[:k]


# ---------- context engineering ----------

SYSTEM_PROMPT = """You answer questions from customer service agents using ONLY the guide extracts provided.
Rules:
1. If the extracts do not contain the answer, reply exactly: NOT IN GUIDE. Ask your team leader.
2. Quote the section ID you used, for example [G03].
3. Keep it to three short sentences an agent can read while on a call.
4. Never invent amounts, timeframes or policies. Never give financial advice."""


def build_prompt(question, hits, sections, max_sections=2):
    context = "\n\n".join(
        f'[{sid}] {sections[sid]["title"]}\n{sections[sid]["text"]}' for sid, _ in hits[:max_sections]
    )
    return {"system": SYSTEM_PROMPT,
            "user": f"Guide extracts:\n{context}\n\nAgent question: {question}"}


def ask_claude(prompt):
    """Optional. Only runs when an API key and the anthropic package are available."""
    try:
        import anthropic
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    client = anthropic.Anthropic()
    msg = client.messages.create(model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5"),
                                 max_tokens=200, system=prompt["system"],
                                 messages=[{"role": "user", "content": prompt["user"]}])
    return msg.content[0].text


# ---------- evaluation ----------

def outcome(expected, top, score, threshold):
    answered = score >= threshold
    if expected == "NONE":
        return "correct_refusal" if not answered else "wrong_answer"
    if not answered:
        return "sent_to_team_leader"
    return "correct_answer" if top == expected else "wrong_answer"


def split(df):
    rng = random.Random(SEED)
    dev_ids = []
    for _, group in df.groupby(df["expected_section"] == "NONE"):
        ids = list(group["id"])
        rng.shuffle(ids)
        dev_ids += ids[: len(ids) // 2]
    return df.assign(split=np.where(df["id"].isin(dev_ids), "dev", "test"))


def score_all(df, retriever):
    rows = []
    for r in df.itertuples():
        clean, pii = redact(r.question)
        hits = retriever.search(clean)
        rows.append({"id": r.id, "split": r.split, "question": clean, "pii_removed": ",".join(pii),
                     "expected": r.expected_section, "top": hits[0][0], "score": round(hits[0][1], 3),
                     "in_top3": r.expected_section in [h[0] for h in hits]})
    return pd.DataFrame(rows)


def tune_threshold(dev):
    best = (0.0, -1)
    for t in np.round(np.arange(0.0, 0.61, 0.01), 2):
        res = [outcome(e, tp, s, t) for e, tp, s in zip(dev.expected, dev.top, dev.score)]
        wrong = res.count("wrong_answer") / len(res)
        right = res.count("correct_answer") + res.count("correct_refusal")
        if wrong <= MAX_WRONG_RATE and right > best[1]:
            best = (float(t), right)
    return best[0]


def summarise(df):
    counts = df["outcome"].value_counts().to_dict()
    inscope = df[df.expected != "NONE"]
    return {
        "questions": len(df),
        "correct_answer": counts.get("correct_answer", 0),
        "correct_refusal": counts.get("correct_refusal", 0),
        "sent_to_team_leader": counts.get("sent_to_team_leader", 0),
        "wrong_answer": counts.get("wrong_answer", 0),
        "top1_hit_rate_in_scope": round((inscope.top == inscope.expected).mean(), 3),
        "top3_hit_rate_in_scope": round(inscope.in_top3.mean(), 3),
    }


# ---------- ROI ----------

def roi(test_summary, cfg):
    inscope = test_summary["questions"] - cfg["_out_of_scope_in_test"]
    answered_share = test_summary["correct_answer"] / inscope
    wrong_share = test_summary["_wrong_in_scope"] / inscope
    q_per_year = cfg["agents"] * cfg["questions_per_agent_per_day"] * cfg["working_days"]
    handled = q_per_year * answered_share
    wrong_cost = q_per_year * wrong_share * cfg["minutes_to_fix_wrong_answer"] / 60 * cfg["team_leader_hourly_cost"]
    agent_saving = handled * cfg["agent_minutes_waiting"] / 60 * cfg["agent_hourly_cost"]
    lead_saving = handled * cfg["team_leader_minutes_per_question"] / 60 * cfg["team_leader_hourly_cost"]
    running = q_per_year * cfg["api_cost_per_question"] + cfg["upkeep_hours_per_month"] * 12 * cfg["team_leader_hourly_cost"]
    net = agent_saving + lead_saving - running - wrong_cost
    payback_weeks = cfg["build_cost"] / (net / 52) if net > 0 else None
    return {
        "questions_per_year": int(q_per_year),
        "answered_by_assistant_share": round(answered_share, 3),
        "questions_handled_per_year": int(handled),
        "agent_time_saved_hours": int(handled * cfg["agent_minutes_waiting"] / 60),
        "team_leader_time_saved_hours": int(handled * cfg["team_leader_minutes_per_question"] / 60),
        "gross_saving_gbp": int(agent_saving + lead_saving),
        "running_cost_gbp": int(running),
        "wrong_answer_share": round(wrong_share, 3),
        "cost_of_fixing_wrong_answers_gbp": int(wrong_cost),
        "net_saving_gbp": int(net),
        "build_cost_gbp": cfg["build_cost"],
        "payback_weeks": round(payback_weeks, 1) if payback_weeks else None,
    }


# ---------- reporting ----------

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    OUT.mkdir(exist_ok=True)
    sections = load_sections()
    smap = {s["section"]: s for s in sections}
    df = split(pd.read_csv(QUESTIONS))

    compare = {}
    for strategy in ["section", "window"]:
        scored = score_all(df, Retriever(sections, strategy))
        dev = scored[scored.split == "dev"]
        t = tune_threshold(dev)
        scored["outcome"] = [outcome(e, tp, s, t) for e, tp, s in zip(scored.expected, scored.top, scored.score)]
        compare[strategy] = {"threshold": t, "scored": scored,
                             "dev": summarise(scored[scored.split == "dev"]),
                             "test": summarise(scored[scored.split == "test"])}

    # Pick the strategy on dev results only, then report its untouched test results.
    chosen = max(compare, key=lambda k: (compare[k]["dev"]["correct_answer"] + compare[k]["dev"]["correct_refusal"],
                                         -compare[k]["dev"]["wrong_answer"]))
    final = compare[chosen]
    final["scored"].to_csv(OUT / "results.csv", index=False)

    cfg = json.loads(CONFIG.read_text())
    cfg["_out_of_scope_in_test"] = int((final["scored"].query("split == 'test'").expected == "NONE").sum())
    t = final["scored"].query("split == 'test' and expected != 'NONE'")
    final["test"]["_wrong_in_scope"] = int(((t.outcome == "wrong_answer")).sum())
    money = roi(final["test"], cfg)

    example_q = "customer wants 400 pounds back, card ending 4111 1111 1111 1111, can I approve it"
    clean, pii = redact(example_q)
    retr = Retriever(sections, chosen)
    prompt = build_prompt(clean, retr.search(clean), smap)
    (OUT / "sample_prompt.md").write_text(
        f"Original question: {example_q}\n\nPII removed: {', '.join(pii)}\n\n"
        f"## System prompt\n```\n{prompt['system']}\n```\n\n## User message\n```\n{prompt['user']}\n```\n",
        encoding="utf-8")

    manifest = {
        "data_hashes": {"ops_guide.md": sha(GUIDE), "questions.csv": sha(QUESTIONS), "roi_config.json": sha(CONFIG)},
        "split": {"seed": SEED, "dev": int((df.split == "dev").sum()), "test": int((df.split == "test").sum())},
        "chosen_strategy": chosen, "threshold": final["threshold"], "max_wrong_rate_dev": MAX_WRONG_RATE,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    low = dict(cfg, questions_per_agent_per_day=1)
    money_low = roi(final["test"], low)
    (OUT / "roi.json").write_text(json.dumps({"base": money, "if_1_question_per_agent_per_day": money_low}, indent=2))

    test = final["scored"][final["scored"].split == "test"]
    misses = test[test.outcome.isin(["wrong_answer", "sent_to_team_leader"])]
    lines = ["# Ask the Guide: evaluation report", "",
             f"Chunking strategy chosen on dev questions: **{chosen}** (threshold {final['threshold']}).", "",
             "| Strategy | Split | Correct answers | Correct refusals | Sent to team leader | Wrong answers | Top-1 hit | Top-3 hit |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in compare.items():
        for sp in ["dev", "test"]:
            s = v[sp]
            lines.append(f"| {k} | {sp} | {s['correct_answer']} | {s['correct_refusal']} | {s['sent_to_team_leader']} | "
                         f"{s['wrong_answer']} | {s['top1_hit_rate_in_scope']:.0%} | {s['top3_hit_rate_in_scope']:.0%} |")
    lines += ["", "## Test questions the assistant did not answer correctly", ""]
    for r in misses.itertuples():
        lines.append(f"- {r.id} ({r.outcome}): \"{r.question}\" expected {r.expected}, top match {r.top} at {r.score}")
    lines += ["", "## ROI (assumptions in roi_config.json)", ""]
    lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in money.items()]
    lines += ["", f"Sensitivity: with only 1 question per agent per day, net saving is £{money_low['net_saving_gbp']:,} "
              f"and payback is {money_low['payback_weeks']} weeks."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def ask(question):
    sections = load_sections()
    smap = {s["section"]: s for s in sections}
    manifest = json.loads((OUT / "manifest.json").read_text())
    retr = Retriever(sections, manifest["chosen_strategy"])
    clean, pii = redact(question)
    hits = retr.search(clean)
    if pii:
        print(f"Removed personal data: {', '.join(pii)}")
    if hits[0][1] < manifest["threshold"]:
        print("NOT IN GUIDE. Ask your team leader.")
        return
    answer = ask_claude(build_prompt(clean, hits, smap))
    sid = hits[0][0]
    print(answer or f'[{sid}] {smap[sid]["title"]}\n{smap[sid]["text"]}')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        main()
