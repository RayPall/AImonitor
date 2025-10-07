"""
AIVM — AI Visibility Monitor (Streamlit MVP)

Run:
  1) pip install -r requirements.txt
  2) streamlit run streamlit_app.py

Secrets/keys:
  - Set env vars OPENAI_API_KEY and (optionally) GOOGLE_API_KEY
  - Or paste keys into the Streamlit sidebar (stored only in session state)

This MVP supports:
  - Loading scenarios from a JSON file (or using a tiny built‑in sample)
  - Calling OpenAI (gpt-5 by default) with n samples per query
  - Optional Gemini support (if GOOGLE_API_KEY present)
  - Judge/Parser pass to normalize mentions & recommendations (LLM-as-a-judge)
  - SQLite persistence (aivm.db) for raw & parsed results
  - Simple KPIs (Visibility Score, Recommendation Index) and tables

Note: Focused on a simple, auditable baseline. Extend as needed.
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# --- Optional: OpenAI + Gemini clients (import guarded) ---
try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import google.generativeai as genai  # google-generativeai
except Exception:  # pragma: no cover
    genai = None  # type: ignore

DB_PATH_DEFAULT = "aivm.db"
RAW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aivm_raw (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  assistant TEXT NOT NULL,
  model TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  query TEXT NOT NULL,
  language TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  run_meta TEXT NOT NULL,
  created_at TEXT DEFAULT (DATETIME('now'))
);
"""
PARSED_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS aivm_parsed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  raw_id INTEGER REFERENCES aivm_raw(id) ON DELETE CASCADE,
  mentions TEXT NOT NULL,
  competitors TEXT NOT NULL,
  facts_flags TEXT NOT NULL,
  created_at TEXT DEFAULT (DATETIME('now'))
);
"""

BUILTIN_SCENARIOS = [
    {
        "id": "S01_erp_smb_cz",
        "persona": "smb_owner",
        "language": "cs",
        "variants": [
            "Jaký ERP systém doporučíte pro malou až střední firmu v Česku?",
            "Co je dobré ERP pro SMB v ČR?",
            "Který ERP byste zvolili pro malou až střední firmu v ČR?",
        ],
        "topic": "erp_general",
    },
    {
        "id": "S02_accounting_vat_cz",
        "persona": "finance_manager",
        "language": "cs",
        "variants": [
            "Jaký účetní software v češtině se hodí pro plátce DPH (CZ)?",
            "Nejlepší účetní software s podporou české DPH?",
            "Doporučení účetního softwaru s CZ DPH?",
        ],
        "topic": "accounting_vat",
    },
]

PARSER_SCHEMA_HINT = {
    "assistant": "openai|gemini|copilot",
    "model": "string",
    "scenario_id": "string",
    "query": "string",
    "language": "cs|sk|en|other",
    "raw_text": "string",
    "mentions": [
        {
            "vendor": "string",
            "product": "string|null",
            "stance": "positive|neutral|negative",
            "recommendation": "explicit|implicit|none",
            "evidence": "string"
        }
    ],
    "competitors": [
        {
            "vendor": "string",
            "product": "string|null",
            "stance": "positive|neutral|negative",
            "recommendation": "explicit|implicit|none",
            "evidence": "string"
        }
    ],
    "facts_flags": {"possible_inaccuracy": True, "notes": "string"},
    "run_meta": {"timestamp": "ISO-8601", "sample_ix": 0}
}

SYSTEM_PROMPT = (
    "Jsi nezávislý odborný konzultant. Odpovídej stručně a bez brand biasu. "
    "Neodkazuj na interní zdroje tazatele. Pokud si nejsi jistý, napiš 'nejsem si jistý'. "
    "Mluv jazykem dotazu."
)

PARSER_PROMPT = f"""
Úkol: Z této odpovědi vytěž strukturovaná data o zmínkách značek/produktů a síle doporučení.
Vrať POUZE validní JSON (nic okolo) podle schématu níže. Pokud něco chybí, vyplň prázdné pole nebo []
(přesné klíče, typy a hodnoty enumů):

{json.dumps(PARSER_SCHEMA_HINT, ensure_ascii=False)}
"""

# -------------------------- Helpers --------------------------

def get_db(conn_path: str):
    conn = sqlite3.connect(conn_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(RAW_TABLE_SQL)
    conn.execute(PARSED_TABLE_SQL)
    return conn


def insert_raw(conn, row: Dict[str, Any]) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO aivm_raw(assistant, model, scenario_id, query, language, raw_text, run_meta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["assistant"],
            row["model"],
            row["scenario_id"],
            row["query"],
            row["language"],
            row["raw_text"],
            json.dumps(row.get("run_meta", {}), ensure_ascii=False),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_parsed(conn, raw_id: int, parsed: Dict[str, Any]):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO aivm_parsed(raw_id, mentions, competitors, facts_flags)
        VALUES (?, ?, ?, ?)
        """,
        (
            raw_id,
            json.dumps(parsed.get("mentions", []), ensure_ascii=False),
            json.dumps(parsed.get("competitors", []), ensure_ascii=False),
            json.dumps(parsed.get("facts_flags", {}), ensure_ascii=False),
        ),
    )
    conn.commit()


# --- Model calls ---

def call_openai_answers(api_key: str, model: str, query: str, n: int = 3, temperature: float = 0.2) -> List[str]:
    if OpenAI is None:
        raise RuntimeError("openai package not available. pip install openai>=1.0.0")
    client = OpenAI(api_key=api_key)
    # We issue n calls to get independent samples (more robust vs. single call with n)
    answers: List[str] = []
    for _ in range(n):
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        answers.append(resp.choices[0].message.content or "")
    return answers


def call_gemini_answers(api_key: str, model: str, query: str, n: int = 3, temperature: float = 0.2) -> List[str]:
    if genai is None:
        raise RuntimeError("google-generativeai not available. pip install google-generativeai")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    answers: List[str] = []
    for _ in range(n):
        out = m.generate_content(query, generation_config={"temperature": temperature})
        answers.append(out.text or "")
    return answers


def judge_parse_with_openai(api_key: str, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not available. pip install openai>=1.0.0")
    client = OpenAI(api_key=api_key)
    parser_user = (
        f"{PARSER_PROMPT}\n\nNyní parsuj tuto odpověď (včetně metadat):\n<<<\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n>>>"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Jsi přísný JSON extraktor. Vrať pouze JSON bez komentářů."},
            {"role": "user", "content": parser_user},
        ],
    )
    txt = resp.choices[0].message.content or "{}"
    # Try to locate JSON block if model added extra text.
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        txt = txt[start : end + 1]
    try:
        return json.loads(txt)
    except Exception:
        return {"mentions": [], "competitors": [], "facts_flags": {"possible_inaccuracy": True, "notes": "Parser failed"}}


# --- KPI helpers ---

TARGET_PRODUCTS = {"Byznys", "HELIOS"}
TARGET_VENDOR = "Seyfor"


def score_recommendation(rec: str) -> float:
    if rec == "explicit":
        return 2.0
    if rec == "implicit":
        return 1.0
    return 0.0


def compute_kpis(conn) -> Dict[str, Any]:
    cur = conn.cursor()
    # Fetch joined rows
    cur.execute(
        """
        SELECT r.id, r.assistant, r.model, r.scenario_id, r.language, r.raw_text,
               p.mentions
        FROM aivm_raw r
        JOIN aivm_parsed p ON p.raw_id = r.id
        ORDER BY r.id DESC
        """
    )
    rows = cur.fetchall()
    total_samples = len(rows)
    if total_samples == 0:
        return {"total_samples": 0}

    vs_hits = 0
    ri_scores: List[float] = []

    for (_id, assistant, model, scenario_id, language, raw_text, mentions_json) in rows:
        try:
            mentions = json.loads(mentions_json or "[]")
        except Exception:
            mentions = []
        hit_this_sample = False
        ri_this_sample = 0.0
        for m in mentions:
            vendor = (m.get("vendor") or "").strip()
            product = (m.get("product") or "").strip()
            rec = (m.get("recommendation") or "none").strip()
            if vendor == TARGET_VENDOR or (product and product in TARGET_PRODUCTS):
                hit_this_sample = True
                ri_this_sample = max(ri_this_sample, score_recommendation(rec))
        if hit_this_sample:
            vs_hits += 1
        ri_scores.append(ri_this_sample)

    vs = vs_hits / total_samples if total_samples else 0.0
    ri = sum(ri_scores) / total_samples if total_samples else 0.0
    return {"total_samples": total_samples, "visibility_score": vs, "recommendation_index": ri}


# -------------------------- UI --------------------------

st.set_page_config(page_title="AIVM — Streamlit MVP", layout="wide")
st.title("AIVM — AI Visibility Monitor (Streamlit MVP)")

with st.sidebar:
    st.header("Configuration")
    db_path = st.text_input("SQLite DB path", value=DB_PATH_DEFAULT)
    openai_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    use_openai = st.checkbox("Use OpenAI", value=True)
    openai_model = st.selectbox("OpenAI model", ["gpt-5"], index=0)

    gemini_key = st.text_input("GOOGLE_API_KEY (optional)", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    use_gemini = st.checkbox("Use Gemini (optional)", value=False, help="Requires google-generativeai package and API key")
    gemini_model = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

    n_samples = st.slider("Samples per query", 1, 5, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    st.caption("Scenarios JSON: one array with objects {id, language, variants[]}.")
    scenarios_file = st.file_uploader("Upload scenarios.json (optional)", type=["json"]) 

    st.divider()
    run_btn = st.button("▶ Run measurement", use_container_width=True)

# DB connection
conn = get_db(db_path)

# Load scenarios
scenarios: List[Dict[str, Any]] = []
if scenarios_file is not None:
    scenarios = json.load(scenarios_file)
else:
    scenarios = BUILTIN_SCENARIOS

# Show scenarios preview
with st.expander("Scenarios loaded (preview)", expanded=False):
    st.json(scenarios[:5])

# Run workflow
if run_btn:
    if not use_openai and not use_gemini:
        st.error("Select at least one provider (OpenAI or Gemini).")
        st.stop()

    progress = st.progress(0.0)
    status = st.empty()

    total_tasks = sum(len(s.get("variants", [])) for s in scenarios)
    done = 0

    for s in scenarios:
        scenario_id = s.get("id", "unknown")
        language = s.get("language", "other")
        for v_ix, query in enumerate(s.get("variants", [])):
            status.info(f"{scenario_id} — variant {v_ix+1}/{len(s['variants'])}")

            # --- OpenAI ---
            if use_openai:
                if not openai_key:
                    st.warning("Missing OPENAI_API_KEY; skipping OpenAI.")
                else:
                    try:
                        answers = call_openai_answers(openai_key, openai_model, query, n=n_samples, temperature=temperature)
                    except Exception as e:
                        st.error(f"OpenAI call failed: {e}")
                        answers = []
                    for i, txt in enumerate(answers):
                        raw = {
                            "assistant": "openai",
                            "model": openai_model,
                            "scenario_id": scenario_id,
                            "query": query,
                            "language": language,
                            "raw_text": txt,
                            "run_meta": {
                                "timestamp": datetime.utcnow().isoformat(),
                                "sample_ix": i,
                            },
                        }
                        raw_id = insert_raw(conn, raw)
                        # Judge/parse
                        payload = raw | {"assistant": "openai"}
                        try:
                            parsed = judge_parse_with_openai(openai_key, openai_model, payload)
                        except Exception as e:
                            parsed = {"mentions": [], "competitors": [], "facts_flags": {"possible_inaccuracy": True, "notes": str(e)}}
                        insert_parsed(conn, raw_id, parsed)

            # --- Gemini (optional) ---
            if use_gemini:
                if not gemini_key:
                    st.warning("Missing GOOGLE_API_KEY; skipping Gemini.")
                else:
                    try:
                        answers = call_gemini_answers(gemini_key, gemini_model, query, n=n_samples, temperature=temperature)
                    except Exception as e:
                        st.error(f"Gemini call failed: {e}")
                        answers = []
                    for i, txt in enumerate(answers):
                        raw = {
                            "assistant": "gemini",
                            "model": gemini_model,
                            "scenario_id": scenario_id,
                            "query": query,
                            "language": language,
                            "raw_text": txt,
                            "run_meta": {
                                "timestamp": datetime.utcnow().isoformat(),
                                "sample_ix": i,
                            },
                        }
                        raw_id = insert_raw(conn, raw)
                        # Judge/parse via OpenAI parser (more deterministic)
                        if openai_key:
                            try:
                                payload = raw | {"assistant": "gemini"}
                                parsed = judge_parse_with_openai(openai_key, openai_model, payload)
                            except Exception as e:
                                parsed = {"mentions": [], "competitors": [], "facts_flags": {"possible_inaccuracy": True, "notes": str(e)}}
                        else:
                            # Fallback: naive parser if OpenAI key not available
                            parsed = {"mentions": [], "competitors": [], "facts_flags": {"possible_inaccuracy": True, "notes": "No OpenAI key for parser"}}
                        insert_parsed(conn, raw_id, parsed)

            done += 1
            progress.progress(min(1.0, done / max(1, total_tasks)))
            time.sleep(0.05)

    status.success("Run completed.")

# KPIs & Tables
st.subheader("KPIs")
kpis = compute_kpis(conn)
if kpis.get("total_samples", 0) == 0:
    st.info("No data yet. Run a measurement.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", kpis["total_samples"])
    col2.metric("Visibility Score (Seyfor/Byznys/HELIOS)", f"{kpis['visibility_score']:.2f}")
    col3.metric("Recommendation Index", f"{kpis['recommendation_index']:.2f}")

    st.divider()
    st.subheader("Latest samples (joined)")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.id, r.created_at, r.assistant, r.model, r.scenario_id, r.language, r.query, r.raw_text,
               p.mentions, p.competitors, p.facts_flags
        FROM aivm_raw r
        JOIN aivm_parsed p ON p.raw_id = r.id
        ORDER BY r.id DESC
        LIMIT 50
        """
    )
    rows = cur.fetchall()
    # Render as expandable items for readability
    for (rid, created, assistant, model, sid, lang, query, raw_text, mentions, competitors, flags) in rows:
        with st.expander(f"#{rid} · {created} · {assistant}/{model} · {sid} [{lang}] — {query[:60]}…"):
            st.write("**Answer:**")
            st.write(raw_text)
            st.write("**Mentions:**")
            st.json(json.loads(mentions or "[]"))
            st.write("**Competitors:**")
            st.json(json.loads(competitors or "[]"))
            st.write("**Flags:**")
            st.json(json.loads(flags or "{}"))

st.caption("MVP © AIVM · This tool queries public LLMs with generic prompts and stores outputs locally.")
