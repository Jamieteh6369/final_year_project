from datetime import datetime

import requests
import streamlit as st
import trafilatura
import nltk
from bs4 import BeautifulSoup

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Article fetching (no backend dependency) ──────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_article_url(hn_url: str) -> str:
    resp = requests.get(hn_url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title_span = soup.select_one("span.titleline > a")
    if not title_span:
        raise ValueError("Could not find an article link on that HackerNews page.")
    article_url = title_span["href"]
    if article_url.startswith("item?"):
        raise ValueError("This looks like a self-post (Ask HN) — no external article to fetch.")
    return article_url


def _bs4_extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(p for p in paragraphs if len(p) > 40)


def _is_cloudflare_block(html: str) -> bool:
    markers = ["Just a moment", "cf-browser-verification", "Enable JavaScript", "Checking your browser"]
    return any(m in html for m in markers)


def fetch_article_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            downloaded = resp.text
        except requests.RequestException as e:
            raise ValueError(f"Could not download page: {url} — {e}")
    if _is_cloudflare_block(downloaded):
        raise ValueError(
            f"'{url}' is protected by Cloudflare bot detection. Try a different article."
        )
    text = (
        trafilatura.extract(downloaded, include_comments=False, no_fallback=False)
        or trafilatura.extract(downloaded, favour_recall=True)
        or _bs4_extract(downloaded)
    )
    if not text:
        raise ValueError(f"Could not extract article text from: {url}")
    return text

# ── Config ────────────────────────────────────────────────────────────────────

API_URL       = "http://localhost:8000/predict"
BATCH_API_URL = "http://localhost:8000/predict/batch"

SEVERITY_COLOUR = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
    "Unknown":  "⚪",
}

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TTP Classifier",
    page_icon="🛡️",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of analysis entries

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        health = requests.get("http://localhost:8000/health", timeout=3).json()
        if health.get("models_loaded"):
            st.success("✅ Backend online")
        else:
            st.warning("⏳ Models loading…")
    except Exception:
        st.error("❌ Backend offline\n\nRun: `uvicorn app:app --reload`")

    with st.expander("⚙️ Settings", expanded=False):
        threshold = st.slider("Confidence threshold", 0.50, 0.95, 0.80, 0.05)
        min_conf  = st.slider("Hide noise below",      0.10, 0.50, 0.20, 0.05)

    st.divider()
    st.markdown("### 🕘 Analysis History")
    if not st.session_state.history:
        st.caption("No analyses yet.")
    else:
        if st.button("🗑️ Clear history", key="btn_clear"):
            st.session_state.history = []
            st.rerun()
        for entry in st.session_state.history:
            s   = entry["summary"]
            tag = "📰" if entry["type"] == "article" else "🔍"
            label = entry["source"] if len(entry["source"]) <= 40 else entry["source"][:37] + "…"
            with st.expander(f"{tag} {entry['timestamp']} — {label}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Confirmed", s["passed"])
                c2.metric("Flagged",   s["flagged"])
                c3.metric("Total",     s["total"])
                st.divider()
                passed_results = [r for r in entry["results"] if r.get("status") == "passed"]
                if passed_results:
                    st.markdown("**✅ Confirmed**")
                    for r in passed_results:
                        icon = SEVERITY_COLOUR.get(r.get("severity", "Unknown"), "⚪")
                        tech = r.get("technique_name") or r.get("technique") or "N/A"
                        st.markdown(
                            f"- {icon} **{r['tactic']}** → {tech}  \n"
                            f"  <small>{r['sentence'][:70]}{'…' if len(r['sentence']) > 70 else ''}</small>",
                            unsafe_allow_html=True,
                        )
                flagged_results = [
                    r for r in entry["results"]
                    if r.get("status") in ("flagged_stage1", "flagged_stage2")
                    and r.get("tactic_confidence", 0) >= min_conf
                ]
                if flagged_results:
                    st.markdown("**⚠️ Flagged**")
                    for r in flagged_results:
                        icon = SEVERITY_COLOUR.get(r.get("severity", "Unknown"), "⚪")
                        st.markdown(
                            f"- {icon} **{r['tactic']}** ({r['tactic_confidence']:.0%})  \n"
                            f"  <small>{r['sentence'][:70]}{'…' if len(r['sentence']) > 70 else ''}</small>",
                            unsafe_allow_html=True,
                        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_result(r: dict) -> None:
    sev  = r.get("severity", "Unknown")
    icon = SEVERITY_COLOUR.get(sev, "⚪")
    tech_label = (
        f"{r.get('technique')} — {r.get('technique_name', '')}"
        if r.get("technique") else "N/A"
    )
    with st.expander(f"{icon} **[{sev}]** {r['tactic']} → {tech_label}"):
        st.markdown(f"**Sentence:** {r['sentence']}")
        st.divider()
        cl, cr = st.columns(2)
        with cl:
            st.markdown(f"**Tactic:** {r['tactic']}")
            st.caption(r.get("tactic_definition", ""))
        with cr:
            if r.get("technique"):
                st.markdown(f"**Technique:** {r.get('technique')} — {r.get('technique_name', '')}")
                st.caption(r.get("technique_definition", ""))
        st.divider()
        c1, c2, c3 = st.columns(3)
        overall = r.get("overall_confidence") or r.get("tactic_confidence", 0)
        c1.metric("Overall Confidence", f"{overall:.0%}")
        c2.metric("Stage 1", f"{r['tactic_confidence']:.0%}")
        c3.metric("Stage 2", f"{r.get('technique_confidence', 0):.0%}" if r.get("technique_confidence") else "—")
        if r.get("reference_link"):
            st.markdown(f"🔗 [View on MITRE ATT&CK]({r['reference_link']})")


def _partition(results: list[dict], noise_threshold: float):
    passed   = [r for r in results if r.get("status") == "passed"]
    flagged1 = [r for r in results if r.get("status") == "flagged_stage1"
                and r.get("tactic_confidence", 0) >= noise_threshold]
    flagged2 = [r for r in results if r.get("status") == "flagged_stage2"
                and r.get("tactic_confidence", 0) >= noise_threshold]
    errors   = [r for r in results if "error" in r]
    noise    = len(results) - len(passed) - len(flagged1) - len(flagged2) - len(errors)
    return passed, flagged1, flagged2, errors, noise


def _save_to_history(entry_type: str, source: str, results: list[dict], noise_threshold: float) -> None:
    passed, flagged1, flagged2, errors, noise = _partition(results, noise_threshold)
    st.session_state.history.insert(0, {
        "type":      entry_type,
        "source":    source,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "results":   results,
        "summary": {
            "total":   len(results),
            "passed":  len(passed),
            "flagged": len(flagged1) + len(flagged2),
            "noise":   noise,
            "errors":  len(errors),
        },
    })


# ── Main content ─────────────────────────────────────────────────────────────

st.title("🛡️ MITRE ATT&CK TTP Classifier")
st.caption("Two-stage threat intelligence classifier — maps text to MITRE ATT&CK Tactics and Techniques.")

tab_article, tab_sentence = st.tabs(["📰 Analyse Article", "🔍 Classify Sentence"])

# ── Tab 1: Article URL ───────────────────────────────────────────────────────

with tab_article:
    url = st.text_input("Article URL", placeholder="https://blog.talosintelligence.com/…")

    if st.button("Analyse Article", type="primary", key="btn_article"):
        if not url.strip():
            st.warning("Please enter a URL.")
            st.stop()

        with st.spinner("Fetching article…"):
            try:
                if "news.ycombinator.com" in url:
                    article_url = get_article_url(url)
                    st.info(f"Article URL resolved: {article_url}")
                else:
                    article_url = url
                raw_text = fetch_article_text(article_url)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        sentences = [s.strip() for s in sent_tokenize(raw_text) if len(s.strip()) > 40]
        st.success(f"Extracted **{len(sentences)}** sentences.")

        results  = []
        chunk_size = 20
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        progress = st.progress(0, text="Classifying…")
        for i, chunk in enumerate(chunks):
            try:
                resp = requests.post(
                    BATCH_API_URL,
                    json={"sentences": chunk, "threshold": threshold},
                    timeout=120,
                )
                resp.raise_for_status()
                results.extend(resp.json())
            except Exception as e:
                for s in chunk:
                    results.append({"sentence": s, "error": str(e)})
            done = min((i + 1) * chunk_size, len(sentences))
            progress.progress(done / len(sentences), text=f"Classifying sentence {done}/{len(sentences)}…")
        progress.empty()

        _save_to_history("article", article_url, results, min_conf)

        passed, flagged1, flagged2, errors, noise = _partition(results, min_conf)

        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Sentences",   len(results))
        m2.metric("✅ Confirmed TTP",  len(passed))
        m3.metric("⚠️ Flagged",        len(flagged1) + len(flagged2))
        m4.metric("❌ Errors",          len(errors))
        m5.metric("🔇 Hidden (noise)",  noise)
        st.divider()

        if passed:
            st.subheader(f"✅ Confirmed TTP Mappings — {len(passed)}")
            for r in passed:
                _render_result(r)

        if flagged1 or flagged2:
            st.subheader(f"⚠️ Flagged for Review — {len(flagged1) + len(flagged2)}")
            for r in flagged1 + flagged2:
                _render_result(r)

        if errors:
            st.subheader(f"❌ Errors — {len(errors)}")
            for r in errors:
                st.error(f"{r['error']}\n\n*Sentence:* {r['sentence'][:120]}")

# ── Tab 2: Single Sentence ────────────────────────────────────────────────────

with tab_sentence:
    sentence = st.text_area(
        "Enter a threat description",
        placeholder="The attacker dumped LSASS memory to extract plaintext passwords.",
        height=120,
    )

    if st.button("Classify", type="primary", key="btn_sentence"):
        if not sentence.strip():
            st.warning("Please enter a sentence.")
            st.stop()

        with st.spinner("Classifying…"):
            try:
                resp = requests.post(
                    API_URL,
                    json={"sentence": sentence, "threshold": threshold},
                    timeout=30,
                )
                resp.raise_for_status()
                r = resp.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        _save_to_history("sentence", sentence[:80], [r], min_conf)

        status = r.get("status", "")
        sev    = r.get("severity", "Unknown")
        icon   = SEVERITY_COLOUR.get(sev, "⚪")

        if status == "passed":
            st.success("✅ PASSED — Confident TTP mapping found")
        elif status == "flagged_stage1":
            st.warning("⚠️ FLAGGED — Stage 1 confidence below threshold")
        else:
            st.warning("⚠️ FLAGGED — Stage 2 confidence below threshold")

        st.divider()
        cl, cr = st.columns(2)
        with cl:
            st.markdown("### Tactic")
            st.markdown(f"**{r['tactic']}**")
            st.caption(r.get("tactic_definition", ""))
            st.metric("Stage 1 Confidence", f"{r['tactic_confidence']:.0%}")
        with cr:
            st.markdown("### Technique")
            if r.get("technique"):
                st.markdown(f"**{r.get('technique')} — {r.get('technique_name', '')}**")
                st.caption(r.get("technique_definition", ""))
                st.metric("Stage 2 Confidence", f"{r.get('technique_confidence', 0):.0%}")
            else:
                st.info("Not reached — flagged at Stage 1.")

        st.divider()
        s1, s2, s3 = st.columns(3)
        overall = r.get("overall_confidence") or r.get("tactic_confidence", 0)
        s1.metric("Overall Confidence", f"{overall:.0%}")
        s2.metric("Severity", f"{icon} {sev}")
        s3.metric("Status", status.replace("_", " ").title())

        if r.get("reference_link"):
            st.markdown(f"🔗 [View on MITRE ATT&CK]({r['reference_link']})")
