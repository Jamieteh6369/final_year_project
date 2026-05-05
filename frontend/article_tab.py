import requests
import streamlit as st
from nltk.tokenize import sent_tokenize

from article_fetcher import fetch_article_text, get_article_url
from config import BATCH_API_URL
from result_card import _partition, _render_result, _save_to_history


def render_article_tab(threshold: float, min_conf: float) -> None:
    url = st.text_input("Article URL", placeholder="https://blog.talosintelligence.com/…")

    if not st.button("Analyse Article", type="primary", key="btn_article"):
        return

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

    results    = []
    chunk_size = 20
    chunks     = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    progress   = st.progress(0, text="Classifying…")
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
    m1.metric("Total Sentences",  len(results))
    m2.metric("✅ Confirmed TTP", len(passed))
    m3.metric("⚠️ Flagged",       len(flagged1) + len(flagged2))
    m4.metric("❌ Errors",         len(errors))
    m5.metric("🔇 Hidden (noise)", noise)
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
