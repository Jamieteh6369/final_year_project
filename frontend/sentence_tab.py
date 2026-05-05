import requests
import streamlit as st

from config import API_URL, SEVERITY_COLOUR
from result_card import _save_to_history


def render_sentence_tab(threshold: float, min_conf: float) -> None:
    sentence = st.text_area(
        "Enter a threat description",
        placeholder="The attacker dumped LSASS memory to extract plaintext passwords.",
        height=120,
    )

    if not st.button("Classify", type="primary", key="btn_sentence"):
        return

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
