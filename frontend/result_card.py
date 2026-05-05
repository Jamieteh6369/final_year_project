from datetime import datetime

import streamlit as st

from config import SEVERITY_COLOUR


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
