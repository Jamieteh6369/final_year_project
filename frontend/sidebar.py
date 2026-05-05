import requests
import streamlit as st

from config import SEVERITY_COLOUR


def render_sidebar() -> tuple[float, float]:
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

    return threshold, min_conf
