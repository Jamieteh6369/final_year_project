import nltk
import streamlit as st

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from article_tab import render_article_tab
from sentence_tab import render_sentence_tab
from sidebar import render_sidebar

st.set_page_config(
    page_title="TTP Classifier",
    page_icon="🛡️",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []

threshold, min_conf = render_sidebar()

st.title("🛡️ MITRE ATT&CK TTP Classifier")
st.caption("Two-stage threat intelligence classifier — maps text to MITRE ATT&CK Tactics and Techniques.")

tab_article, tab_sentence = st.tabs(["📰 Analyse Article", "🔍 Classify Sentence"])

with tab_article:
    render_article_tab(threshold, min_conf)

with tab_sentence:
    render_sentence_tab(threshold, min_conf)
