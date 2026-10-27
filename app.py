"""
app.py

Main file of the Streamlit web app. It loads the saved models, draws the
header and sidebar, and puts the four pages into tabs.

Run with:  streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from config import paths
from frontend.templates import hero_banner
from layout import inject_styles, render_sidebar
from views import email_checker, home, results_viewer, url_checker


# This must be the first Streamlit call on the page.
st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load frontend/styles.css once per page run.
inject_styles()


# ---------- cached loaders ----------
# Streamlit re-runs this whole file on every click. The cache keeps the models in
# memory so they are not read from disk again each time.

@st.cache_resource
def load_url_model():
    if not paths.URL_MODEL.exists():
        raise FileNotFoundError(f"URL model not found: {paths.URL_MODEL}")
    if not paths.URL_FEAT_COLS.exists():
        raise FileNotFoundError("URL feature columns file not found.")
    return joblib.load(paths.URL_MODEL), joblib.load(paths.URL_FEAT_COLS)


@st.cache_resource
def load_email_model():
    if not paths.EMAIL_MODEL.exists():
        raise FileNotFoundError(f"Email model not found: {paths.EMAIL_MODEL}")
    if not paths.EMAIL_VECTORIZER.exists():
        raise FileNotFoundError("Email TF-IDF vectorizer not found.")
    return joblib.load(paths.EMAIL_MODEL), joblib.load(paths.EMAIL_VECTORIZER)


@st.cache_data
def load_json(path):
    """Read a JSON file, or return an empty dict if it does not exist."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    """Read a CSV file, or return an empty table if it does not exist."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ---------- load everything ----------

# A missing model file should show a clear message instead of crashing the page.
load_error = None
try:
    url_model, url_feat_cols = load_url_model()
    email_model, email_vec = load_email_model()
except Exception as e:
    load_error = str(e)

url_meta = load_json(paths.URL_META)
email_meta = load_json(paths.EMAIL_META)
url_comp = load_csv(paths.URL_COMPARISON)
email_comp = load_csv(paths.EMAIL_COMPARISON)


# ---------- header and sidebar ----------

st.markdown(
    hero_banner(
        "Phishing Detection System",
        "URL and email phishing classification with plain-language guidance.",
    ),
    unsafe_allow_html=True,
)

render_sidebar(url_meta, email_meta)

if load_error:
    st.error("Error loading model artifacts.")
    st.code(load_error)
    st.stop()


# ---------- tabs ----------

tab_home, tab_url, tab_email, tab_results = st.tabs([
    "🏠 System Overview",
    "🌐 URL Threat Analyzer",
    "📧 Email Body Analyzer",
    "📊 Model Analytics & Benchmark",
])

with tab_home:
    home.render(url_meta, email_meta)

with tab_url:
    url_checker.render(url_model, url_feat_cols)

with tab_email:
    email_checker.render(email_model, email_vec)

with tab_results:
    results_viewer.render(url_comp, email_comp)


st.markdown(
    '<div class="app-footer">'
    "Academic Prototype | MSc Project CMP504 | Not a replacement for professional security tools"
    "</div>",
    unsafe_allow_html=True,
)