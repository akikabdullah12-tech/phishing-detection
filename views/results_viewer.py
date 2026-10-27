"""
results_viewer.py

The "Model Analytics & Benchmark" tab. It shows the model comparison tables
and the charts that the two training scripts saved in the results folder.
"""

import numpy as np
import streamlit as st

from config import paths


def show_table(comparison, missing_message):
    """Show a comparison table, rounded to 4 decimals, or a note if the file is missing."""
    if comparison.empty:
        st.info(missing_message)
        return
    df = comparison.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].round(4)
    st.dataframe(df, width="stretch", hide_index=True)


def show_image(path, caption):
    """Show one chart if its file exists. A missing file is simply skipped."""
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def render(url_comp, email_comp):
    """Draw the analytics tab with one sub-tab per model family."""
    st.subheader("📊 Empirical Model Benchmark Analytics")

    url_tab, email_tab = st.tabs(["🌐 URL Model Suite (5 Models)", "📧 Email Model Suite (5 Models)"])

    with url_tab:
        st.markdown("#### URL Classifier Benchmark (held-out test set)")
        show_table(url_comp, "URL model comparison metrics unavailable.")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show_image(paths.URL_CONF_MATRIX, "Confusion matrix of the best URL model")
        with c2:
            show_image(paths.URL_FEAT_IMP, "Random Forest feature importance")

        c3, c4 = st.columns(2)
        with c3:
            show_image(paths.URL_ROC, "ROC curves of all 5 URL classifiers")
        with c4:
            show_image(paths.URL_PR, "Precision-recall curves of all 5 URL classifiers")

    with email_tab:
        st.markdown("#### Email Classifier Benchmark (held-out test set)")
        show_table(email_comp, "Email model comparison metrics unavailable.")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show_image(paths.EMAIL_CONF_MATRIX, "Confusion matrix of the best email model")
        with c2:
            show_image(paths.EMAIL_TOP_TERMS, "Top words linked to phishing (Logistic Regression weights)")

        c3, c4 = st.columns(2)
        with c3:
            show_image(paths.EMAIL_ROC, "ROC curves of all 5 email classifiers")
        with c4:
            show_image(paths.EMAIL_PR, "Precision-recall curves of all 5 email classifiers")