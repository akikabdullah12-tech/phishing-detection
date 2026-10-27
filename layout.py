"""
layout.py

Two things that every page of the app needs: loading the CSS file and
drawing the sidebar.
"""

from pathlib import Path

import streamlit as st

from frontend.templates import sidebar_model_card


# The stylesheet sits in the frontend folder, next to this file's parent folder.
_CSS_FILE = Path(__file__).resolve().parent / "frontend" / "styles.css"


def inject_styles():
    """Read frontend/styles.css and add it to the page."""
    css = _CSS_FILE.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar(url_meta, email_meta):
    """Draw the sidebar: project info, which models are loaded, and the survey link."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🛡️ Control Center</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 1.25rem;">
                <strong>Project:</strong> Phishing Detection System<br>
                <strong>Author:</strong> Akik Abdullah
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown('<div class="sidebar-brand" style="font-size: 1rem;">⚡ Engine Status</div>', unsafe_allow_html=True)

        # The model names come from the metadata files written during training.
        # The second value in each .get() is only a fallback if the file is missing.
        url_model_name = url_meta.get("best_model", "XGBoost")
        email_model_name = email_meta.get("best_model", "Support Vector Machine")

        st.markdown(sidebar_model_card("URL Model", url_model_name), unsafe_allow_html=True)
        st.markdown(sidebar_model_card("Email Model", email_model_name), unsafe_allow_html=True)

        st.divider()

        st.markdown(
            """
            <div style="background: rgba(22, 48, 92, 0.5); border: 1px solid #16305C; border-radius: 10px; padding: 14px; font-size: 0.82rem; color: #C7D3E8;">
                🔒 <strong>Academic Prototype</strong><br>
                Local analytical classification for cybersecurity evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("**📋 User Evaluation Survey**")
        st.markdown(
            "<div style='font-size:0.82rem; color:#94b8b4; margin-bottom:0.6rem;'>"
            "Please complete the survey after testing the system.</div>",
            unsafe_allow_html=True,
        )
        st.link_button(
            "📝 Take the Survey",
            "https://forms.cloud.microsoft/r/3b4ycszw9x",
            use_container_width=True,
            type="primary",
        )