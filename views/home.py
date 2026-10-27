"""
home.py

The "System Overview" tab: what the two detectors do, how well the models
scored, how to use the app, and some sample inputs to try.
"""

import streamlit as st

from frontend.templates import info_card


# --- Sample inputs for people who want to try the app ---

URL_SAMPLES = [
    {
        "label": "🔴 Phishing Example 1",
        "url": "http://192.168.1.1/secure-login/verify?user=admin&token=abc123xyz&session=true",
        "reason": "Uses a raw IP address, no HTTPS, and multiple query parameters.",
    },
    {
        "label": "🔴 Phishing Example 2",
        "url": "http://paypa1-secure.login.verify.account-update.com/signin?redirect=true&next=dashboard",
        "reason": "Misspelled brand name, deep subdomain chain, no HTTPS, long URL.",
    },
    {
        "label": "🔴 Phishing Example 3",
        "url": "http://secure-banking.login-update.com/account%40verify%2Ftoken?id=99321&confirm=yes&auth=1",
        "reason": "Percent-encoded obfuscation, no HTTPS, high special character density.",
    },
    {
        "label": "🟢 Legitimate Example 1",
        "url": "https://www.bbc.co.uk/news",
        "reason": "HTTPS active, short clean URL, trusted .co.uk TLD, no suspicious patterns.",
    },
    {
        "label": "🟢 Legitimate Example 2",
        "url": "https://www.gov.uk/apply-universal-credit",
        "reason": "HTTPS active, official .gov.uk domain, clean URL structure.",
    },
]

EMAIL_SAMPLES = [
    {
        "label": "🔴 Phishing Email Example 1: Account Suspension",
        "text": (
            "Dear Customer,\n\n"
            "Your account has been flagged for suspicious activity. "
            "Your access will be suspended within 24 hours unless you verify your identity immediately.\n\n"
            "Please click the secure link below to confirm your account and prevent suspension:\n"
            "http://account-verify.secure-login.com/confirm?user=12345\n\n"
            "Do not share this message with anyone.\n\n"
            "Regards,\nSecurity Team"
        ),
    },
    {
        "label": "🔴 Phishing Email Example 2: Prize Scam",
        "text": (
            "Congratulations! You have been selected as our lucky winner this month.\n\n"
            "You have won a £500 Amazon gift card. To claim your prize, please confirm your details "
            "by clicking the link below within 48 hours or the reward will be forfeited.\n\n"
            "Click here to claim: http://prize-claim.win-reward.net/amazon?ref=winner99\n\n"
            "This offer is strictly confidential. Do not inform others."
        ),
    },
    {
        "label": "🔴 Phishing Email Example 3: Bank Credential Harvest",
        "text": (
            "Important Notice from Your Bank\n\n"
            "We have detected unusual login activity on your account. "
            "To protect your funds, we require you to verify your login credentials immediately.\n\n"
            "Please provide your username and password on our secure verification page:\n"
            "http://barclays-secure.account-login.net/verify\n\n"
            "Failure to act within 12 hours will result in your account being locked.\n\n"
            "The Security Department"
        ),
    },
    {
        "label": "🟢 Legitimate Email Example: Meeting Request",
        "text": (
            "Hi Sarah,\n\n"
            "I hope you are well. I wanted to follow up regarding the project update we discussed last week.\n\n"
            "Could we schedule a meeting for Thursday at 2pm? I have attached the agenda and the latest "
            "report for your review ahead of the session.\n\n"
            "Please let me know if the time works for you or if you would prefer an alternative slot.\n\n"
            "Kind regards,\nJames"
        ),
    },
]


def render(url_meta, email_meta):
    """Draw the overview tab. The metadata dictionaries come from the training scripts."""
    st.subheader("Architecture & Detection Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            info_card(
                "🌐 URL Lexical Threat Engine",
                "<p>Analyzes 18 structural and lexical features extracted directly from URL strings "
                "(length, subdomain count, special character ratios, obfuscation patterns).</p>"
                "<p><strong>No site visits:</strong> The analyzer only reads the text of the link. It never connects to or visits the website.</p>",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            info_card(
                "📧 Email Content NLP Classifier",
                "<p>Transforms raw email body text into a high-dimensional TF-IDF vector matrix "
                "(10,000 features, 1-2 word n-grams) to detect phishing language signals.</p>"
                "<p><strong>Local by default:</strong> Text is classified on this machine. It leaves the app only if the optional AI explainer is switched on.</p>",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    perf_col, survey_col = st.columns([3, 1])

    with perf_col:
        st.subheader("Model Performance Summary")
        # These numbers are read from the metadata files saved during training,
        # so they always match the models that are actually loaded.
        url_f1 = url_meta.get("f1_score")
        email_f1 = email_meta.get("f1_score")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("URL Classifier", url_meta.get("best_model", "n/a"))
        m2.metric("URL F1 Benchmark", f"{url_f1:.4f}" if isinstance(url_f1, float) else "n/a")
        m3.metric("Email Classifier", email_meta.get("best_model", "n/a"))
        m4.metric("Email F1 Benchmark", f"{email_f1:.4f}" if isinstance(email_f1, float) else "n/a")

    with survey_col:
        st.subheader("📋 Participant Survey")
        st.markdown(
            "<div style='font-size:0.85rem; color:#64748B; margin-bottom:0.75rem;'>"
            "After testing the system, please share your feedback to support this research."
            "</div>",
            unsafe_allow_html=True,
        )
        st.link_button(
            "📝 Take the Survey",
            "https://forms.cloud.microsoft/r/3b4ycszw9x",
            use_container_width=True,
            type="primary",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card">
            <h3>📋 Workflow Protocol</h3>
            <ol style="color: var(--ink-soft); line-height: 1.8; margin-bottom: 0; padding-left: 20px; font-weight: 500;">
                <li>Select either the <strong>URL Threat Analyzer</strong> or <strong>Email Body Analyzer</strong> tab.</li>
                <li>Enter the target URL string or paste raw email body text.</li>
                <li>Click <strong>Analyse</strong> to trigger the underlying feature extractor and model pipeline.</li>
                <li>Inspect the confidence probability, threat classification, and contextual warning indicators.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sample inputs ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧪 Sample Test Cases")
    st.markdown(
        "Use the examples below to test the system. Copy any URL or email text and paste it into "
        "the corresponding analyser tab. These cases cover both phishing and legitimate examples."
    )

    url_tab, email_tab = st.tabs(["🌐 URL Test Cases", "📧 Email Test Cases"])

    with url_tab:
        st.markdown("Copy any URL below and paste it into the **URL Threat Analyzer** tab.")
        st.markdown("---")
        for sample in URL_SAMPLES:
            st.markdown(f"**{sample['label']}**")
            st.code(sample["url"], language=None)
            st.caption(f"💬 Why: {sample['reason']}")
            st.markdown("---")

    with email_tab:
        st.markdown("Copy any email text below and paste it into the **Email Body Analyzer** tab.")
        st.markdown("---")
        for sample in EMAIL_SAMPLES:
            st.markdown(f"**{sample['label']}**")
            st.code(sample["text"], language=None)
            st.markdown("---")