"""
url_checker.py

The "URL Threat Analyzer" tab. The user types a URL, the app extracts the 18
features, the model predicts, and the page explains the result.
"""

from urllib.parse import urlparse

import streamlit as st
import tldextract

from features.url_features import add_scheme, get_url_features, is_known_url_tld
from frontend.templates import anatomy_strip, verdict_panel
from utils.ai_explainer import explain_url
from utils.prediction import get_phishing_prob, risk_label
from utils.url_rules import build_url_explanation, is_trusted_domain


# Advice shown when a URL is flagged as phishing.
PHISHING_ACTIONS = """
**Immediate Actions:**
- 🚫 **Do not visit this URL.** Close any tab where this link was opened.
- 🚫 **Do not enter any credentials** (username, password, bank details) on pages reached via this link.
- 🚫 **Do not download any files** linked from this URL. They may contain malware.
- 📧 **If received via email**, report the message to your IT or security team immediately.
- 🔗 **If you clicked the link**, run a malware scan on your device and change any passwords you may have entered.

**How to Protect Yourself:**
- Always check the full URL in your browser address bar before logging in anywhere.
- Look for the padlock icon and `https://` at the start of any website you trust.
- When in doubt, navigate directly to the official website by typing the address yourself.
- Enable two-factor authentication (2FA) on all important accounts so stolen passwords alone cannot grant access.
"""

PHISHING_AWARENESS = """
**What is Phishing?**
Phishing is a type of cyber attack where criminals create fake websites or links that impersonate
trusted organisations (banks, government services, online shops) to steal your login credentials,
financial information, or personal data.

**Common Warning Signs:**
- Urgent language, such as "Your account will be suspended in 24 hours!"
- Requests for passwords, card numbers, or personal information via a link
- Slight misspellings in domain names (e.g. paypa1.com instead of paypal.com)
- Unexpected emails or messages asking you to verify your account
- Links that look different when you hover over them versus what is displayed
"""

SAFE_REMINDERS = """
**General Cybersecurity Reminders:**
- ✅ Always verify the full domain name in your browser address bar before logging in.
- ✅ Check that the padlock icon is present, which shows an active HTTPS certificate.
- ✅ Be cautious of unexpected login prompts, even on sites that look familiar.
- ✅ Use a password manager. It will not auto-fill credentials on fake domains.
- ✅ Enable two-factor authentication (2FA) on your accounts for an extra layer of security.
- ✅ Keep your browser and operating system up to date to protect against known exploits.

> ⚠️ **Note:** This tool performs structural text analysis only. It does not visit the URL or
> verify the live website content. Always apply your own judgement alongside the model's result.
"""


def _result_box(pred, prob):
    """Show the main verdict card (verdict, confidence, risk level, probability meter)."""
    # Confidence is how sure the model is about the class it chose.
    confidence = prob if pred == 1 else 1 - prob
    risk = risk_label(prob)

    if pred == 1:
        st.markdown(
            verdict_panel(
                "danger",
                "Phishing URL detected",
                "The prediction is based on structural characteristics commonly associated with phishing URLs. "
                "The indicators below contributed to the risk assessment.",
                prob, risk, confidence,
            ),
            unsafe_allow_html=True,
        )
        st.error("⚠️ Security Warning: Do not visit this URL, enter credentials, or download linked assets.")
    else:
        st.markdown(
            verdict_panel(
                "safe",
                "Legitimate URL detected",
                "No structural patterns commonly associated with phishing were identified in this URL. "
                "The risk assessment is based on learned feature patterns from the training dataset.",
                prob, risk, confidence,
            ),
            unsafe_allow_html=True,
        )
        st.info("💡 This URL passed structural screening. Always verify domain identity before entering personal information.")


def _anatomy_section(url_input, feats, flagged):
    """
    Show the URL split into labelled parts (scheme, user info, subdomain, domain, path and query).

    If the model flagged the URL, a part is outlined when a simple rule matches it. If the URL
    was judged legitimate, nothing is outlined, because the rules are broad and would
    contradict the verdict. This is display only and never changes the prediction.
    """
    raw = url_input.strip()
    row = feats.iloc[0]
    parsed = urlparse(add_scheme(raw))
    ext = tldextract.extract(parsed.hostname or "")

    typed_scheme = "://" in raw
    is_ip = int(row["IsDomainIP"]) == 1
    deep_subdomains = int(row["NoOfSubDomain"]) >= 3
    path_flagged = (
        float(row["URLLength"]) > 75
        or int(row["HasObfuscation"]) == 1
        or float(row["SpacialCharRatioInURL"]) > 0.20
        or int(row["NoOfQMarkInURL"]) > 1
        or int(row["NoOfEqualsInURL"]) >= 3
    )

    # Everything before an "@" is not the real destination of the link.
    userinfo = parsed.netloc.rpartition("@")[0]
    rest = parsed.path + (f"?{parsed.query}" if parsed.query else "") + (f"#{parsed.fragment}" if parsed.fragment else "")
    if len(rest) > 120:
        rest = rest[:117] + "..."

    parts = [("scheme" if typed_scheme else "scheme (assumed)", f"{parsed.scheme}://", int(row["IsHTTPS"]) == 0)]
    if userinfo:
        parts.append(("user info (not the destination)", f"{userinfo}@", True))
    if is_ip:
        parts.append(("IP address instead of a domain", parsed.hostname or "", True))
    else:
        if ext.subdomain:
            parts.append(("subdomain", f"{ext.subdomain}.", deep_subdomains))
        domain = ext.domain + (f".{ext.suffix}" if ext.suffix else "")
        parts.append(("domain", domain, False))
    parts.append(("path and query", rest, path_flagged))

    if not flagged:
        # Legitimate verdict: show the parts without any outline.
        parts = [(label, text, False) for label, text, _ in parts]
        note = "The model did not flag this link, so no part is outlined."
    elif any(is_flagged for _, _, is_flagged in parts):
        note = "Outlined parts match one of the risk indicators listed further down."
    else:
        note = "The model flagged this link, but no single part matched a simple rule. The decision comes from a combination of features."

    st.markdown(
        '<div class="anat-wrap"><p class="anat-wrap-title">How this link is built</p>'
        + anatomy_strip(parts)
        + f'<p class="anat-wrap-note">{note}</p></div>',
        unsafe_allow_html=True,
    )


def _awareness_section(pred, suspicious, safe_points, url, prob):
    """Show the explanation and the safety advice below the verdict."""
    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.subheader("🔍 Why Was This URL Flagged as Phishing?")
        st.caption(
            "ℹ️ Disclaimer: This tool performs offline structural text analysis only and does not "
            "visit the URL. Occasionally, legitimate websites with short domains or complex URL "
            "paths may be flagged incorrectly (false positive). If you recognise this as a trusted "
            "site, verify it directly in your browser address bar and check for the padlock icon."
        )

        # The optional AI explainer gets only the indicator labels (text). The ML model
        # has already made the decision. Without an API key this returns None.
        indicator_labels = [title.replace("🔴 ", "") for title, _ in suspicious]
        ai_text = explain_url(url, pred, prob, indicator_labels)

        if ai_text:
            st.markdown(ai_text)
            st.markdown("---")
        else:
            st.markdown(
                "The prediction is based on structural characteristics commonly associated with phishing URLs. "
                "The **indicators below contributed to the risk assessment.**"
            )

        if suspicious:
            with st.expander("🔎 View Detected Structural Risk Indicators", expanded=False):
                for title, explanation in suspicious:
                    st.markdown(f"**{title}**")
                    st.markdown(explanation)
                    st.markdown("---")
        else:
            st.info(
                "The model's decision was driven by a combination of subtle feature patterns "
                "in the URL structure that collectively match phishing behaviour."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🛡️ What Should You Do Now?")
        st.markdown(PHISHING_ACTIONS)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📚 Phishing Awareness")
        st.info(PHISHING_AWARENESS)

    else:
        st.subheader("✅ Why This URL Appears Legitimate")

        indicator_labels = [point.replace("✅ ", "") for point in safe_points]
        ai_text = explain_url(url, pred, prob, indicator_labels)

        if ai_text:
            st.markdown(ai_text)
            st.markdown("---")
        else:
            st.markdown(
                "No significant structural patterns associated with phishing were detected in this URL. "
                "The **signals below contributed to the low-risk assessment:**"
            )

        if safe_points:
            with st.expander("🔎 View Detected Safe Structural Indicators", expanded=False):
                for point in safe_points:
                    st.success(point)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💡 Stay Safe, Even on Legitimate-Looking Sites")
        st.markdown(SAFE_REMINDERS)


def render(model, feat_cols):
    """Draw the URL analyser tab."""
    st.subheader("🌐 URL Structural Threat Inspection")
    st.caption("Structural feature extraction is performed locally. AI-powered explanations provide additional context for each result.")

    url_input = st.text_input(
        "Enter URL to Analyse",
        placeholder="e.g. https://secure-login.example.com/verify?token=abc123",
    )

    # Nothing happens until the button is pressed.
    if not st.button("Analyse URL", type="primary", width="stretch"):
        return

    if not url_input.strip():
        st.warning("Please enter a URL before running the analysis.")
        return

    try:
        feats = get_url_features(url_input)
        # Put the columns in exactly the order the model was trained with.
        feats = feats.reindex(columns=feat_cols)

        if feats.isnull().any().any():
            raise ValueError("Feature extraction failed for the input string.")

        # pred is the class (0 or 1); prob is the phishing probability used for the risk level.
        pred = int(model.predict(feats)[0])
        prob = get_phishing_prob(model, feats)
        trusted_tld = is_known_url_tld(url_input)

        # Rule on top of the model (not learned from data): if the model flags a URL
        # from a well-known domain as phishing, the app shows it as legitimate.
        # This changes the output of the app and must be described in the report.
        if pred == 1 and is_trusted_domain(url_input):
            st.info(
                "ℹ️ **Known Legitimate Domain Detected:** This URL belongs to a verified organisation "
                "(e.g. Google, Microsoft, Apple, Amazon). The structural classifier flagged it due to "
                "authentication path patterns (e.g. /login, /oauth, /account), which are common in "
                "phishing URLs but also appear on legitimate login pages. The domain itself is trusted."
            )
            pred = 0
            prob = 1 - prob

        _result_box(pred, prob)
        _anatomy_section(url_input, feats, pred == 1)

        suspicious, safe_points = build_url_explanation(feats, trusted_tld)
        _awareness_section(pred, suspicious, safe_points, url_input, prob)

        with st.expander("🔍 Inspect Raw 18-Feature Vector"):
            st.dataframe(feats.T.rename(columns={0: "Feature Value"}), width="stretch")

    except Exception as err:
        st.error("Analysis encountered an error.")
        st.code(str(err))