"""
ai_explainer.py

OPTIONAL feature: asks OpenAI's GPT-4o to write a short, friendly explanation
of a result. The machine learning model always makes the actual decision. The
language model only writes explanation text.

The feature is OFF unless an API key is stored in .streamlit/secrets.toml:
    OPENAI_API_KEY = "your key here"
Without a key, every function here returns None and the app shows the
built-in rule-based explanation instead.

Privacy note: when the feature is ON, the URL, or the first 600 characters of
the email, is sent to OpenAI. This must be stated in the report.
"""

import streamlit as st

# The openai package is optional. If it is not installed, the feature stays off.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _get_client():
    """Create the OpenAI client from the Streamlit secrets, or return None if there is no key."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-openai-api-key-here":
            return None
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _ask_gpt(prompt):
    """Send one prompt to GPT-4o. Returns the text, or None if anything goes wrong."""
    if not OPENAI_AVAILABLE:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4,  # low value = calm, consistent wording
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # No internet, wrong key, rate limit and so on: the app simply shows
        # the rule-based explanation instead.
        return None


def explain_url(url: str, prediction: int, confidence: float, indicators: list[str]) -> str | None:
    """
    Explanation for a URL result.

    prediction  1 = phishing, 0 = legitimate
    confidence  model confidence from 0 to 1
    indicators  short labels from the rule-based explanation
    """
    label = "Phishing" if prediction == 1 else "Legitimate"
    indicator_text = (
        "\n".join(f"- {i}" for i in indicators)
        if indicators
        else "- No single dominant indicator; overall structural pattern matches phishing."
    )

    prompt = f"""You are a cybersecurity expert assistant for a phishing detection research system.

A URL has been classified by a machine learning model trained on 18 lexical URL features.

URL: {url}
Classification: {label}
Model Confidence: {confidence * 100:.1f}%
Structural Risk Indicators Detected:
{indicator_text}

Write a concise, professional explanation (2-3 short paragraphs) for a non-technical user that:
1. Clearly explains why this URL was classified as {label} based on the structural indicators above.
2. If Phishing: Explains what the user should do immediately and how to protect themselves.
   If Legitimate: Reminds the user to still stay cautious and verify the URL themselves.
3. Ends with one key cybersecurity tip relevant to this specific case.

Important rules:
- Do NOT use bullet points. Write in clear, flowing paragraphs.
- Do NOT claim the URL is definitely safe or dangerous. Acknowledge it is a structural analysis.
- Keep the tone professional, calm, and educational, not alarmist.
- Keep total response under 200 words.
"""
    return _ask_gpt(prompt)


def explain_email(email_text: str, prediction: int, confidence: float, found_signals: list[str]) -> str | None:
    """
    Explanation for an email result.

    prediction     1 = phishing, 0 = legitimate
    confidence     model confidence from 0 to 1
    found_signals  short labels of phishing phrases found in the text
    """
    label = "Phishing" if prediction == 1 else "Legitimate (Safe)"
    signal_text = (
        "\n".join(f"- {s}" for s in found_signals)
        if found_signals
        else "- No dominant trigger phrase found; overall vocabulary pattern matches phishing."
    )

    # Only the first 600 characters are sent, to keep the request small.
    truncated_email = email_text[:600] + ("..." if len(email_text) > 600 else "")

    prompt = f"""You are a cybersecurity expert assistant for a phishing email detection research system.

An email has been classified by a machine learning NLP model trained on TF-IDF vocabulary features.

Classification: {label}
Model Confidence: {confidence * 100:.1f}%
Phishing Language Signals Detected:
{signal_text}

Email Text (first 600 characters):
\"\"\"{truncated_email}\"\"\"

Write a concise, professional explanation (2-3 short paragraphs) for a non-technical user that:
1. Clearly explains why this email was classified as {label} based on the language signals above.
2. If Phishing: Explains what the user should do immediately.
   If Legitimate: Reminds the user to still verify the sender and stay cautious.
3. Ends with one key email safety tip relevant to this specific case.

Important rules:
- Do NOT use bullet points. Write in clear, flowing paragraphs.
- Do NOT claim the email is definitely malicious or safe. It is a statistical prediction.
- Keep the tone professional, calm, and educational.
- Keep total response under 200 words.
"""
    return _ask_gpt(prompt)