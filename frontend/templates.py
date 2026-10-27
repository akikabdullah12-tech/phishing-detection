"""
templates.py

Small functions that build the HTML pieces of the web page (header, result
card, information boxes, sidebar cards). The app passes the returned text to
st.markdown(..., unsafe_allow_html=True). The matching styles are in
frontend/styles.css.

Anything the user types (for example the URL) is escaped with html.escape
before it goes into the HTML, so it can never run as code on the page.
"""

from html import escape


# The three limits between the four risk bands (Low, Medium, High, Critical).
# They are the same values used in utils/prediction.risk_label.
RISK_BANDS = (0.40, 0.65, 0.85)


def hero_banner(title, subtitle):
    """Page header, with a small fixed example of how a URL is split into parts."""
    return f"""
    <div class="hero">
        <div class="hero-text">
                        <div class="hero-title">{escape(title)}</div>
            <p>{escape(subtitle)}</p>
        </div>
        <div class="hero-demo" aria-hidden="true">
            <div class="hero-demo-caption">Example of a link the analyser would flag</div>
            {anatomy_strip([
                ("scheme", "http://", True),
                ("subdomain", "login.paypa1-secure.", True),
                ("domain", "example.com", False),
                ("path and query", "/verify?id=48213", False),
            ])}
        </div>
    </div>
    """


def anatomy_strip(parts):
    """
    Show a URL as labelled boxes, one box per part.

    parts is a list of (label, text, flagged) tuples, for example
    ("subdomain", "login.secure.", True). Flagged parts get a red outline.
    """
    cells = []
    for label, text, flagged in parts:
        # Skip parts that are empty (for example a URL without a query).
        if not text:
            continue
        cls = "anat-part flagged" if flagged else "anat-part"
        cells.append(
            f'<div class="{cls}">'
            f'<span class="anat-text">{escape(text)}</span>'
            f'<span class="anat-label">{escape(label)}</span>'
            f"</div>"
        )
    return f'<div class="anat-strip">{"".join(cells)}</div>'


def verdict_panel(kind, title, message, probability, risk, confidence):
    """
    The main result card, used by both the URL and the email analyser.

    kind         'danger' or 'safe' (decides the colour)
    probability  phishing probability from 0 to 1 (moves the marker on the meter)
    risk         text such as 'High Risk'
    confidence   how sure the model is about its answer, from 0 to 1
    """
    # Keep the marker inside the bar even if the value is slightly out of range.
    pct = max(0.0, min(1.0, float(probability))) * 100
    low, mid, high = (b * 100 for b in RISK_BANDS)
    return f"""
    <section class="verdict {kind}" aria-live="polite">
        <div class="verdict-head">
            <h3>{escape(title)}</h3>
            <div class="verdict-facts">
                <span><b>{confidence * 100:.1f}%</b> confidence</span>
                <span><b>{escape(risk)}</b></span>
            </div>
        </div>
        <p>{escape(message)}</p>
        <div class="meter" role="img"
             aria-label="Phishing probability {pct:.0f} percent, {escape(risk)}">
            <div class="meter-track">
                <div class="meter-band" style="left:0;width:{low}%"></div>
                <div class="meter-band" style="left:{low}%;width:{mid - low}%"></div>
                <div class="meter-band" style="left:{mid}%;width:{high - mid}%"></div>
                <div class="meter-band" style="left:{high}%;width:{100 - high}%"></div>
                <div class="meter-marker" style="left:{pct}%"></div>
            </div>
            <div class="meter-scale">
                <span style="width:{low}%">Low</span>
                <span style="width:{mid - low}%">Medium</span>
                <span style="width:{high - mid}%">High</span>
                <span style="width:{100 - high}%">Critical</span>
            </div>
            <div class="meter-caption">Phishing probability: <b>{pct:.1f}%</b></div>
        </div>
    </section>
    """


def risk_banner(kind, title, message):
    """A plain notification banner. kind is 'danger', 'safe' or 'warning'."""
    return f"""
    <div class="risk-banner {kind}">
        <h3>{escape(title)}</h3>
        <p>{escape(message)}</p>
    </div>
    """


def info_card(title, body_html):
    """An information box. body_html is text written by us in the source code, not by the user."""
    return f"""
    <div class="info-card">
        <h3>{title}</h3>
        {body_html}
    </div>
    """


def sidebar_model_card(label, model_name):
    """A small sidebar entry that shows which model is loaded and ready."""
    return f"""
    <div class="sidebar-model-card">
        <div>
            <div class="sidebar-model-label">{escape(label)}</div>
            <div class="sidebar-model-name">{escape(model_name)}</div>
        </div>
        <div class="sidebar-model-badge">Ready</div>
    </div>
    """