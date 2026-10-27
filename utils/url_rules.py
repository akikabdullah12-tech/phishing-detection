"""
url_rules.py

Rule-based helpers for the URL analyser. They do NOT change what the machine
learning model learns. They are used in two ways:

1. A short list of well-known domains (TRUSTED_DOMAINS). If the model flags one
   of these as phishing, the app treats it as legitimate instead.
2. Plain-language explanations of which parts of a URL look suspicious or safe.
   The fixed limits used there (for example "longer than 75 characters") are only
   for the explanation text. They have no effect on the model's prediction.
"""

from urllib.parse import urlparse


# Domains of well-known organisations. This list is written by hand, it is not
# learned from data, and it must be described as such in the report.
TRUSTED_DOMAINS = {
    # Search and technology
    "google.com", "www.google.com", "google.co.uk",
    "microsoft.com", "www.microsoft.com", "login.microsoftonline.com", "microsoftonline.com",
    "apple.com", "www.apple.com", "appleid.apple.com", "support.apple.com",
    "amazon.com", "www.amazon.com", "amazon.co.uk", "www.amazon.co.uk",
    "facebook.com", "www.facebook.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "linkedin.com", "www.linkedin.com",
    "github.com", "www.github.com",
    "youtube.com", "www.youtube.com",
    "instagram.com", "www.instagram.com",
    "netflix.com", "www.netflix.com",
    "spotify.com", "www.spotify.com",
    "openai.com", "www.openai.com", "chatgpt.com", "www.chatgpt.com",
    # Banking and finance
    "paypal.com", "www.paypal.com",
    "barclays.co.uk", "www.barclays.co.uk",
    "hsbc.co.uk", "www.hsbc.co.uk",
    "lloydsbank.com", "www.lloydsbank.com",
    "natwest.com", "www.natwest.com",
    # Government, health and media
    "gov.uk", "bbc.co.uk", "www.bbc.co.uk", "bbc.com", "www.bbc.com",
    "nhs.uk", "www.nhs.uk",
    # Microsoft consumer accounts
    "live.com", "account.live.com", "login.live.com", "outlook.com", "www.outlook.com",
    # Cloud storage
    "dropbox.com", "www.dropbox.com",
}


def is_trusted_domain(url):
    """Return True if the URL's host is on the trusted list, or is a subdomain of a listed domain."""
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        hostname = (parsed.hostname or "").lower().strip()
        return hostname in TRUSTED_DOMAINS or any(
            hostname.endswith("." + d) for d in TRUSTED_DOMAINS
        )
    except Exception:
        return False


def build_url_explanation(feat_df, trusted_tld):
    """
    Turn the 18 feature values into readable explanations.

    Returns two lists:
      suspicious   (title, explanation) pairs for features that look risky
      safe_points  short sentences for features that look normal
    """
    row = feat_df.iloc[0]
    suspicious = []
    safe_points = []

    url_len = float(row["URLLength"])
    subdomains = int(row["NoOfSubDomain"])

    # --- Things that look suspicious ---
    if int(row["IsDomainIP"]) == 1:
        suspicious.append((
            "🔴 IP Address Used as Domain",
            "This URL uses a raw IP address (e.g. http://192.168.1.1/login) instead of a proper "
            "domain name. Legitimate websites almost never use raw IP addresses. Attackers use this "
            "technique to avoid registering a traceable domain name."
        ))

    if int(row["IsHTTPS"]) == 0:
        suspicious.append((
            "🔴 No HTTPS Encryption",
            "This URL does not use HTTPS, meaning the connection is unencrypted. Any data you enter "
            "(passwords, card numbers) could be intercepted. All reputable websites use HTTPS today."
        ))

    if int(row["HasObfuscation"]) == 1:
        suspicious.append((
            "🔴 Character Obfuscation Detected",
            "The URL contains percent-encoded characters (e.g. %40, %2F) that are used to disguise "
            "its true destination. Phishing links often encode characters to bypass security filters "
            "and confuse users reading the link."
        ))

    if url_len > 75:
        suspicious.append((
            f"🔴 Abnormally Long URL ({int(url_len)} characters)",
            f"This URL is {int(url_len)} characters long. Phishing links are frequently padded with "
            "extra parameters or random strings to hide the real destination and make the link harder "
            "to read and analyse."
        ))

    if subdomains >= 3:
        suspicious.append((
            f"🔴 Excessive Subdomain Depth ({subdomains} subdomains)",
            f"This URL contains {subdomains} subdomains. Attackers often create deep subdomain chains "
            "like 'login.secure.verify.paypal.attacker.com' to make the link appear to belong to a "
            "trusted brand while the real domain is controlled by the attacker."
        ))

    if float(row["SpacialCharRatioInURL"]) > 0.20:
        ratio = float(row["SpacialCharRatioInURL"])
        suspicious.append((
            f"🔴 High Special Character Density ({ratio:.0%})",
            "The URL path or query string contains an unusually high proportion of special characters "
            "such as @, -, _, ~, and %. Legitimate URLs tend to be clean and readable. This pattern "
            "is commonly seen in obfuscated phishing links."
        ))

    if int(row["NoOfQMarkInURL"]) > 1:
        suspicious.append((
            "🔴 Multiple Query Delimiters (?)",
            "This URL contains more than one '?' character. Legitimate URLs use a single query string "
            "delimiter. Multiple '?' marks can indicate a malformed or deliberately confusing URL "
            "structure designed to bypass filters."
        ))

    if int(row["NoOfEqualsInURL"]) >= 3:
        eq_count = int(row["NoOfEqualsInURL"])
        suspicious.append((
            f"🔴 High Parameter Count ({eq_count} '=' signs)",
            f"The URL contains {eq_count} assignment operators in its query string. Phishing links "
            "frequently carry many hidden tracking or redirect parameters to route victims through "
            "multiple servers before reaching the fake login page."
        ))

    # --- Things that look normal ---
    if int(row["IsHTTPS"]) == 1:
        safe_points.append("✅ HTTPS encryption is active — the connection is secure.")

    if int(row["IsDomainIP"]) == 0:
        safe_points.append("✅ A proper domain name is used rather than a raw IP address.")

    if url_len <= 75:
        safe_points.append(f"✅ URL length is reasonable ({int(url_len)} characters) — not excessively padded.")

    if subdomains < 3:
        safe_points.append(f"✅ Subdomain depth is normal ({subdomains} subdomain(s) detected).")

    if int(row["HasObfuscation"]) == 0:
        safe_points.append("✅ No character obfuscation patterns detected in the URL.")

    if trusted_tld:
        safe_points.append("✅ The domain uses a recognised trusted TLD (e.g. .ac.uk, .gov.uk, .edu, .de).")

    return suspicious, safe_points