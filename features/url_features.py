"""
url_features.py

Turns a URL into 18 numbers (features) that a machine learning model can use.
Only the text of the URL is looked at. The website itself is never visited.

The same function is used for training and inside the web app, so a live
prediction always uses exactly the same features as the training data.
"""

import ipaddress
import re
from urllib.parse import urlparse

import pandas as pd
import tldextract


# The 18 feature names, in a fixed order. The model expects this exact order.
# The names (including the spellings "Degit" and "Spacial") follow the original dataset.
FEATURE_COLS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

# Well-known domain endings (UK, European and a few international ones).
# The web app uses this list only to show a hint in the explanation.
# It is NOT one of the 18 model features.
TRUSTED_TLDS = {
    "ac.uk", "co.uk", "gov.uk", "nhs.uk", "org.uk", "police.uk", "sch.uk", "me.uk", "net.uk", "uk",
    "edu", "gov", "mil", "int", "org",
    "de", "fr", "nl", "se", "no", "dk", "fi", "at", "ch", "be", "pt", "ie", "es", "it", "eu",
    "edu.au", "ac.nz", "ac.jp", "edu.sg", "edu.cn",
}


def add_scheme(url):
    """Add http:// in front if the URL has no scheme, so it can be parsed properly."""
    url = str(url).strip()
    if not url:
        raise ValueError("URL cannot be empty.")
    # A scheme looks like "http://", "https://", "ftp://" and so on.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def is_ip(host):
    """Return 1 if the host is a raw IP address (like 192.168.1.1), otherwise 0."""
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def count_percent_encoded(url):
    """Count percent-encoded pieces such as %20 or %3A (often used to hide text)."""
    return len(re.findall(r"%[0-9a-fA-F]{2}", url))


def is_known_tld(tld):
    """Check if a domain ending is in our list of well-known endings."""
    return tld in TRUSTED_TLDS


def is_known_url_tld(url):
    """Check if the domain ending of a full URL is in our list of well-known endings."""
    try:
        host = (urlparse(add_scheme(str(url).strip())).hostname or "").lower()
        tld = tldextract.extract(host).suffix
        return is_known_tld(tld)
    except Exception:
        return False


def get_url_features(url):
    """Extract the 18 features from one URL and return them as a one-row table."""
    raw_url = str(url).strip()

    # add_scheme() lets inputs like "example.com/page" be parsed like a full URL.
    parsed = urlparse(add_scheme(raw_url))

    host = (parsed.hostname or "").lower()

    # tldextract splits the host into subdomain, domain and suffix,
    # and it understands endings like "co.uk".
    ext = tldextract.extract(host)
    tld = ext.suffix or ""

    total_len = len(raw_url)
    # Used as the divisor for the ratios, never zero.
    safe_len = max(total_len, 1)

    subdomains = [s for s in ext.subdomain.split(".") if s]
    num_encoded = count_percent_encoded(raw_url)
    num_letters = sum(c.isalpha() for c in raw_url)
    num_digits = sum(c.isdigit() for c in raw_url)

    # Characters we treat as normal in a URL. Anything else counts as "unusual".
    normal_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/:%?=&")
    num_unusual = sum(c not in normal_chars for c in raw_url)
    num_special = sum(not c.isalnum() for c in raw_url)

    # One value per feature. The keys must match FEATURE_COLS exactly.
    features = {
        "URLLength": total_len,
        "DomainLength": len(host),
        "IsDomainIP": is_ip(host),
        "TLDLength": len(tld),
        "NoOfSubDomain": len(subdomains),
        "HasObfuscation": int(num_encoded > 0),
        "NoOfObfuscatedChar": num_encoded,
        "ObfuscationRatio": num_encoded / safe_len,
        "NoOfLettersInURL": num_letters,
        "LetterRatioInURL": num_letters / safe_len,
        "NoOfDegitsInURL": num_digits,
        "DegitRatioInURL": num_digits / safe_len,
        "NoOfEqualsInURL": raw_url.count("="),
        "NoOfQMarkInURL": raw_url.count("?"),
        "NoOfAmpersandInURL": raw_url.count("&"),
        "NoOfOtherSpecialCharsInURL": num_unusual,
        "SpacialCharRatioInURL": num_special / safe_len,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
    }

    return pd.DataFrame([features], columns=FEATURE_COLS)