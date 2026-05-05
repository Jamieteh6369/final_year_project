import requests
import trafilatura
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_article_url(hn_url: str) -> str:
    resp = requests.get(hn_url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title_span = soup.select_one("span.titleline > a")
    if not title_span:
        raise ValueError("Could not find an article link on that HackerNews page.")
    article_url = title_span["href"]
    if article_url.startswith("item?"):
        raise ValueError("This looks like a self-post (Ask HN) — no external article to fetch.")
    return article_url


def _bs4_extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(p for p in paragraphs if len(p) > 40)


def _is_cloudflare_block(html: str) -> bool:
    markers = ["Just a moment", "cf-browser-verification", "Enable JavaScript", "Checking your browser"]
    return any(m in html for m in markers)


def fetch_article_text(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            downloaded = resp.text
        except requests.RequestException as e:
            raise ValueError(f"Could not download page: {url} — {e}")
    if _is_cloudflare_block(downloaded):
        raise ValueError(
            f"'{url}' is protected by Cloudflare bot detection. Try a different article."
        )
    text = (
        trafilatura.extract(downloaded, include_comments=False, no_fallback=False)
        or trafilatura.extract(downloaded, favour_recall=True)
        or _bs4_extract(downloaded)
    )
    if not text:
        raise ValueError(f"Could not extract article text from: {url}")
    return text
