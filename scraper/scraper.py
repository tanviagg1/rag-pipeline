"""
Scraper — fetches a web page and extracts clean text using BeautifulSoup.

Only handles static HTML pages (no JavaScript rendering).
For JS-heavy sites, see scraper/playwright_scraper.py (Phase 3).

How it works:
  1. Fetches the URL with requests (standard HTTP GET)
  2. Parses the HTML with BeautifulSoup
  3. Strips boilerplate (nav, header, footer, scripts, styles)
  4. Extracts paragraph text and joins into a clean string

Usage:
    scraper = WebScraper()
    result = scraper.scrape("https://en.wikipedia.org/wiki/CAP_theorem")
    print(result.title)    # "CAP theorem"
    print(result.text[:200])
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass


# Tags that contain boilerplate — navigation, ads, scripts, styles.
# Removing these before extracting text prevents noise in the knowledge base.
BOILERPLATE_TAGS = [
    "nav", "header", "footer", "aside", "script",
    "style", "noscript", "form", "button",
]

# Request timeout — don't hang indefinitely on slow/unresponsive sites
REQUEST_TIMEOUT = 10

# User-agent header — some sites block the default Python requests agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; rag-pipeline/1.0; "
        "+https://github.com/tanviagg1/rag-pipeline)"
    )
}


@dataclass
class ScrapedPage:
    """
    The result of scraping a single page.

    url    — the page URL (used as a unique identifier for deduplication)
    title  — page title extracted from <title> or <h1>
    text   — clean body text with whitespace normalised
    """
    url: str
    title: str
    text: str

    def __repr__(self) -> str:
        return f"ScrapedPage(url={self.url!r}, title={self.title!r}, chars={len(self.text)})"


class ScraperError(Exception):
    """Raised when a page cannot be fetched or parsed."""
    pass


class WebScraper:
    """
    Scrapes static HTML pages and returns clean text.

    Usage:
        scraper = WebScraper()
        page = scraper.scrape("https://en.wikipedia.org/wiki/CAP_theorem")
        print(page.title)
        print(page.text[:500])
    """

    def scrape(self, url: str) -> ScrapedPage:
        """
        Fetch and parse a URL, returning clean text.

        Args:
            url: The page URL to scrape

        Returns:
            ScrapedPage with title and cleaned body text

        Raises:
            ScraperError: If the request fails or returns non-200 status
        """
        html = self._fetch(url)
        return self._parse(url, html)

    def _fetch(self, url: str) -> str:
        """
        Fetch raw HTML from a URL.

        Args:
            url: The URL to fetch

        Returns:
            Raw HTML string

        Raises:
            ScraperError: On network error or non-200 response
        """
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as e:
            raise ScraperError(f"HTTP {e.response.status_code} for {url}") from e
        except requests.RequestException as e:
            raise ScraperError(f"Request failed for {url}: {e}") from e

    def _parse(self, url: str, html: str) -> ScrapedPage:
        """
        Parse raw HTML into a ScrapedPage with clean text.

        Removes boilerplate tags, then extracts paragraph text.

        Args:
            url:  The source URL (stored in ScrapedPage for metadata)
            html: Raw HTML string

        Returns:
            ScrapedPage with title and cleaned text
        """
        soup = BeautifulSoup(html, "html.parser")

        # Extract title before removing tags
        title = self._extract_title(soup)

        # Remove boilerplate tags — modifies soup in place
        for tag in BOILERPLATE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Extract paragraph text — <p> tags are the main content in most pages
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40  # skip trivially short paragraphs
        ]

        # Join paragraphs with newlines, normalise whitespace
        text = "\n\n".join(paragraphs)

        if not text.strip():
            raise ScraperError(f"No usable text extracted from {url}")

        return ScrapedPage(url=url, title=title, text=text)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the page title.

        Priority: <title> tag → first <h1> → "Untitled"
        """
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"
