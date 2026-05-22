"""
PlaywrightScraper — scrapes JavaScript-rendered pages using a headless browser.

Why Playwright?
  Many modern sites (React, Vue, Angular) render content via JavaScript.
  BeautifulSoup only sees the raw HTML before JS runs — so dynamic content
  like news feeds, SPA pages, or lazy-loaded sections shows up as empty.
  Playwright runs a real headless browser, waits for the page to render,
  then extracts the fully-rendered HTML.

When to use which scraper:
  - Static HTML (Wikipedia, docs sites) → WebScraper (BeautifulSoup, faster)
  - JS-rendered sites (news sites, SPAs) → PlaywrightScraper (slower, full render)

Usage:
    scraper = PlaywrightScraper()
    page = scraper.scrape("https://news.ycombinator.com")
    print(page.title)
    print(page.text[:500])

    # Or with async
    async with PlaywrightScraper() as scraper:
        page = await scraper.scrape_async("https://example.com")
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from scraper.scraper import ScrapedPage, ScraperError, BOILERPLATE_TAGS
from bs4 import BeautifulSoup

# How long to wait for the page to load (milliseconds)
PAGE_LOAD_TIMEOUT = 15_000  # 15 seconds

# Wait for network to be idle after initial load — catches lazy-loaded content
NETWORK_IDLE_TIMEOUT = 5_000  # 5 seconds


class PlaywrightScraper:
    """
    Headless browser scraper for JavaScript-rendered pages.

    Uses Playwright's Chromium in headless mode. Waits for the page
    to fully render before extracting text.

    Usage:
        scraper = PlaywrightScraper()
        page = scraper.scrape("https://news.ycombinator.com")
    """

    def scrape(self, url: str) -> ScrapedPage:
        """
        Launch a headless browser, load the URL, and extract clean text.

        Args:
            url: The page URL to scrape

        Returns:
            ScrapedPage with title and cleaned body text

        Raises:
            ScraperError: If the page fails to load or has no usable text
        """
        try:
            with sync_playwright() as p:
                # Launch Chromium headless — no visible browser window
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # Navigate and wait for network to idle (page fully rendered)
                try:
                    page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
                    page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)
                except PlaywrightTimeout:
                    # Timeout on network idle is non-fatal — content may still be present
                    pass

                html = page.content()
                browser.close()

        except Exception as e:
            raise ScraperError(f"Playwright failed for {url}: {e}") from e

        return self._parse(url, html)

    def _parse(self, url: str, html: str) -> ScrapedPage:
        """
        Parse the fully-rendered HTML into a ScrapedPage.

        Same logic as WebScraper._parse() — remove boilerplate, extract paragraphs.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Extract title before removing tags
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Remove boilerplate
        for tag in BOILERPLATE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Extract paragraphs
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        ]

        text = "\n\n".join(paragraphs)

        if not text.strip():
            raise ScraperError(f"No usable text extracted from {url}")

        return ScrapedPage(url=url, title=title or "Untitled", text=text)
