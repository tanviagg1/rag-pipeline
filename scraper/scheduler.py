"""
Scheduler — runs the scraping pipeline on a cron schedule using APScheduler.

Why scheduling?
  A knowledge base goes stale without fresh content. The scheduler
  re-scrapes a list of URLs on a regular interval, deduplicating so
  already-stored pages are skipped. New pages (or force-refreshed pages)
  are ingested automatically.

How it works:
  1. You define a list of URLs to watch
  2. The scheduler runs every N minutes (default: 60)
  3. Each run calls ScrapingPipeline.ingest_many() for all URLs
  4. Already-stored URLs are skipped — only new or force-refreshed ones are ingested
  5. Runs until interrupted (Ctrl+C)

Usage:
    scheduler = Scheduler(urls=["https://en.wikipedia.org/wiki/CAP_theorem"])
    scheduler.start()   # blocks until Ctrl+C

    # Or run once immediately without scheduling
    scheduler.run_once()
"""

import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from scraper.pipeline import ScrapingPipeline, IngestResult

# Default interval between scraping runs (minutes)
DEFAULT_INTERVAL_MINUTES = 60

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Runs the scraping pipeline on a fixed interval.

    Args:
        urls:              List of URLs to scrape on each run
        interval_minutes:  How often to run (default: 60 minutes)
        pipeline:          ScrapingPipeline instance (created fresh if not provided)
        force_refresh:     If True, re-ingest all URLs every run (ignores dedup)

    Usage:
        scheduler = Scheduler(
            urls=["https://en.wikipedia.org/wiki/Raft_(algorithm)"],
            interval_minutes=30,
        )
        scheduler.start()
    """

    def __init__(
        self,
        urls: list[str],
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        pipeline: ScrapingPipeline | None = None,
        force_refresh: bool = False,
    ):
        self.urls = urls
        self.interval_minutes = interval_minutes
        self.pipeline = pipeline or ScrapingPipeline()
        self.force_refresh = force_refresh
        self._scheduler = BlockingScheduler()

    def start(self) -> None:
        """
        Start the scheduler — runs immediately, then repeats on the interval.

        Blocks until Ctrl+C. Logs each run's results.
        """
        # Register the job — runs every interval_minutes
        self._scheduler.add_job(
            func=self._run_job,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="scraping_job",
            name="Scraping pipeline",
            next_run_time=datetime.now(),  # run immediately on start
        )

        print(f"Scheduler started — running every {self.interval_minutes} minutes.")
        print(f"Watching {len(self.urls)} URL(s). Press Ctrl+C to stop.\n")

        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("\nScheduler stopped.")

    def run_once(self) -> list[IngestResult]:
        """
        Run the pipeline once immediately without scheduling.

        Useful for manual triggers and testing.

        Returns:
            List of IngestResult for each URL
        """
        return self._run_job()

    def _run_job(self) -> list[IngestResult]:
        """
        The job function called by APScheduler on each interval.

        Runs ingest_many() for all watched URLs and logs a summary.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] Scheduler run starting — {len(self.urls)} URL(s)")
        print("-" * 60)

        results = self.pipeline.ingest_many(self.urls, force=self.force_refresh)

        # Print summary
        ingested = [r for r in results if r.success and not r.skipped]
        skipped = [r for r in results if r.skipped]
        failed = [r for r in results if not r.success]

        print(f"\nRun complete: {len(ingested)} ingested, {len(skipped)} skipped, {len(failed)} failed")

        return results
