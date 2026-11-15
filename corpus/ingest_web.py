#!/usr/bin/env python3
"""Ingest Luganda web pages — politely, and only where you are allowed.

    python -m corpus.ingest_web --check-robots https://www.bukedde.co.ug/news/12345
    python -m corpus.ingest_web --urls urls.txt --out web
    python -m corpus.ingest_web --sitemap https://example.ug/sitemap.xml --limit 200 --out web

Rules this script enforces, because a corpus is not worth a lawsuit or a ban:
  * robots.txt is fetched and obeyed per host (urllib.robotparser), including
    Crawl-delay. Disallowed URLs are skipped and counted, never fetched.
  * one request per host every --delay seconds (default 2s), with your contact in
    the User-Agent.
  * `--check-robots` tells you *before* you invest in a scraper whether the site
    allows you to read its articles at all.

Note: robots.txt permitting a crawl does not give you the copyright. Newspapers,
textbooks and TV are all copyrighted works. Reading them to study the language,
keeping local copies for research, and republishing are three different things.
Get permission (or use openly licensed sources) before you ship a model trained on
them. CORPUS-PLAN.md lists the licences.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from .pipeline import print_stats, process, write_segments
from .text_extract import Segment, html_to_text

UA = "YogeraCorpusBot/1.0 (Luganda language research; +contact: set-your-email)"


def robots_for(client: httpx.Client, url: str, cache: dict) -> urllib.robotparser.RobotFileParser:
    host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if host in cache:
        return cache[host]
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(urljoin(host, "/robots.txt"))
    try:
        resp = client.get(parser.url, headers={"User-Agent": UA}, timeout=20)
        if resp.status_code >= 400:
            parser.parse([])          # no robots.txt = allowed
        else:
            parser.parse(resp.text.splitlines())
    except httpx.HTTPError:
        parser.parse([])
    cache[host] = parser
    return parser


def crawl_delay(parser: urllib.robotparser.RobotFileParser, default: float) -> float:
    delay = parser.crawl_delay(UA) or parser.crawl_delay("*")
    return max(float(delay or default), default)


def fetch_article(client: httpx.Client, url: str) -> Segment | None:
    resp = client.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    text = html_to_text(resp.text)
    if len(text.split()) < 30:
        return None
    return Segment(text, f"web:{urlparse(url).netloc}", "news", url, {"url": url})


def urls_from_sitemap(client: httpx.Client, sitemap: str, limit: int) -> list[str]:
    urls: list[str] = []
    pending = [sitemap]
    while pending and len(urls) < limit:
        current = pending.pop(0)
        try:
            resp = client.get(current, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  ! sitemap {current}: {exc}")
            continue
        body = resp.text
        for chunk in body.split("<url>")[1:]:
            loc = chunk.split("<loc>")[1].split("</loc>")[0].strip() if "<loc>" in chunk else ""
            if loc:
                urls.append(loc)
                if len(urls) >= limit:
                    break
        for chunk in body.split("<sitemap>")[1:]:
            loc = chunk.split("<loc>")[1].split("</loc>")[0].strip() if "<loc>" in chunk else ""
            if loc:
                pending.append(loc)
    return urls[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", help="file with one URL per line")
    group.add_argument("--sitemap", help="sitemap.xml URL to walk")
    group.add_argument("--check-robots", help="only report whether this URL may be fetched")
    ap.add_argument("--out", default="web")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests per site")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--include-english", action="store_true")
    args = ap.parse_args()

    cache: dict = {}
    with httpx.Client(follow_redirects=True) as client:
        if args.check_robots:
            parser = robots_for(client, args.check_robots, cache)
            allowed = parser.can_fetch(UA, args.check_robots)
            print(f"{args.check_robots}\n  allowed for this bot: {allowed}")
            print(f"  crawl-delay: {parser.crawl_delay(UA) or parser.crawl_delay('*') or '(none)'}")
            if not allowed:
                print("\n  → robots.txt disallows this path. Ask the publisher for permission or use"
                      "\n    an openly licensed source (see CORPUS-PLAN.md) instead of crawling it.")
            return 0

        urls = Path(args.urls).read_text(encoding="utf-8").split() if args.urls else \
            urls_from_sitemap(client, args.sitemap, args.limit)
        urls = [u for u in urls if u.startswith("http")][: args.limit]
        print(f"{len(urls)} URL(s) to consider")

        segments: list[Segment] = []
        skipped_robots = 0
        failed = 0
        for i, url in enumerate(urls, 1):
            parser = robots_for(client, url, cache)
            if not parser.can_fetch(UA, url):
                skipped_robots += 1
                print(f"  [{i}/{len(urls)}] robots.txt says no — skipped {url[:90]}")
                continue
            try:
                seg = fetch_article(client, url)
            except httpx.HTTPError as exc:
                failed += 1
                print(f"  [{i}/{len(urls)}] failed: {exc}")
                continue
            if seg:
                segments.append(seg)
                print(f"  [{i}/{len(urls)}] {len(seg.text.split()):>5} words  {url[:90]}")
            time.sleep(crawl_delay(parser, args.delay))

    kept, stats = process(segments, min_words=args.min_words, keep_english=args.include_english)
    if skipped_robots:
        stats["skipped_by_robots_txt"] = skipped_robots
    if failed:
        stats["fetch_failures"] = failed
    path = write_segments(args.out, kept)
    print_stats("Web crawl result", stats, path)
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
