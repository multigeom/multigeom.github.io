"""
Check the static landing page: every local file referenced by an HTML page (links, images, videos, stylesheets, scripts, posters)
exists, the title is present, and the page has no empty links. Run from the repository root:  python scripts/check_site.py
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


class Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.title, self._in_title = [], "", False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        for key in ("href", "src", "poster"):
            if key in a:
                self.refs.append((tag, key, a[key]))
        if tag == "meta" and a.get("property") == "og:image":
            self.refs.append((tag, "content", a.get("content", "")))

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def main() -> int:
    problems = []
    pages = sorted(ROOT.glob("*.html"))
    if not pages:
        problems.append("no HTML page found")
    for page in pages:
        parser = Refs()
        parser.feed(page.read_text())
        if not parser.title.strip():
            problems.append(f"{page.name}: no <title>")
        for tag, key, ref in parser.refs:
            if not ref.strip() or ref.strip() == "#":
                problems.append(f"{page.name}: empty {key} on <{tag}>")
                continue
            parsed = urlparse(ref)
            if parsed.scheme or ref.startswith(("//", "mailto:", "#")):
                continue                                              # external or in-page: not checked here
            if not (ROOT / parsed.path).exists():
                problems.append(f"{page.name}: <{tag} {key}={ref!r}> does not exist")
    for p in problems:
        print("PROBLEM:", p)
    print(f"checked {len(pages)} page(s): {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
