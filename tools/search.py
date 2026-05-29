#!/usr/bin/env python3
"""Wikipedia and web search helpers for Homie tools."""

from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

import config


WIKIPEDIA_TIMEOUT = getattr(config, "WIKIPEDIA_TIMEOUT", 5)
WEB_SEARCH_TIMEOUT = getattr(config, "WEB_SEARCH_TIMEOUT", 10)
WEB_SEARCH_DEFAULT_RESULTS = getattr(config, "WEB_SEARCH_DEFAULT_RESULTS", 3)


class _DuckDuckGoResultsParser(HTMLParser):
	"""Parse DuckDuckGo HTML result pages into title/url/snippet tuples."""

	def __init__(self) -> None:
		super().__init__()
		self.results = []
		self._current = None
		self._capture_title = False
		self._capture_snippet = False

	def handle_starttag(self, tag, attrs):
		attrs_dict = dict(attrs)
		class_name = attrs_dict.get("class", "")

		if tag == "a" and "result__a" in class_name:
			self._flush_current()
			href = attrs_dict.get("href", "")
			self._current = {"title": "", "url": _normalize_duckduckgo_url(href), "snippet": ""}
			self._capture_title = True
		elif self._current and tag in {"a", "div"} and (
			"result__snippet" in class_name or "result-snippet" in class_name
		):
			self._capture_snippet = True

	def handle_endtag(self, tag):
		if tag == "a":
			self._capture_title = False
		elif tag == "div":
			self._capture_snippet = False
			self._flush_current()

	def handle_data(self, data):
		text = " ".join(data.split())
		if not text or not self._current:
			return

		if self._capture_title:
			self._current["title"] += text + " "
		elif self._capture_snippet:
			self._current["snippet"] += text + " "

	def close(self):
		super().close()
		self._flush_current()

	def _flush_current(self):
		if self._current and self._current["title"]:
			self.results.append(
				{
					"title": self._current["title"].strip(),
					"url": self._current["url"].strip(),
					"snippet": self._current["snippet"].strip(),
				}
			)
		self._current = None
		self._capture_title = False
		self._capture_snippet = False


def _http_get(url: str, timeout: int) -> str:
	request = urllib.request.Request(
		url,
		headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Homie/1.0"},
	)
	with urllib.request.urlopen(request, timeout=timeout) as response:
		return response.read().decode("utf-8", errors="replace")


def _normalize_duckduckgo_url(url: str) -> str:
	if not url:
		return ""

	parsed = urllib.parse.urlparse(url)
	if parsed.netloc.endswith("duckduckgo.com"):
		query = urllib.parse.parse_qs(parsed.query)
		target = query.get("uddg", [""])[0]
		if target:
			return urllib.parse.unquote(target)
	return urllib.parse.unquote(url)


def _search_wikipedia_title(query: str) -> str:
	params = urllib.parse.urlencode(
		{
			"action": "query",
			"list": "search",
			"srsearch": query,
			"format": "json",
			"srlimit": 1,
		}
	)
	body = _http_get(f"https://en.wikipedia.org/w/api.php?{params}", WIKIPEDIA_TIMEOUT)
	data = json.loads(body)
	results = data.get("query", {}).get("search", [])
	return results[0]["title"] if results else ""


def search_wikipedia(query: str, max_sentences: int = 5) -> str:
	"""Search Wikipedia for compact background knowledge."""
	query = query.strip()
	if not query:
		return "Wikipedia search needs a topic to look up."

	try:
		page_title = _search_wikipedia_title(query)
		if not page_title:
			return f"No Wikipedia article found for '{query}'."

		encoded_title = urllib.parse.quote(page_title)
		body = _http_get(
			f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}",
			WIKIPEDIA_TIMEOUT,
		)
		data = json.loads(body)
		summary = data.get("extract", "").strip()
		if not summary:
			return f"Wikipedia has a page for '{page_title}', but no summary was available."

		sentences = re.split(r"(?<=[.!?])\s+", summary)
		limited_summary = " ".join(sentences[: max(1, max_sentences)]).strip()
		return f"Wikipedia: {page_title}. {limited_summary}"
	except Exception as exc:
		print(f"[ERROR] Wikipedia search failed for '{query}': {exc}", file=sys.stderr)
		return f"Wikipedia search failed for '{query}'."


def search_web(query: str, num_results: int = WEB_SEARCH_DEFAULT_RESULTS) -> str:
	"""Search the web for current or recent information."""
	query = query.strip()
	if not query:
		return "Web search needs a query to look up."

	try:
		cleaned_results = _search_google_news(query, num_results)
		if not cleaned_results:
			cleaned_results = _search_duckduckgo_fallback(query, num_results)

		if not cleaned_results:
			return f"No web results found for '{query}'."

		lines = [f"Web results for '{query}':"]
		for index, result in enumerate(cleaned_results, start=1):
			entry = f"{index}. {result['title']} - {result['url']}"
			if result["snippet"]:
				entry += f" - {result['snippet']}"
			lines.append(entry)
		return "\n".join(lines)
	except Exception as exc:
		print(f"[ERROR] Web search failed for '{query}': {exc}", file=sys.stderr)
		return f"Web search failed for '{query}'."


def _search_google_news(query: str, num_results: int):
	encoded_query = urllib.parse.quote_plus(query)
	body = _http_get(
		f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en",
		WEB_SEARCH_TIMEOUT,
	)
	root = ET.fromstring(body)
	items = root.findall("./channel/item")

	results = []
	for item in items[: max(1, num_results)]:
		title = (item.findtext("title") or "").strip()
		url = (item.findtext("link") or "").strip()
		if title and url:
			results.append({"title": title, "url": url, "snippet": ""})
	return results


def _search_duckduckgo_fallback(query: str, num_results: int):
	encoded_query = urllib.parse.quote_plus(query)
	body = _http_get(f"https://html.duckduckgo.com/html/?q={encoded_query}", WEB_SEARCH_TIMEOUT)
	parser = _DuckDuckGoResultsParser()
	parser.feed(body)
	parser.close()

	results = []
	for result in parser.results:
		title = html.unescape(result["title"]).strip()
		url = result["url"].strip()
		snippet = html.unescape(result["snippet"]).strip()
		if title and url and url.startswith("http"):
			results.append({"title": title, "url": url, "snippet": snippet})
		if len(results) >= max(1, num_results):
			break
	return results
