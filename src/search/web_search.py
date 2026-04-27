from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


class WebSearcher:

    def search(self, query: str, max_results: int = 5, safe_search: bool = True) -> list[dict]:
        try:
            safesearch = "moderate" if safe_search else "off"
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, safesearch=safesearch))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
        except Exception:
            return []

    def search_videos(self, query: str, max_results: int = 5, safe_search: bool = True) -> list[dict]:
        try:
            safesearch = "moderate" if safe_search else "off"
            with DDGS() as ddgs:
                results = list(ddgs.videos(query, max_results=max_results, safesearch=safesearch))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("content", ""),
                    "thumbnail": r.get("images", {}).get("large", "") if isinstance(r.get("images"), dict) else "",
                    "duration": r.get("duration", ""),
                    "publisher": r.get("publisher", ""),
                    "snippet": r.get("description", ""),
                }
                for r in results
            ]
        except Exception:
            return []

    def scrape_url(self, url: str, max_chars: int = 3000) -> str:
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (compatible; WebSearcher/1.0)"
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            return clean_text[:max_chars]
        except Exception:
            return ""

    def search_and_summarize(self, query: str, max_results: int = 3, safe_search: bool = True, include_videos: bool = False) -> str:
        parts = []

        results = self.search(query, max_results=max_results, safe_search=safe_search)
        if results:
            for i, result in enumerate(results, 1):
                content = self.scrape_url(result["url"])
                if not content:
                    content = result["snippet"]
                parts.append(
                    f"--- Resultado {i} ---\n"
                    f"Título: {result['title']}\n"
                    f"URL: {result['url']}\n"
                    f"Conteúdo: {content}"
                )

        if include_videos:
            videos = self.search_videos(query, max_results=max_results, safe_search=safe_search)
            if videos:
                parts.append("\n--- VÍDEOS ENCONTRADOS ---")
                for i, video in enumerate(videos, 1):
                    entry = f"Vídeo {i}: {video['title']}\n  Link: {video['url']}"
                    if video.get("duration"):
                        entry += f"\n  Duração: {video['duration']}"
                    if video.get("thumbnail"):
                        entry += f"\n  Preview: {video['thumbnail']}"
                    if video.get("publisher"):
                        entry += f"\n  Fonte: {video['publisher']}"
                    parts.append(entry)

        if not parts:
            return "Nenhum resultado encontrado."

        return "\n\n".join(parts)
