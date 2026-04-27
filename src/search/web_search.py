from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


class WebSearcher:

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
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

    def search_and_summarize(self, query: str, max_results: int = 3) -> str:
        results = self.search(query, max_results=max_results)
        if not results:
            return "Nenhum resultado encontrado."

        parts = []
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

        return "\n\n".join(parts)
