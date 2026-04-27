"""Persistência de histórico de conversas em disco."""

import json
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "conversations"


class ConversationHistory:
    def __init__(self):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        return HISTORY_DIR / f"{agent_id}.json"

    def load(self, agent_id: str) -> list[dict]:
        path = self._path(agent_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("messages", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def save(self, agent_id: str, messages: list[dict]) -> None:
        path = self._path(agent_id)
        data = {
            "agent_id": agent_id,
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, agent_id: str, role: str, content: str) -> None:
        messages = self.load(agent_id)
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.save(agent_id, messages)

    def clear(self, agent_id: str) -> None:
        path = self._path(agent_id)
        if path.exists():
            path.unlink()

    def get_user_style(self, agent_id: str) -> str:
        """Analisa mensagens passadas para detectar o estilo de comunicação."""
        messages = self.load(agent_id)
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        if not user_msgs:
            return ""

        recent = user_msgs[-10:]
        text = " ".join(recent).lower()

        vulgar_words = [
            "cuzinho", "cu", "buceta", "pau", "rola", "caralho", "porra",
            "foder", "meter", "gozar", "safada", "putaria", "tesão",
            "chupar", "mamar", "trepar", "arrombad", "foda",
        ]
        formal_words = [
            "poderia", "gostaria", "orientação", "informação",
            "por favor", "obrigad", "gentileza",
        ]

        vulgar_count = sum(1 for w in vulgar_words if w in text)
        formal_count = sum(1 for w in formal_words if w in text)

        if vulgar_count >= 3:
            return (
                "[ESTILO DA USUÁRIA: linguagem direta e vulgar. "
                "Use o mesmo nível — gírias, palavrões, sem frescura. "
                "Ela quer respostas cruas e sem rodeios.]\n"
            )
        if vulgar_count >= 1:
            return (
                "[ESTILO DA USUÁRIA: linguagem informal e aberta. "
                "Use tom descontraído, pode usar gírias leves e linguagem coloquial.]\n"
            )
        if formal_count >= 2:
            return (
                "[ESTILO DA USUÁRIA: linguagem mais formal e educada. "
                "Use tom respeitoso e acolhedor, mas sem ser robótica.]\n"
            )
        return ""
