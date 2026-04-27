from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.chat.llm_providers import get_provider

if TYPE_CHECKING:
    from src.knowledge.embeddings import KnowledgeBase

logger = logging.getLogger(__name__)

FALLBACK_ORDER = ["ollama", "groq", "gemini"]


class ChatEngine:
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self._knowledge_base = knowledge_base

    def chat(
        self,
        agent_id: str,
        agent_config: dict,
        user_message: str,
        chat_history: list[dict] | None = None,
        search_results: str | None = None,
    ) -> str:
        provider_name = agent_config.get("llm_provider", "ollama")
        model = agent_config.get("llm_model")
        temperature = agent_config.get("temperature", 0.7)
        system_prompt = agent_config.get("system_prompt", "")

        context = self._build_context(agent_id, user_message, search_results)

        messages: list[dict] = []

        system_content = system_prompt
        if context:
            system_content += (
                f"\n\n--- Contexto Relevante ---\n{context}\n--- Fim do Contexto ---"
            )
        if system_content.strip():
            messages.append({"role": "system", "content": system_content})

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": user_message})

        providers_to_try = [provider_name] + [
            p for p in FALLBACK_ORDER if p != provider_name
        ]

        last_error: Exception | None = None
        for name in providers_to_try:
            try:
                provider = get_provider(name)
                current_model = model if name == provider_name else None
                return provider.generate(
                    messages, temperature=temperature, model=current_model
                )
            except Exception as e:
                logger.warning("Provider '%s' falhou: %s", name, e)
                last_error = e

        raise RuntimeError(
            f"Todos os providers falharam. Último erro: {last_error}"
        )

    def _build_context(
        self,
        agent_id: str,
        query: str,
        search_results: str | None = None,
    ) -> str:
        parts: list[str] = []

        if self._knowledge_base:
            try:
                docs = self._knowledge_base.query(agent_id, query)
                if docs:
                    rag_text = "\n\n".join(
                        doc["content"] for doc in docs if doc.get("content")
                    )
                    if rag_text:
                        parts.append(f"[Base de Conhecimento]\n{rag_text}")
            except Exception as e:
                logger.warning("Erro ao consultar knowledge base: %s", e)

        if search_results:
            parts.append(f"[Resultados da Web]\n{search_results}")

        return "\n\n".join(parts)
