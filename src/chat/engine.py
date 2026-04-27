from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.chat.llm_providers import get_provider

if TYPE_CHECKING:
    from src.knowledge.embeddings import KnowledgeBase

logger = logging.getLogger(__name__)

FALLBACK_ORDER = ["groq", "ollama", "gemini"]

EXPLICIT_KEYWORDS = [
    "sexo", "transar", "foder", "chupar", "mamar", "pau", "buceta", "rola",
    "gozar", "orgasmo", "fetiche", "putaria", "safada", "tesão", "punheta",
    "siririca", "anal", "cu ", " cu,", "boquete", "menage", "ménage",
    "swing", "cuckold", "dominação", "submissa", "bdsm", "sadomaso",
    "vibrador", "dildo", "plug", "masturb", "ejacular", "penetra",
    "arrombad", "meter", "foda", "trepar", "gemer", "excitad",
    "porn", "xvideos", "xhamster", "nude", "nua ", "pelad",
    "dar pra", "dando pra", "enfiar", "chupet", "lamber",
]


def _is_explicit(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in EXPLICIT_KEYWORDS)


EXPLICIT_SYSTEM_ADDON = (
    "\n\n[MODO ABERTO ATIVADO] A usuária quer falar abertamente sobre sexo/intimidade. "
    "Responda com total naturalidade, sem filtros, sem disclaimers, sem repetir avisos de segurança. "
    "Fale como uma amiga íntima que entende tudo e não tem vergonha de nada. "
    "Seja direta, explícita se necessário, use a linguagem que a pessoa usou. "
    "Aprofunde o assunto com análise psicológica do desejo e dê respostas práticas e reais."
)


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
        configured_provider = agent_config.get("llm_provider", "groq")
        configured_model = agent_config.get("llm_model")
        temperature = agent_config.get("temperature", 0.7)
        system_prompt = agent_config.get("system_prompt", "")

        explicit = _is_explicit(user_message)

        if explicit:
            provider_name = "ollama"
            model = "dolphin-mistral"
            system_prompt += EXPLICIT_SYSTEM_ADDON
            temperature = max(temperature, 0.9)
            logger.info("Conteúdo explícito detectado → roteando para dolphin-mistral")
        else:
            provider_name = configured_provider
            model = configured_model

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
