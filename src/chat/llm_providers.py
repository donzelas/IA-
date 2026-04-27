from abc import ABC, abstractmethod
import os

from openai import OpenAI


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str: ...


class OllamaProvider(LLMProvider):
    DEFAULT_MODEL = "llama3.1:8b"

    def __init__(self):
        self._client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=90.0,
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model or self.DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Ollama falhou: {e}") from e


class GroqProvider(LLMProvider):
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY não definida nas variáveis de ambiente")
        self._client = Groq(api_key=api_key)

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model or self.DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq falhou: {e}") from e


class GeminiProvider(LLMProvider):
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self):
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não definida nas variáveis de ambiente")
        genai.configure(api_key=api_key)
        self._genai = genai

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None,
    ) -> str:
        import time

        gemini_model = self._genai.GenerativeModel(
            model or self.DEFAULT_MODEL
        )

        history = []
        system_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_text += content + "\n"
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        if system_text and history:
            first_user = history[0]["parts"][0]
            history[0]["parts"][0] = f"{system_text.strip()}\n\n{first_user}"

        for attempt in range(3):
            try:
                response = gemini_model.generate_content(
                    history,
                    generation_config={"temperature": temperature},
                )
                return response.text
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "quota" in err or "429" in err or "resource" in err:
                    if attempt < 2:
                        time.sleep(10)
                        continue
                raise RuntimeError(f"Gemini falhou: {e}") from e
        raise RuntimeError("Gemini falhou após 3 tentativas")


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


def get_provider(provider_name: str) -> LLMProvider:
    provider_cls = _PROVIDERS.get(provider_name.lower())
    if provider_cls is None:
        available = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Provider '{provider_name}' não suportado. Disponíveis: {available}"
        )
    return provider_cls()
