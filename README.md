# IA Multi-Agente

Sistema de IA personalizável onde você cria agentes especializados, alimenta com documentos e conversa com eles. Cada agente tem sua própria personalidade, base de conhecimento e acesso à internet.

## Funcionalidades

- **Multi-agente**: crie quantos agentes quiser, cada um com personalidade e conhecimento próprio
- **Multi-LLM**: usa Ollama (local/grátis), Groq e Gemini com fallback automático
- **RAG**: faça upload de PDFs, TXT, DOCX e a IA aprende com eles
- **Busca web**: agentes buscam informações atuais na internet automaticamente
- **Interface web**: painel completo com chat, gerenciamento de agentes e upload de documentos

## Setup Rápido

### 1. Instalar dependências Python

```bash
pip install -r requirements.txt
```

### 2. Instalar Ollama (LLM local e grátis)

Baixe em [ollama.com](https://ollama.com) e depois:

```bash
ollama pull llama3.1:8b
```

### 3. Configurar API keys (opcional, para Groq e Gemini)

```bash
cp .env.example .env
```

Edite o `.env` com suas chaves:
- Groq: [console.groq.com/keys](https://console.groq.com/keys)
- Gemini: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 4. Rodar

```bash
streamlit run src/ui/app.py
```

## Estrutura

```
├── src/
│   ├── agents/         # Gerenciamento de agentes
│   ├── chat/           # Motor de chat multi-LLM
│   ├── knowledge/      # RAG (embeddings + documentos)
│   ├── search/         # Busca na internet
│   └── ui/             # Interface Streamlit
├── data/
│   ├── agents/         # Configs dos agentes (JSON)
│   ├── documents/      # Documentos uploadados
│   └── chroma_db/      # Banco vetorial
├── .env.example
├── requirements.txt
└── README.md
```
