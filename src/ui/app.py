"""Interface principal do sistema de IA multi-agente."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from src.agents.manager import AgentManager
from src.knowledge.embeddings import KnowledgeBase
from src.knowledge.loader import load_document, SUPPORTED_EXTENSIONS
from src.chat.engine import ChatEngine
from src.chat.history import ConversationHistory
from src.search.web_search import WebSearcher

st.set_page_config(
    page_title="IA Multi-Agente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_manager():
    return AgentManager()


@st.cache_resource
def get_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def get_chat_engine():
    return ChatEngine(knowledge_base=get_knowledge_base())


@st.cache_resource
def get_web_searcher():
    return WebSearcher()


@st.cache_resource
def get_conversation_history():
    return ConversationHistory()


def init_session():
    defaults = {
        "current_agent_id": None,
        "chat_history": {},
        "page": "chat",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar():
    manager = get_manager()

    with st.sidebar:
        st.title("🤖 IA Multi-Agente")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()
        with col2:
            if st.button("⚙️ Agentes", use_container_width=True):
                st.session_state.page = "manage"
                st.rerun()

        st.divider()
        st.subheader("Seus Agentes")

        agents = manager.list_agents()
        if not agents:
            st.info("Nenhum agente criado ainda. Vá em ⚙️ Agentes para criar.")
        else:
            for agent in agents:
                label = f"**{agent.name}**"
                is_selected = st.session_state.current_agent_id == agent.id
                if st.button(
                    f"{'▶ ' if is_selected else ''}{agent.name}",
                    key=f"agent_{agent.id}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state.current_agent_id = agent.id
                    st.session_state.page = "chat"
                    st.rerun()
                st.caption(agent.description[:80])


def render_chat_page():
    manager = get_manager()
    engine = get_chat_engine()
    searcher = get_web_searcher()
    conv_history = get_conversation_history()

    agent_id = st.session_state.current_agent_id
    if not agent_id:
        st.title("💬 Chat")
        st.info("Selecione um agente na barra lateral para começar a conversar.")
        return

    agent = manager.get_agent(agent_id)
    if not agent:
        st.error("Agente não encontrado.")
        st.session_state.current_agent_id = None
        return

    st.title(f"💬 {agent.name}")
    st.caption(f"{agent.description} | LLM: {agent.llm_provider}/{agent.llm_model}")

    if agent_id not in st.session_state.chat_history:
        saved = conv_history.load(agent_id)
        st.session_state.chat_history[agent_id] = [
            {"role": m["role"], "content": m["content"]} for m in saved
        ]

    history = st.session_state.chat_history[agent_id]

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(f"Fale com {agent.name}..."):
        history.append({"role": "user", "content": prompt})
        conv_history.append(agent_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                search_results = None
                if agent.web_search_enabled:
                    with st.status("🔍 Buscando na internet...", expanded=False):
                        from src.chat.engine import _is_explicit
                        is_nsfw = _is_explicit(prompt)
                        search_results = searcher.search_and_summarize(
                            prompt,
                            safe_search=not is_nsfw,
                            include_videos=is_nsfw,
                        )
                        if search_results and search_results != "Nenhum resultado encontrado.":
                            st.write("Resultados encontrados!")
                        else:
                            search_results = None
                            st.write("Sem resultados relevantes.")

                chat_msgs = [
                    {"role": m["role"], "content": m["content"]}
                    for m in history[:-1]
                ]

                user_style = conv_history.get_user_style(agent_id)

                agent_config = {
                    "system_prompt": agent.system_prompt,
                    "llm_provider": agent.llm_provider,
                    "llm_model": agent.llm_model,
                    "temperature": agent.temperature,
                    "web_search_enabled": agent.web_search_enabled,
                    "user_style": user_style,
                }

                try:
                    response = engine.chat(
                        agent_id=agent.id,
                        agent_config=agent_config,
                        user_message=prompt,
                        chat_history=chat_msgs if chat_msgs else None,
                        search_results=search_results,
                    )
                except Exception as e:
                    response = f"❌ Erro: {e}\n\nVerifique se o Ollama está rodando ou se as API keys estão configuradas no `.env`."

                st.markdown(response)
                history.append({"role": "assistant", "content": response})
                conv_history.append(agent_id, "assistant", response)

    col1, col2 = st.columns([1, 1])
    with col1:
        if history and st.button("🗑️ Limpar conversa"):
            st.session_state.chat_history[agent_id] = []
            conv_history.clear(agent_id)
            st.rerun()


def render_manage_page():
    manager = get_manager()
    kb = get_knowledge_base()

    st.title("⚙️ Gerenciar Agentes")

    tab_create, tab_edit, tab_docs = st.tabs([
        "➕ Criar Agente", "✏️ Editar/Excluir", "📄 Documentos"
    ])

    with tab_create:
        render_create_agent(manager)

    with tab_edit:
        render_edit_agents(manager, kb)

    with tab_docs:
        render_documents(manager, kb)


def render_create_agent(manager: AgentManager):
    st.subheader("Criar Novo Agente")

    with st.form("create_agent", clear_on_submit=True):
        name = st.text_input("Nome do Agente", placeholder="Ex: Conselheira para Mulheres")
        description = st.text_area(
            "Descrição",
            placeholder="Ex: IA especializada em acolhimento e orientação para mulheres",
            height=80,
        )
        system_prompt = st.text_area(
            "Instruções (System Prompt)",
            placeholder="Descreva em detalhes como a IA deve se comportar, o que ela sabe, como deve responder...",
            height=200,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            llm_provider = st.selectbox(
                "Provider LLM",
                ["ollama", "groq", "gemini"],
                help="Ollama = local/grátis | Groq = API grátis | Gemini = API grátis",
            )
        with col2:
            model_options = {
                "ollama": ["llama3.1:8b", "mistral", "gemma2", "phi3"],
                "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            }
            llm_model = st.selectbox("Modelo", model_options.get(llm_provider, []))
        with col3:
            temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, 0.1)

        web_search = st.toggle("🔍 Busca na internet", value=True)

        submitted = st.form_submit_button("✅ Criar Agente", use_container_width=True)

        if submitted:
            if not name or not system_prompt:
                st.error("Nome e Instruções são obrigatórios!")
            else:
                agent = manager.create_agent(
                    name=name,
                    description=description or name,
                    system_prompt=system_prompt,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    temperature=temperature,
                    web_search_enabled=web_search,
                )
                st.success(f"Agente '{agent.name}' criado com sucesso!")
                st.cache_resource.clear()
                st.rerun()


def render_edit_agents(manager: AgentManager, kb: KnowledgeBase):
    agents = manager.list_agents()
    if not agents:
        st.info("Nenhum agente para editar.")
        return

    for agent in agents:
        with st.expander(f"🤖 {agent.name}", expanded=False):
            with st.form(f"edit_{agent.id}"):
                name = st.text_input("Nome", value=agent.name, key=f"name_{agent.id}")
                description = st.text_area(
                    "Descrição", value=agent.description, key=f"desc_{agent.id}", height=60
                )
                system_prompt = st.text_area(
                    "Instruções", value=agent.system_prompt, key=f"sp_{agent.id}", height=150
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    providers = ["ollama", "groq", "gemini"]
                    provider_idx = providers.index(agent.llm_provider) if agent.llm_provider in providers else 0
                    llm_provider = st.selectbox(
                        "Provider", providers, index=provider_idx, key=f"prov_{agent.id}"
                    )
                with col2:
                    llm_model = st.text_input("Modelo", value=agent.llm_model, key=f"model_{agent.id}")
                with col3:
                    temperature = st.slider(
                        "Temperatura", 0.0, 1.5, agent.temperature, 0.1, key=f"temp_{agent.id}"
                    )

                web_search = st.toggle(
                    "Busca na internet", value=agent.web_search_enabled, key=f"web_{agent.id}"
                )

                col_save, col_del = st.columns(2)
                with col_save:
                    if st.form_submit_button("💾 Salvar", use_container_width=True):
                        manager.update_agent(
                            agent.id,
                            name=name,
                            description=description,
                            system_prompt=system_prompt,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            temperature=temperature,
                            web_search_enabled=web_search,
                        )
                        st.success("Atualizado!")
                        st.cache_resource.clear()
                        st.rerun()
                with col_del:
                    if st.form_submit_button("🗑️ Excluir", use_container_width=True):
                        try:
                            kb.delete_collection(agent.id)
                        except Exception:
                            pass
                        manager.delete_agent(agent.id)
                        st.success("Excluído!")
                        st.cache_resource.clear()
                        st.rerun()


def render_documents(manager: AgentManager, kb: KnowledgeBase):
    agents = manager.list_agents()
    if not agents:
        st.info("Crie um agente primeiro para adicionar documentos.")
        return

    agent_names = {a.id: a.name for a in agents}
    selected_id = st.selectbox(
        "Selecione o Agente",
        options=list(agent_names.keys()),
        format_func=lambda x: agent_names[x],
    )

    if not selected_id:
        return

    st.subheader(f"📄 Documentos de: {agent_names[selected_id]}")

    extensions = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader(
        "Faça upload de documentos",
        type=extensions,
        accept_multiple_files=True,
        key=f"upload_{selected_id}",
    )

    if uploaded_files and st.button("📤 Processar documentos", use_container_width=True):
        for uploaded_file in uploaded_files:
            with st.status(f"Processando {uploaded_file.name}..."):
                file_path = DOCS_DIR / f"{selected_id}_{uploaded_file.name}"
                file_path.write_bytes(uploaded_file.getvalue())
                st.write("Arquivo salvo.")

                chunks = load_document(file_path)
                st.write(f"{len(chunks)} trechos extraídos.")

                metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]
                kb.add_documents(selected_id, chunks, metadatas)
                st.write("Adicionado à base de conhecimento!")

        st.success(f"{len(uploaded_files)} documento(s) processado(s)!")
        st.cache_resource.clear()

    st.divider()
    st.subheader("Testar base de conhecimento")
    test_query = st.text_input("Digite uma pergunta para testar a busca:", key="test_q")
    if test_query and st.button("🔎 Buscar", key="test_btn"):
        results = kb.query(selected_id, test_query)
        if results:
            for i, r in enumerate(results, 1):
                st.markdown(f"**Resultado {i}** (distância: {r['distance']:.4f})")
                st.text(r["content"][:500])
                st.divider()
        else:
            st.warning("Nenhum documento encontrado. Faça upload de documentos primeiro.")


def main():
    init_session()
    render_sidebar()

    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "manage":
        render_manage_page()


if __name__ == "__main__":
    main()
