"""
TP3 Web - Chat RAG con agente LangGraph + Pinecone + Groq
========================================================

Esta app reutiliza la interfaz de TP2, pero reemplaza la lógica de generación
por el agente base del notebook `Agentes_Langchain.ipynb`.
El agente:
- Parsea la consulta para detectar persona y motivo.
- Recupera contexto del CV correspondiente en Pinecone.
- Responde usando solo ese contexto con un modelo de Groq.

Ejecutar con:
    streamlit run TP3_web.py

Variables de entorno necesarias:
    GROQ_API_KEY
    PINECONE_API_KEY
"""

import json
import operator
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pinecone import Pinecone
from transformers import AutoModel

# Cargar variables de entorno
load_dotenv()

# ========================================
# CONSTANTES Y CONFIGURACIÓN
# ========================================
TOP_K_RESULTS = 3
DEFAULT_PERSONA = "rodrigo-mesa"
INDEX_BY_PERSON = {
    "rodrigo-mesa": os.environ.get("PINECONE_INDEX_RODRIGO", "rodrigo-mesa"),
    "danilo-reitano": os.environ.get("PINECONE_INDEX_DANILO", "danilo-reitano"),
    "juan-garcia": os.environ.get("PINECONE_INDEX_JUAN", "juan-garcia"),
}
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "vs_namespace")
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v2-base-es"
ANSWER_SYSTEM_PROMPT = (
    "Sos un asistente que responde en español usando solo el contexto recuperado de Pinecone "
    "sobre el CV de la persona indicada. Si la información no está en el contexto, decilo explícitamente."
    "Respondé unicamente lo que te consulten, no des información que no se te preguntó."
)
PARSER_PROMPT = (
    "Sos un agente encargado de identificar dos aspectos en las solicitudes de usuario: el nombre de la persona del "
    "que se solicita información, y lo que se desea encontrar de la persona. Esto es para identificar secciones del CV "
    "que se están consultando. Por ejemplo: si tengo la frase 'Quiero ver la experiencia y la educacion de Rodrigo Mesa', "
    'me tenes que responder {"motivo": "Quiero ver la experiencia y la educacion de Rodrigo Mesa","persona":"rodrigo-mesa"}. '
    "Siempre tu respuesta debe estar en formato JSON, con las claves motivo y persona. Las unicas personas validas son: "
    "'rodrigo-mesa', 'danilo-reitano' y 'juan-garcia'. Si no se aclara una persona, tomar como opción por defecto "
    "'rodrigo-mesa'. No completar con ningún texto adicional, tu unica respuesta valida es el JSON indicado."
)

# ========================================
# VALIDACIÓN DE CLAVES
# ========================================
groq_api_key = os.environ.get("GROQ_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
missing_keys = []
if not groq_api_key:
    missing_keys.append("GROQ_API_KEY")
if not pinecone_api_key:
    missing_keys.append("PINECONE_API_KEY")

if missing_keys:
    st.error(f"⚠️ Faltan variables de entorno: {', '.join(missing_keys)}")
    st.stop()


# ========================================
# RECURSOS CACHEADOS
# ========================================
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return AutoModel.from_pretrained(EMBEDDING_MODEL_ID, trust_remote_code=True)


@st.cache_resource(show_spinner=False)
def get_pinecone_indexes(api_key: str):
    client = Pinecone(api_key=api_key)
    available = set(client.list_indexes().names())
    missing = [
        f"{persona} -> {index_name}"
        for persona, index_name in INDEX_BY_PERSON.items()
        if index_name not in available
    ]
    index_map = {
        persona: client.Index(index_name)
        for persona, index_name in INDEX_BY_PERSON.items()
        if index_name in available
    }
    return index_map, missing


@st.cache_resource(show_spinner=False)
def get_chat_model(model_name: str, api_key: str):
    return ChatGroq(model=model_name, api_key=api_key)


# ========================================
# DEFINICIÓN DEL AGENTE (LangGraph)
# ========================================
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    parsed_query: Optional[Dict[str, Any]]
    context: Optional[str]
    selected_persona: Optional[str]
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    retrieval_debug: Optional[Dict[str, Any]]


class Agent:
    def __init__(
        self,
        model,
        parser_prompt: str = PARSER_PROMPT,
        system_prompt: str = ANSWER_SYSTEM_PROMPT,
        index_by_person: Optional[Dict[str, str]] = None,
        top_k: int = TOP_K_RESULTS,
        embedding_model=None,
        pinecone_indexes: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.system = system_prompt
        self.parser_prompt = parser_prompt
        self.default_persona = DEFAULT_PERSONA
        self.index_by_person = index_by_person or INDEX_BY_PERSON
        self.top_k = top_k

        self.embedding_model = embedding_model or AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID, trust_remote_code=True
        )

        self.pinecone_indexes = pinecone_indexes
        if self.pinecone_indexes is None:
            if not pinecone_api_key:
                raise RuntimeError("Falta la variable de entorno PINECONE_API_KEY.")
            client = Pinecone(api_key=pinecone_api_key)
            self.pinecone_indexes = {
                persona: client.Index(index_name)
                for persona, index_name in self.index_by_person.items()
            }

        graph = StateGraph(AgentState)
        graph.add_node("parser_llm", self.call_groq_parser)
        graph.add_node("call_pinecone", self.take_action)
        graph.add_node("answer_llm", self.call_groq_general)
        graph.add_edge("parser_llm", "call_pinecone")
        graph.add_edge("call_pinecone", "answer_llm")
        graph.add_edge("answer_llm", END)
        graph.set_entry_point("parser_llm")
        self.graph = graph.compile()

    def call_groq_parser(self, state: AgentState):
        messages = [SystemMessage(content=self.parser_prompt)] + state["messages"]
        message = self.model.invoke(messages)
        parsed = self._safe_extract_json(message.content)
        if parsed.get("persona") not in self.index_by_person:
            parsed["persona"] = self.default_persona
        if "motivo" not in parsed:
            parsed["motivo"] = ""
        return {"messages": [message], "parsed_query": parsed}

    def call_groq_general(self, state: AgentState):
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        messages = []
        if self.system:
            messages.append(SystemMessage(content=self.system))
        if state.get("context"):
            messages.append(SystemMessage(content=state["context"]))
        messages.extend(human_messages)
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        parsed = state.get("parsed_query") or {}
        persona = parsed.get("persona") or self.default_persona
        if persona not in self.index_by_person:
            persona = self.default_persona

        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        query_text = parsed.get("motivo") or (user_messages[-1].content if user_messages else "")

        contexts, retrieval_debug = self._retrieve_context(persona, query_text)
        context_prompt = self._format_context_for_prompt(persona, contexts)

        return {
            "context": context_prompt,
            "selected_persona": persona,
            "retrieved_chunks": contexts,
            "retrieval_debug": retrieval_debug,
        }

    def _safe_extract_json(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def _retrieve_context(self, persona: str, query: str):
        debug = {
            "persona": persona,
            "index": self.index_by_person.get(persona),
            "namespace": PINECONE_NAMESPACE or "(sin namespace)",
            "top_k": self.top_k,
            "query_text": query,
        }

        if not query:
            debug["error"] = "Consulta vacia"
            return [], debug

        index = self.pinecone_indexes.get(persona)
        if not index:
            debug["error"] = "Indice no encontrado para persona"
            return [], debug

        try:
            vector = self.embedding_model.encode(query)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            query_kwargs = {
                "vector": vector,
                "top_k": self.top_k,
                "include_metadata": True,
            }
            if PINECONE_NAMESPACE:
                query_kwargs["namespace"] = PINECONE_NAMESPACE

            response = index.query(**query_kwargs)
            matches = response.get("matches") if isinstance(response, dict) else getattr(response, "matches", [])

            raw_matches = []
            for match in matches or []:
                if isinstance(match, dict):
                    raw_matches.append(match)
                else:
                    raw_matches.append(
                        {
                            "id": getattr(match, "id", None),
                            "score": getattr(match, "score", None),
                            "values": getattr(match, "values", None),
                            "metadata": getattr(match, "metadata", None),
                        }
                    )

            contexts = []
            for match in matches or []:
                metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {}
                score = match.get("score") if isinstance(match, dict) else getattr(match, "score", None)
                section = metadata.get("seccion") or metadata.get("section") or metadata.get("title")
                if not section and isinstance(match, dict):
                    section = match.get("id")
                elif not section:
                    section = getattr(match, "id", None)
                contexts.append(
                    {
                        "persona": persona,
                        "section": section or "Seccion no identificada",
                        "content": metadata.get("content", ""),
                        "score": score,
                        "id": match.get("id") if isinstance(match, dict) else getattr(match, "id", None),
                    }
                )

            debug["match_count"] = len(contexts)
            debug["raw_matches"] = raw_matches
            return contexts, debug
        except Exception as exc:
            debug["error"] = str(exc)
            return [], debug

    def _format_context_for_prompt(self, persona: str, contexts):
        if not contexts:
            return f"Contexto recuperado desde Pinecone para {persona}: no se encontraron coincidencias relevantes."

        blocks = []
        for ctx in contexts:
            score_txt = ""
            if isinstance(ctx.get("score"), (int, float)):
                score_txt = f" (score: {ctx['score']:.3f})"
            blocks.append(f"{ctx.get('section')}{score_txt}\n{ctx.get('content')}")

        return f"Usa exclusivamente el siguiente contexto del CV de {persona}:\n" + "\n\n".join(blocks)


# ========================================
# PREPARAR RECURSOS
# ========================================
try:
    embedding_model = get_embedding_model()
    st.sidebar.success("✅ Modelo de embeddings cargado")
except Exception as exc:
    st.sidebar.error(f"❌ Error al cargar embeddings: {exc}")
    st.stop()

try:
    pinecone_indexes, missing_indexes = get_pinecone_indexes(pinecone_api_key)
    if missing_indexes:
        st.sidebar.warning("⚠️ Índices faltantes: " + ", ".join(missing_indexes))
    if not pinecone_indexes:
        st.sidebar.error("❌ No se encontraron índices de Pinecone disponibles.")
        st.stop()
    st.sidebar.success(f"✅ Pinecone conectado ({len(pinecone_indexes)} índice(s) disponible(s))")
except Exception as exc:
    st.sidebar.error(f"❌ Error al conectar con Pinecone: {exc}")
    st.stop()


# ========================================
# UI PRINCIPAL
# ========================================
st.title("TP3 - Chat con agente LangGraph sobre CVs")
st.caption("Usa Groq + Pinecone. Incluye trazas del agente y contexto recuperado para depurar.")

st.sidebar.markdown("### Configuración del modelo")
MODEL_ID = st.sidebar.selectbox(
    "Modelo de Groq",
    options=[
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ],
    index=0,
)

try:
    chat_model = get_chat_model(MODEL_ID, groq_api_key)
    st.sidebar.success("✅ Cliente Groq listo")
except Exception as exc:
    st.sidebar.error(f"❌ Error al crear cliente Groq: {exc}")
    st.stop()

agent = Agent(
    model=chat_model,
    parser_prompt=PARSER_PROMPT,
    system_prompt=ANSWER_SYSTEM_PROMPT,
    index_by_person=INDEX_BY_PERSON,
    top_k=TOP_K_RESULTS,
    embedding_model=embedding_model,
    pinecone_indexes=pinecone_indexes,
)

st.sidebar.markdown("### Índices por persona")
for persona, index_name in INDEX_BY_PERSON.items():
    st.sidebar.caption(f"{persona}: {index_name}")

# Sesiones para historial y debugging
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = {}

if st.sidebar.button("🗑️ Limpiar conversación"):
    st.session_state.conversation_history = []
    st.session_state.retrieved_context = []
    st.session_state.agent_trace = {}
    st.sidebar.success("✅ Conversación reiniciada")
    st.rerun()

# Estilos mínimos para la respuesta
st.markdown(
    """
<style>
.response-box {
    background-color: #f0f8ff;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    color: #0b2545;
}
</style>
""",
    unsafe_allow_html=True,
)


# ========================================
# FUNCIONES DE NEGOCIO
# ========================================
def generate_agent_response(user_text: str):
    try:
        result = agent.graph.invoke({"messages": [HumanMessage(content=user_text)]})
    except Exception as exc:
        st.sidebar.error(f"❌ Error al ejecutar el agente: {exc}")
        return f"Error al generar respuesta: {exc}"

    contexts = result.get("retrieved_chunks") or []
    parsed_query = result.get("parsed_query") or {}
    persona = result.get("selected_persona") or parsed_query.get("persona") or DEFAULT_PERSONA
    context_prompt = result.get("context")
    agent_messages = result.get("messages") or []
    retrieval_debug = result.get("retrieval_debug") or {}

    response_text = next(
        (getattr(msg, "content", "") for msg in reversed(agent_messages) if getattr(msg, "content", "")),
        "No se obtuvo respuesta del agente.",
    )

    st.session_state.retrieved_context = contexts
    st.session_state.agent_trace = {
        "parsed_query": parsed_query,
        "selected_persona": persona,
        "context_prompt": context_prompt,
        "messages": [
            {
                "role": getattr(msg, "type", msg.__class__.__name__),
                "content": getattr(msg, "content", str(msg)),
            }
            for msg in agent_messages
        ],
        "retrieved_chunks": contexts,
        "retrieval_debug": retrieval_debug,
    }
    st.session_state.conversation_history.append({"role": "user", "content": user_text})
    st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
    return response_text


# ========================================
# ENTRADA DEL USUARIO
# ========================================
st.markdown("### 💬 Escribe tu mensaje:")
user_input = st.text_input(
    "Usuario:",
    placeholder="Ejemplo: ¿Puedes resumir la educación de Juan García?",
    label_visibility="collapsed",
)
send_button = st.button("📤 Enviar", type="primary")


# ========================================
# PROCESAMIENTO
# ========================================
if user_input and (user_input.strip() or send_button):
    with st.spinner("🤔 Ejecutando agente..."):
        response = generate_agent_response(user_input)

    st.markdown("### 🤖 Respuesta del agente:")
    st.markdown(
        f"""
    <div class="response-box">
        {response}
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.retrieved_context:
        with st.expander(f"📎 Contexto recuperado desde Pinecone ({len(st.session_state.retrieved_context)} fragmentos)"):
            for idx, ctx in enumerate(st.session_state.retrieved_context, start=1):
                score = ctx.get("score")
                score_txt = f" | score: {score:.3f}" if isinstance(score, (int, float)) else ""
                st.markdown(f"**{idx}. {ctx.get('persona', 'persona-desconocida')} — {ctx.get('section')}**{score_txt}")
                if ctx.get("id"):
                    st.caption(f"ID en índice: {ctx['id']}")
                st.markdown(ctx.get("content") or "_Sin contenido disponible en el índice._")
                if idx < len(st.session_state.retrieved_context):
                    st.markdown("---")
    else:
        st.info("No se recuperó contexto desde Pinecone para esta consulta.")

    if st.session_state.agent_trace:
        with st.expander("🧠 Trazas del agente (debug)"):
            trace = st.session_state.agent_trace
            st.markdown(f"**Persona seleccionada:** {trace.get('selected_persona')}")
            st.markdown("**Consulta interpretada (parser):**")
            st.json(trace.get("parsed_query") or {})
            st.markdown("**Contexto enviado al LLM:**")
            st.code(trace.get("context_prompt") or "Sin contexto", language="markdown")
            st.markdown("**Mensajes de la corrida:**")
            st.json(trace.get("messages") or [])
            st.markdown("**Diagnóstico de Pinecone:**")
            st.json(trace.get("retrieval_debug") or {})
            raw_matches = (trace.get("retrieval_debug") or {}).get("raw_matches") or []
            if raw_matches:
                st.markdown("**Raw matches (Pinecone):**")
                st.json(raw_matches)

# ========================================
# HISTORIAL
# ========================================
if st.session_state.conversation_history:
    with st.expander(f"📜 Historial completo ({len(st.session_state.conversation_history)} mensajes)"):
        for i, message in enumerate(st.session_state.conversation_history):
            role = "👤 Usuario" if message["role"] == "user" else "🤖 Agente"
            st.markdown(f"**{role}**: {message['content']}")
            if i < len(st.session_state.conversation_history) - 1:
                st.markdown("---")
