"""
Chatbot RAG con Groq + Pinecone
===============================

Este archivo implementa un chatbot que combina:
- Recuperación de contexto desde Pinecone (index `cv-search`, namespace `vs_namespace`)
- Embeddings en español con `jinaai/jina-embeddings-v2-base-es`
- Respuestas generadas con Groq usando solo la pregunta actual + contexto recuperado
- Sin memoria entre turnos: cada pregunta hace una nueva búsqueda en Pinecone
- Interfaz simple con Streamlit

Instrucciones para ejecutar:
    streamlit run chatbot_simulacion_contexto.py

Variables de entorno necesarias:
    GROQ_API_KEY: Clave API de Groq
    PINECONE_API_KEY: Clave API de Pinecone
"""

# ========================================
# IMPORTACIÓN DE LIBRERÍAS
# ========================================

import streamlit as st    # Framework para interfaz web
import os                # Para acceder a variables de entorno
from groq import Groq    # Cliente directo de Groq (sin LangChain)
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoModel
load_dotenv()
# ========================================
# CONFIGURACIÓN INICIAL Y AUTENTICACIÓN
# ========================================

EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v2-base-es"
PINECONE_INDEX_NAME = "cv-search"
PINECONE_NAMESPACE = "vs_namespace"
TOP_K_RESULTS = 7
SYSTEM_PROMPT = (
    "Eres un asistente que responde en español usando el contexto recuperado de CVs. "
    "Si la información no aparece en el contexto, aclara que no está disponible."
)

# Obtener claves API desde variables de entorno
groq_api_key = os.environ.get("GROQ_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

missing_keys = []
if not groq_api_key:
    missing_keys.append("GROQ_API_KEY")
if not pinecone_api_key:
    missing_keys.append("PINECONE_API_KEY")

if missing_keys:
    st.error(f"⚠️ Faltan variables de entorno: {', '.join(missing_keys)}")
    st.info("💡 Configura las claves necesarias antes de continuar.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_pinecone_index(api_key: str):
    pc = Pinecone(api_key=api_key)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise RuntimeError(f"El índice {PINECONE_INDEX_NAME} no existe en Pinecone")
    return pc.Index(PINECONE_INDEX_NAME)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return AutoModel.from_pretrained(EMBEDDING_MODEL_ID, trust_remote_code=True)


# Crear clientes externos
try:
    client = get_groq_client(groq_api_key)
    st.sidebar.success("✅ Cliente Groq conectado exitosamente")
except Exception as e:
    st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
    st.stop()

try:
    pinecone_index = get_pinecone_index(pinecone_api_key)
    st.sidebar.success(f"✅ Pinecone conectado (índice: {PINECONE_INDEX_NAME})")
except Exception as e:
    st.sidebar.error(f"❌ Error al conectar con Pinecone: {str(e)}")
    st.stop()

try:
    embedding_model = get_embedding_model()
    st.sidebar.success("✅ Modelo de embeddings cargado")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar el modelo de embeddings: {str(e)}")
    st.stop()

# ========================================
# UTILIDADES PARA RAG
# ========================================
def retrieve_context(query: str, top_k: int = TOP_K_RESULTS):
    """
    Recupera las secciones más relevantes desde Pinecone y devuelve el contenido asociado.
    """
    try:
        vector = embedding_model.encode(query)
        if hasattr(vector, "tolist"):  # Asegurar formato serializable para Pinecone
            vector = vector.tolist()
        response = pinecone_index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        matches = response.get("matches") if isinstance(response, dict) else getattr(response, "matches", [])
        contexts = []

        for match in matches or []:
            metadata = match.get("metadata", {}) or {}
            section_id = match.get("id")
            content = metadata.get("content", "")

            cv_name = metadata.get("nombre_cv")
            if not cv_name and section_id and "_" in section_id:
                cv_name = section_id.split("_")[0]

            section_title = metadata.get("seccion") or section_id

            contexts.append(
                {
                    "cv_name": cv_name or "desconocido",
                    "section": section_title,
                    "content": content,
                    "score": match.get("score"),
                }
            )

        return contexts
    except Exception as e:
        st.sidebar.error(f"❌ Error al recuperar contexto: {e}")
        return []


def format_context_for_prompt(contexts):
    if not contexts:
        return ""

    blocks = []
    for ctx in contexts:
        header = f"CV: {ctx['cv_name']} | Sección: {ctx['section']}"
        body = ctx["content"] or "Contenido no disponible en el índice."
        blocks.append(f"{header}\n{body}")

    return "Contexto recuperado desde Pinecone:\n" + "\n\n".join(blocks)

# ========================================
# GESTIÓN DE MEMORIA CONVERSACIONAL
# ========================================

# Inicializar el historial de conversación en el estado de la sesión de Streamlit
# st.session_state permite mantener datos entre ejecuciones de la aplicación.
# Se usa solo para visualización; el modelo no recibe historial previo.
if "conversation_history" not in st.session_state:
    # Formato de lista de diccionarios compatible con la API de Groq
    # Cada mensaje tiene: {"role": "user"/"assistant", "content": "texto"}
    st.session_state.conversation_history = []
    st.sidebar.info("💬 Nueva conversación iniciada")
else:
    # Mostrar información del historial actual
    num_mensajes = len(st.session_state.conversation_history)
    st.sidebar.info(f"💬 Conversación activa: {num_mensajes} mensajes")

if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []

# Botón para limpiar el historial de conversación
if st.sidebar.button("🗑️ Limpiar Conversación"):
    st.session_state.conversation_history = []
    st.session_state.retrieved_context = []
    st.sidebar.success("✅ Conversación reiniciada")
    st.rerun()  # Recargar la aplicación


def generate_response(input_text):
    """
    Genera una respuesta usando RAG (Pinecone + Groq) y mantiene el historial.
    """

    try:
        # Paso 1: Recuperar contexto desde Pinecone (siempre fresh, sin historial)
        retrieved_context = retrieve_context(input_text, top_k=TOP_K_RESULTS)
        st.session_state.retrieved_context = retrieved_context
        context_prompt = format_context_for_prompt(retrieved_context)

        # Paso 2: Preparar mensajes para la API (solo turno actual + contexto)
        messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context_prompt:
            messages_for_api.append({"role": "system", "content": context_prompt})
        messages_for_api.append({"role": "user", "content": input_text})

        st.sidebar.caption("📤 Consulta sin historial previo (contexto recuperado de Pinecone)")

        # Paso 3: Llamada a Groq
        chat_completion = client.chat.completions.create(
            messages=messages_for_api,
            model=MODEL_ID,
            temperature=0.5,
            max_tokens=800,
            top_p=0.9,
        )

        response = chat_completion.choices[0].message.content

        # Paso 4: Guardar interacción solo para visualización (no se usa como contexto)
        st.session_state.conversation_history.append({"role": "user", "content": input_text})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        st.sidebar.success(f"✅ Respuesta generada ({len(response)} caracteres)")

        return response

    except Exception as e:
        error_msg = f"Error al generar respuesta: {str(e)}"
        st.sidebar.error(f"❌ {error_msg}")

        return f"Lo siento, ocurrió un error: {error_msg}"

# ========================================
# CONFIGURACIÓN DE LA INTERFAZ PRINCIPAL
# ========================================

# Configurar el título y descripción de la aplicación
st.title("Chatbot RAG sobre CVs (Pinecone + Groq)")

# Información del modelo en la barra lateral
st.sidebar.markdown("### Configuración del Modelo")
MODEL_ID = st.sidebar.selectbox(
    "Modelo de lenguaje",
    options=[
        "llama-3.1-8b-instant",   # Reemplazo recomendado para 8B
        "llama-3.3-70b-versatile" # Reemplazo recomendado para 70B
    ],
    index=0,
    help="Modelos recomendados por Groq (no deprecados)."
)
_model_info = {
    "llama-3.1-8b-instant": "🦙 Llama 3.1 8B Instant: excelente precio-rendimiento y baja latencia",
    "llama-3.3-70b-versatile": "🦙 Llama 3.3 70B Versatile: mayor calidad general",
}
st.sidebar.info(_model_info.get(MODEL_ID, "Modelo seleccionado"))

# ========================================
# INTERFAZ DE ENTRADA DEL USUARIO
# ========================================

# Ajustes de estilo para garantizar contraste en temas claros/oscursos
st.markdown("""
<style>
.response-box {
    background-color: #f0f8ff;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    color: #0b2545; /* texto oscuro para buen contraste */
}
</style>
""", unsafe_allow_html=True)

# Campo de entrada para el usuario
st.markdown("### 💬 Escribe tu mensaje:")
user_input = st.text_input(
    "Usuario:",
    placeholder="Ejemplo: Hola, ¿cómo estás? ¿De qué hablamos antes?",
    label_visibility="collapsed"
)

# Botón adicional para enviar
col1, col2 = st.columns([4, 1])
with col2:
    send_button = st.button("📤 Enviar", type="primary")

# ========================================
# PROCESAMIENTO Y VISUALIZACIÓN
# ========================================

# Procesar la entrada del usuario
if user_input and (user_input.strip() or send_button):
    # Mostrar indicador de carga
    with st.spinner('🤔 Generando respuesta...'):
        response = generate_response(user_input)
    
    # Mostrar la respuesta
    st.markdown("### 🤖 Respuesta del Chatbot:")
    st.markdown(f"""
    <div class="response-box">
        {response}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.retrieved_context:
        with st.expander(f"📎 Contexto utilizado ({len(st.session_state.retrieved_context)} resultados)"):
            for idx, ctx in enumerate(st.session_state.retrieved_context, start=1):
                score = ctx.get("score")
                score_txt = f" | score: {score:.3f}" if isinstance(score, (int, float)) else ""
                st.markdown(f"**{idx}. {ctx['cv_name']} — {ctx['section']}**{score_txt}")
                st.markdown(ctx["content"] or "_Sin contenido disponible en el índice._")
                if idx < len(st.session_state.retrieved_context):
                    st.markdown("---")
    else:
        st.info("No se recuperó contexto desde Pinecone para esta consulta.")

# ========================================
# MOSTRAR HISTORIAL DE CONVERSACIÓN
# ========================================

# Panel expandible con el historial completo
if st.session_state.conversation_history:
    with st.expander(f"📜 Ver Historial Completo ({len(st.session_state.conversation_history)} mensajes)"):
        for i, message in enumerate(st.session_state.conversation_history):
            role = "👤 Usuario" if message["role"] == "user" else "🤖 Chatbot"
            st.markdown(f"**{role}**: {message['content']}")
            if i < len(st.session_state.conversation_history) - 1:
                st.markdown("---")

# Información de debug para desarrollo
if st.sidebar.checkbox("🔧 Modo Debug (para desarrolladores)"):
    st.sidebar.markdown("### Debug Info:")
    st.sidebar.json({
        "total_mensajes": len(st.session_state.conversation_history),
        "ultimo_mensaje": st.session_state.conversation_history[-1] if st.session_state.conversation_history else "Ninguno",
        "tokens_aproximados": sum(len(msg["content"]) for msg in st.session_state.conversation_history) // 4,
        "contexto_recuperado": [
            {
                "cv": ctx.get("cv_name"),
                "seccion": ctx.get("section"),
                "score": ctx.get("score"),
            }
            for ctx in st.session_state.retrieved_context
        ],
    })
