from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import os
from langchain import hub
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_react_agent, tool

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/distiluse-base-multilingual-cased",
    encode_kwargs={"normalize_embeddings": True},
)
print("\033c")


class ConsultaAPI(BaseModel):
    query: str


@tool
def consultar_db_via_api(query: str):
    """
    Consulta la DB SQLite con una consulta puntual. Máximo puedes solicitar hasta 20 registros.
    NO USES COMILLAS DOBLES AL INICIO Y AL FINAL DE LA CONSULTA.


    Parámetros:
    - query (str): La consulta SQL a ejecutar en la base de datos.

    Retorna:
    - dict: Los resultados de la consulta en formato JSON.
    """
    try:
        query = query.strip('"')
        if query.endswith(";"):
            query = query[:-1]
        query = query.replace("'", "\\'")
        format_query_json = {"query": query}
        response = requests.post(
            url="https://jairodanielmt-arduino-data-post.hf.space/execute",
            json=format_query_json,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error al consultar la API: {e}")
        if e.response is not None:
            print(e.response.text)
        return None


prompt = hub.pull("hwchase17/react")
tools = [consultar_db_via_api]

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    temperature=0.3,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,
)


def ask_agent(consulta) -> str:
    d = "Eres un asistente, tienes acceso a herramientas tools y tienes permitido ejecutar sentencias SQLite, la unica tabla existente es:     la unica tabla tiene la siguiente estructura  nombre de la tabla: sensor_data columnas (id INTEGER PK AUTOINCREMENT, timestamp TEXT,humedad_suelo INTEGER, luz INTEGER, turbidez INTEGER, voltaje REAL, estado TEXT) piensa bien antes de generar la consulta SQL:"
    query = f"{d} {consulta}"
    output = agent_executor.invoke({"input": query})
    return output["output"]


import streamlit as st

# configurar la página
st.set_page_config(
    page_title="Chatbot - Arduino 🤖",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Chatbot monitoreo de sensores de Arduino 🤖")

if "history" not in st.session_state:
    st.session_state["history"] = []

pregunta = st.chat_input("Escribe tu consulta...")

if pregunta:
    st.session_state["history"].append({"role": "user", "content": pregunta})
    respuesta = ask_agent(pregunta)
    st.session_state["history"].append({"role": "ai", "content": respuesta})

for message in st.session_state["history"]:
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="👩‍💻"):
            st.write(message["content"])
    else:
        with st.chat_message(name="ai", avatar="🍦"):
            st.write(message["content"])
