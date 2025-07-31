# Mr. Darcy GPT Chatbot using LangChain + OpenAI + Streamlit
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os
import tempfile

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Streamlit setup
st.set_page_config(page_title="Chat with Mr. Darcy", page_icon="🤵", layout="centered")
st.title("Chat with Mr. Darcy 🤵")
st.markdown("Talk to Mr. Darcy from *Pride and Prejudice*. Expect formality, pride, and wit.")

# File uploader
uploaded_file = st.file_uploader("Upload a document Mr. Darcy may reference (PDF or TXT)", type=["pdf", "txt"])
retriever = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith(".pdf") else TextLoader(tmp_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

# Prompts
darcy_retrieval_template = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are Mr. Fitzwilliam Darcy from Jane Austen's 'Pride and Prejudice'. Respond as Mr. Darcy would: formal, eloquent, sometimes aloof, always intelligent.
Maintain 19th-century English tone. Do not break character.

Use the following context if relevant:
{context}

Chat History:
{chat_history}

User: {question}
Mr. Darcy:""")

darcy_conversation_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are Mr. Fitzwilliam Darcy from Jane Austen's 'Pride and Prejudice'. Respond as Mr. Darcy would: formal, eloquent, sometimes aloof, always intelligent.
Maintain 19th-century English tone. Do not break character.

Conversation history:
{history}

User: {input}
Mr. Darcy:""")

# LLM setup
llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Chain selection
if retriever:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": darcy_retrieval_template}
    )
else:
    qa = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=darcy_conversation_template,
        verbose=False,
    )

# Chat input
user_input = st.chat_input("What would you like to ask Mr. Darcy?")
if user_input:
    with st.spinner("Mr. Darcy is composing a response..."):
        response = qa.run(user_input)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)

# Optional: show memory
with st.expander("Show memory (debug)"):
    st.write(memory.buffer)
with st.expander("Show memory (debug)"):
    st.write(memory.buffer)
