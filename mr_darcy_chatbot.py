
import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


# Page setup


st.set_page_config(page_title="Chat with Mr. Darcy 🤵", layout="centered")
st.title("Chat with Mr. Darcy 🤵")
st.markdown("Talk to Mr. Darcy from *Pride and Prejudice*. Expect formality, pride, and wit.")



# HuggingFace API Key
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.warning("Please set your Hugging Face token in .streamlit/secrets.toml")
    st.stop()

# Prompt Template
darcy_conversation_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are Mr. Fitzwilliam Darcy from Jane Austen's 'Pride and Prejudice'.
Respond as Mr. Darcy would: formal, eloquent, sometimes aloof, always intelligent.
Maintain 19th-century English tone. Do not break character.

Conversation history:
{history}

User: {input}
Mr. Darcy:"""
)

# Load Zephyr 7B from HuggingFace Hub

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 200},
    huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

qa = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=darcy_conversation_template
)

# Chat input
user_input = st.chat_input("What would you like to ask Mr. Darcy?")
if user_input:
    with st.spinner("Mr. Darcy is composing a response..."):
       
        response = qa.invoke(user_input)
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)

with st.expander("Show conversation memory"):
    st.write(memory.buffer)
