import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/dn_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt_template(template_str):
    return PromptTemplate(template=template_str, input_variables=["context", "question"])

def load_llm(repo_id, HF_TOKEN):
    # Specify task explicitly to avoid "Task unknown" error.
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",  # explicitly set the task
        model_kwargs={"hf_token": HF_TOKEN, "max_length": "512"},
        temperature=0.5
    )

def main():
    st.set_page_config(page_title="Medibot Chat", layout="wide")
    st.title("Ask Health Related Queries :)")

    # Sidebar with instructions and Clear Chat button.
    with st.sidebar:
        st.header("Instructions")
        st.write("Enter your query in the chat input below. Click 'Clear Chat' to reset the conversation.")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            # If experimental_rerun is available, use it; otherwise, prompt a manual refresh.
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.write("Chat cleared. Please refresh the page manually.")

    # Initialize conversation history.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history.
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Get user input.
    prompt = st.chat_input("Pass Your Prompt here")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prompt instructing the model to respond with "I am not trained for this data" if out of context.
        custom_prompt_template = """
Use the piece provided in the context to answer the user's question.
If you don't know the answer or if the question is out of context, just say "I am not trained for this data" and do not try to make up an answer.
Don't provide anything out of the given context.
context:{context}
question:{question}

Start the answer directly, no small talk please.
"""

        huggingface_repoid = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.getenv("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vectorstore not found")
                return

            # Build the retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            # First, check if we have relevant documents.
            with st.spinner("Checking relevant documents..."):
                docs = retriever.get_relevant_documents(prompt)

            # If no documents are found, respond with fallback text.
            if not docs:
                fallback_msg = "I am not trained for this data"
                st.chat_message("assistant").markdown(fallback_msg)
                st.session_state.messages.append({"role": "assistant", "content": fallback_msg})
                return

            # Otherwise, proceed with the RetrievalQA chain.
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repoid, HF_TOKEN),
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": set_custom_prompt_template(custom_prompt_template)},
                return_source_documents=True
            )

            with st.spinner("Generating response..."):
                response = qa_chain.invoke({"query": prompt})

            result = response["result"]
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
