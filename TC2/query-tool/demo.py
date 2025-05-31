import streamlit as st
import pickle
import numpy as np
import faiss
from llama_cpp import Llama

@st.cache_resource
def load_models(model_path: str):
    llm_emb = Llama(
        model_path=model_path,
        gpu_layers=-1,
        n_ctx=2048,
        n_threads=8,
        use_mmap=True,
        use_mlock=True,
        embedding=True
    )
    llm_chat = Llama(
        model_path=model_path,
        gpu_layers=-1,
        n_ctx=2048,
        n_threads=8,
        use_mmap=True,
        use_mlock=True
    )
    return llm_emb, llm_chat

@st.cache_resource
def load_index(idx_path: str, chunks_path: str):
    index = faiss.read_index(idx_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


MAX_CTX = 2048
MAX_GEN = 128
RESERVED = MAX_GEN + 50
MAX_PROMPT_TOKENS = MAX_CTX - RESERVED
SYSTEM = (
    "You are a concise, professional assistant advising a finance company about Uber. "
    "Use only the provided context; justify claims with snippet citations."
)

@st.cache_data
def retrieve(query: str, k: int = 3) -> list[str]:
    resp = llm_emb.create_embedding(input=[query])
    tok_emb = np.array(resp["data"][0]["embedding"], dtype=np.float32)
    emb = tok_emb.mean(axis=0).reshape(1, -1)
    _, I = index.search(emb, k)
    return [chunks[i] for i in I[0]]

@st.cache_data
def build_prompt(query: str, docs: list[str]) -> str:
    numbered = [f"[snippet {i}]\n{txt}" for i, txt in enumerate(docs)]
    while True:
        context = "\n\n---\n\n".join(numbered)
        prompt = (
            SYSTEM + "\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
        if len(prompt.split()) <= MAX_PROMPT_TOKENS or not numbered:
            return prompt
        numbered.pop()



def main():
    st.title("Uber Finance RAG Assistant")
    st.markdown("Ask questions about Uber's financial outlook.")

    model_path = st.sidebar.text_input("Model path", "./llama-2-7b.Q4_K_S.gguf")
    idx_path = st.sidebar.text_input("FAISS index path", "rag_index.faiss")
    chunks_path = st.sidebar.text_input("Chunks pickle path", "chunks.pkl")
    k = st.sidebar.slider("Number of snippets to retrieve (k)", 1, 10, 3)

    global llm_emb, llm_chat, index, chunks
    llm_emb, llm_chat = load_models(model_path)
    index, chunks = load_index(idx_path, chunks_path)

    query = st.text_input("Enter your query:")
    if st.button("Get Answer") and query:
        with st.spinner("Retrieving relevant documents..."):
            docs = retrieve(query, k)
        st.subheader("Retrieved Snippets:")
        for i, doc in enumerate(docs):
            st.markdown(f"**Snippet {i}:** {doc}")

        prompt = build_prompt(query, docs)
        st.subheader("Answer:")
        output_box = st.empty()
        answer_text = ""
        for chunk in llm_chat(
            prompt,
            max_tokens=MAX_GEN,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.2,
            stream=True
        ):
            
            token = chunk["choices"][0]["text"]
            answer_text += token
            output_box.text(answer_text)


if __name__ == "__main__":
    main()
