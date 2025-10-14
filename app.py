!pip install gradio pandas faiss-cpu llama-index langchain openai tiktoken langchain-groq
!pip install llama-index-vector-stores-faiss
!pip install sentence-transformers
!pip install llama-index-embeddings-huggingface
!pip install langchain-community
import pandas as pd
import json
from llama_index.core import Document

csv_files = ['jiopay_links_content.csv', 'jiopay_faqs.csv']
json_files = ['jiopay_help_center_faqs.json', 'jiopay_links_content.json']

docs = []

# Load CSV
for file in csv_files:
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        text = " ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(text=text, metadata={"source": file}))

# Load JSON
for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                text = " ".join([f"{k}: {v}" for k, v in item.items()])
                docs.append(Document(text=text, metadata={"source": file}))
        elif isinstance(data, dict):
            text = " ".join([f"{k}: {v}" for k, v in data.items()])
            docs.append(Document(text=text, metadata={"source": file}))

print(f"Loaded {len(docs)} documents total.")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Use HuggingFace embeddings (no API key required)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
faiss_index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM embeddings
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build the index with embeddings, explicitly setting llm to None
index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("Semantic FAISS index built successfully")

from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key="GROQ_API_KEY",  #  replace with your actual key
    model_name="llama-3.1-8b-instant"
)

from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from pydantic import Field

# Create a custom retriever that uses the modern LlamaIndex API
class ModernLlamaIndexRetriever(BaseRetriever):
    index: object = Field(...)  # Required field
    similarity_top_k: int = Field(default=3)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Use the modern as_retriever() API
        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = retriever.retrieve(query)
        
        # Convert LlamaIndex nodes to LangChain documents
        docs = [
            Document(
                page_content=node.get_content(),
                metadata=node.metadata or {}
            )
            for node in nodes
        ]
        return docs

retriever = ModernLlamaIndexRetriever(index=index, similarity_top_k=3)

# Create RAG chain
chatbot_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

print("LangChain ConversationalRetrievalChain created successfully.")

import gradio as gr

def rag_chatbot(user_input, chat_history):
    try:
        # RetrievalQA uses "query" not "question"
        result = chatbot_chain.invoke({"query": user_input})
        answer = result["result"]
        
        # Extract sources (top snippets)
        if "source_documents" in result:
            sources = result["source_documents"]
            source_texts = []
            for s in sources:
                snippet = s.page_content[:250].replace("\n", " ")  # show first 250 chars
                file_name = s.metadata.get("source", "unknown")
                source_texts.append(f"**Source:** {file_name}\n *{snippet}...*")
            sources_display = "\n\n".join(source_texts)
        else:
            sources_display = "No source found."

        full_response = f"**Answer:** {answer}\n\n---\n{sources_display}"
        return full_response
    
    except Exception as e:
        return f"Error: {str(e)}"

# Build interface
chat_ui = gr.ChatInterface(
    fn=rag_chatbot,
    title="JioPay Customer Service Chatbot",
    description="Ask anything about JioPay — this bot retrieves real info from JioPay's official site data and shows the source snippet.",
    theme="soft"
)

chat_ui.launch(server_name="0.0.0.0", server_port=7860)

