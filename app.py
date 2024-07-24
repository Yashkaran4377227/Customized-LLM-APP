import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re  
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Cyber Threat Intelligence.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text and metadata from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract metadata
            metadata = self.extract_metadata(page)
            
            self.documents.append({"page": page_num + 1, "content": text, "metadata": metadata})
        print("PDF processed successfully!")

    def extract_metadata(self, page) -> dict:
        """Extracts metadata from a PDF page."""
        metadata = {}
        # Example: Extract title and author metadata using regular expressions
        text = page.get_text()
        title = re.search(r"Title: (.+)", text)
        author = re.search(r"Author: (.+)", text)
        if title:
            metadata["title"] = title.group(1)
        if author:
            metadata["author"] = author.group(1)
        # Add more metadata fields as needed
        return metadata

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate embeddings for all document contents
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        # Add the embeddings to the index
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate an embedding for the query
        query_embedding = model.encode([query])
        # Perform a search in the FAISS index
        D, I = self.index.search(np.array(query_embedding), k)
        # Retrieve the top-k documents
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]


app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str = "You are a knowledgeable Cybersecurity Threat Intelligence Advisor. Provide accurate and up-to-date information on cyber threats, vulnerabilities, and mitigation strategies. Use the provided knowledge base to offer relevant advice on cybersecurity issues. Always prioritize current best practices and verified threat intelligence in your responses.",
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown("🛡️ **Cybersecurity Threat Intelligence Advisor**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on a cybersecurity knowledge base that is publicly available. "
        "We are not responsible for any actions taken based on the advice provided. Use this information at your own risk.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["What are the most common types of cyber attacks?"],
            ["How can I protect my organization from ransomware?"],
            ["What is phishing and how can it be prevented?"],
            ["Can you explain the concept of zero trust security?"],
            ["What are some best practices for incident response?"]
        ],
        title='Cybersecurity Threat Intelligence Advisor 🛡️'
    )

if __name__ == "__main__":
    demo.launch()
