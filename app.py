"""Corrective RAG System - standalone Gradio app.

Run with:
    python app.py

Then open the printed local URL (usually http://127.0.0.1:7860) in your browser.

Requires:
  - Ollama running locally (https://ollama.com) with the models pulled:
      ollama pull nomic-embed-text
      ollama pull llama3.2:3b
  - A populated vector store: run 02_Embeddings_VectorStore.ipynb at least once first
    (it builds data/processed/chunks.json and vectorstore/chroma_db).
"""

import os
import urllib.request
from pathlib import Path
from typing import Literal

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"
COLLECTION_NAME = "crag_course_docs"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def check_ollama() -> None:
    try:
        urllib.request.urlopen(OLLAMA_BASE_URL, timeout=3)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Make sure the Ollama app is running "
            "(it starts automatically after installation, or run 'ollama serve' manually)."
        ) from exc


class GradeDocument(BaseModel):
    """Binary relevance grade for a retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, otherwise 'no'"
    )


class CorrectiveRAGPipeline:
    """End-to-end Corrective RAG: retrieve -> grade -> rewrite/re-retrieve -> generate."""

    GRADER_SYSTEM_PROMPT = (
        "You are a grader assessing the relevance of a retrieved document to a user question. "
        "If the document contains information that helps answer the question, grade it as relevant."
    )
    REWRITER_SYSTEM_PROMPT = (
        "You are a query re-writer that converts an input question into a better version optimized "
        "for vector store retrieval, focused on the underlying semantic intent. Return only the "
        "rewritten question."
    )
    ANSWER_SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions using ONLY the provided context. "
        "Never rely on outside knowledge. If the context is insufficient, say so clearly instead of "
        "guessing. Cite the source file(s) you used in square brackets, using the exact file name shown "
        "after 'Source:' above (for example, if a chunk's source line says 'Source: 01_rag_basics.txt', "
        "cite it as [01_rag_basics.txt]). Never write the literal placeholder text 'source.txt'."
    )
    NO_ANSWER_MESSAGE = (
        "I don't have enough verified information in the available documents to confidently answer "
        "this question. Try rephrasing it, or add more documents to data/raw."
    )

    def __init__(self, vectorstore, llm_model: str = LLM_MODEL, base_url: str = OLLAMA_BASE_URL,
                 k: int = 4, relevance_threshold: float = 0.5, max_rewrites: int = 2):
        self.vectorstore = vectorstore
        self.llm = ChatOllama(model=llm_model, temperature=0, base_url=base_url)
        self.grader_llm = self.llm.with_structured_output(GradeDocument)
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.max_rewrites = max_rewrites

    def retrieve(self, query: str):
        return self.vectorstore.similarity_search(query, k=self.k)

    def grade_document(self, question: str, document_text: str) -> str:
        prompt = f"Retrieved document:\n\n{document_text}\n\nUser question: {question}"
        result = self.grader_llm.invoke(
            [
                {"role": "system", "content": self.GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return result.binary_score

    def rewrite_query(self, question: str) -> str:
        response = self.llm.invoke(
            [
                {"role": "system", "content": self.REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original question: {question}\n\nRewritten question:"},
            ]
        )
        return response.content.strip()

    def corrective_retrieve(self, question: str):
        trace = []
        current_query = question
        best_relevant_docs = []

        for attempt in range(self.max_rewrites + 1):
            docs = self.retrieve(current_query)
            grades = [self.grade_document(question, doc.page_content) for doc in docs]
            relevant_docs = [doc for doc, grade in zip(docs, grades) if grade == "yes"]
            ratio = len(relevant_docs) / len(docs) if docs else 0.0

            trace.append(
                {
                    "attempt": attempt,
                    "query_used": current_query,
                    "retrieved": len(docs),
                    "relevant": len(relevant_docs),
                    "relevance_ratio": round(ratio, 2),
                }
            )

            if len(relevant_docs) > len(best_relevant_docs):
                best_relevant_docs = relevant_docs

            if ratio >= self.relevance_threshold:
                return relevant_docs, trace

            if attempt < self.max_rewrites:
                current_query = self.rewrite_query(current_query)

        return best_relevant_docs, trace

    @staticmethod
    def build_context(docs) -> str:
        blocks = [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
        return "\n\n---\n\n".join(blocks)

    def generate_answer(self, question: str, verified_docs) -> str:
        context = self.build_context(verified_docs)
        user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
        response = self.llm.invoke(
            [
                {"role": "system", "content": self.ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.content.strip()

    def answer(self, question: str) -> dict:
        verified_docs, trace = self.corrective_retrieve(question)

        if not verified_docs:
            return {"answer": self.NO_ANSWER_MESSAGE, "sources": [], "trace": trace}

        answer_text = self.generate_answer(question, verified_docs)
        sources = sorted({doc.metadata["source"] for doc in verified_docs})
        return {"answer": answer_text, "sources": sources, "trace": trace}


def build_pipeline() -> CorrectiveRAGPipeline:
    check_ollama()

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "No vector store found. Run 02_Embeddings_VectorStore.ipynb at least once first."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    return CorrectiveRAGPipeline(vectorstore=vectorstore, llm_model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


pipeline = build_pipeline()


def chat_fn(message, history):
    result = pipeline.answer(message)

    reply = result["answer"]
    if result["sources"]:
        reply += "\n\n**Sources:** " + ", ".join(result["sources"])

    trace_lines = [
        f"- attempt {t['attempt']}: query=\"{t['query_used']}\" -> "
        f"{t['relevant']}/{t['retrieved']} relevant"
        for t in result["trace"]
    ]
    reply += (
        "\n\n<details><summary>Correction trace (retrieval steps)</summary>\n\n"
        + "\n".join(trace_lines)
        + "\n\n</details>"
    )

    return reply


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Corrective RAG System",
    description=(
        "Ask any question about the documents in data/raw. The system grades retrieval quality, "
        "corrects the query if needed, and answers only from verified sources."
    ),
    examples=[
        "How does Corrective RAG reduce hallucination compared to standard RAG?",
        "What is the difference between Chroma and FAISS?",
        "What is the capital of France?",
    ],
)


if __name__ == "__main__":
    demo.launch()
