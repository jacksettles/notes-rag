# rag_core.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)


class ReadingRAG:
    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name

        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_text(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def add_note(
        self,
        text: str,
        author: Optional[str] = None,
        book: Optional[str] = None,
        page: Optional[str] = None,
        topic: Optional[str] = None,
        note_type: Optional[str] = None,
    ) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Cannot add an empty note.")

        note_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        metadata = {
            "author": author or "",
            "book": book or "",
            "page": page or "",
            "topic": topic or "",
            "note_type": note_type or "",
            "created_at": created_at,
        }

        embedding = self.embed_text(text)

        self.collection.add(
            ids=[note_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        return note_id

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        query_embedding = self.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output = []
        for note_id, doc, meta, dist in zip(ids, docs, metas, distances):
            output.append(
                {
                    "id": note_id,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": dist,
                    "similarity": 1 - dist if dist is not None else None,
                }
            )

        return output

    def build_context(self, retrieved_notes: List[Dict[str, Any]]) -> str:
        if not retrieved_notes:
            return "No relevant notes were retrieved."

        context_blocks = []

        for i, item in enumerate(retrieved_notes, start=1):
            meta = item.get("metadata", {})
            text = item.get("text", "")

            author = meta.get("author", "")
            book = meta.get("book", "")
            page = meta.get("page", "")
            topic = meta.get("topic", "")
            note_type = meta.get("note_type", "")

            source_parts = []
            if author:
                source_parts.append(f"Author: {author}")
            if book:
                source_parts.append(f"Book: {book}")
            if page:
                source_parts.append(f"Page: {page}")
            if topic:
                source_parts.append(f"Topic: {topic}")
            if note_type:
                source_parts.append(f"Type: {note_type}")

            source = " | ".join(source_parts) if source_parts else "No metadata"

            block = f"""
                [Note {i}]
                Source: {source}
                Text:
                {text}
            """.strip()

            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def chat(
        self,
        user_question: str,
        top_k: int = TOP_K,
        llm_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = llm_model_name or self.llm_model_name

        retrieved_notes = self.search(user_question, top_k=top_k)
        context = self.build_context(retrieved_notes)

        user_prompt = f"""
            Use the retrieved notes below to answer the user's question.

            Retrieved notes:
            {context}

            User question:
            {user_question}

            Answer:
        """.strip()

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response["message"]["content"]

        return {
            "answer": answer,
            "retrieved_notes": retrieved_notes,
        }

    def list_notes(self, limit: int = 100) -> List[Dict[str, Any]]:
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )

        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        notes = []
        for note_id, doc, meta in zip(ids, docs, metas):
            notes.append(
                {
                    "id": note_id,
                    "text": doc,
                    "metadata": meta or {},
                }
            )

        return notes

    def count_notes(self) -> int:
        return self.collection.count()

    def delete_note(self, note_id: str) -> None:
        self.collection.delete(ids=[note_id])

    def edit_note(
        self,
        note_id: str,
        text: Optional[str] = None,
        author: Optional[str] = None,
        book: Optional[str] = None,
        page: Optional[str] = None,
        topic: Optional[str] = None,
        note_type: Optional[str] = None,
    ) -> None:
        note_id = note_id.strip()
        if not note_id:
            raise ValueError("note_id is required.")

        existing = self.collection.get(
            ids=[note_id],
            include=["documents", "metadatas"],
        )

        ids = existing.get("ids", [])
        if not ids:
            raise ValueError(f"No note found with id: {note_id}")

        current_text = existing.get("documents", [""])[0] or ""
        current_metadata = existing.get("metadatas", [{}])[0] or {}

        new_text = current_text if text is None else text.strip()
        if not new_text:
            raise ValueError("Cannot save an empty note.")

        new_metadata = {
            "author": current_metadata.get("author", ""),
            "book": current_metadata.get("book", ""),
            "page": current_metadata.get("page", ""),
            "topic": current_metadata.get("topic", ""),
            "note_type": current_metadata.get("note_type", ""),
            "created_at": current_metadata.get("created_at", ""),
        }

        if author is not None:
            new_metadata["author"] = author
        if book is not None:
            new_metadata["book"] = book
        if page is not None:
            new_metadata["page"] = page
        if topic is not None:
            new_metadata["topic"] = topic
        if note_type is not None:
            new_metadata["note_type"] = note_type

        new_embedding = self.embed_text(new_text)

        self.collection.update(
            ids=[note_id],
            documents=[new_text],
            embeddings=[new_embedding],
            metadatas=[new_metadata],
        )

    def reset_collection(self) -> None:
        ids = self.collection.get().get("ids", [])
        if ids:
            self.collection.delete(ids=ids)