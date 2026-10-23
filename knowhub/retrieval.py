"""Multilingual dense + BM25 retrieval with reciprocal rank fusion."""

import re
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from rank_bm25 import BM25Okapi


def tokens(text):
    # Latin identifiers and Chinese unigrams/bigrams; no whitespace-only Chinese tokenizer.
    latin = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return (
        latin
        + [c for word in chinese for c in word]
        + [word[i : i + 2] for word in chinese for i in range(len(word) - 1)]
    )


class Embedder:
    def __init__(self, name="intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name, device="cpu")
        if self.model.get_embedding_dimension() != 384:
            raise ValueError("This schema requires a 384-dimensional embedding model")

    def encode(self, texts, *, query=False):
        prefix = "query: " if query else "passage: "
        return self.model.encode(
            [prefix + x for x in texts], normalize_embeddings=True, show_progress_bar=False
        )


class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def ingest(self, name, text, source, page=None, pages=None):
        if not text.strip():
            raise ValueError("Document has no extractable text")
        doc_id = str(uuid5(NAMESPACE_URL, source))
        chunks = []
        for page_number, page_text in pages or [(page, text)]:
            section = name
            # Split section-aware, then bound passages to <=700 chars (E5 max sequence length).
            for block in re.split(r"(?m)(?=^#{1,6} )", page_text):
                heading = re.match(r"^#{1,6} (.+)", block)
                if heading:
                    section = heading[1].strip()
                for start in range(0, len(block), 580):
                    passage = block[start : start + 700].strip()
                    if passage:
                        key = f"{source}:{page_number}:{len(chunks)}:{passage}"
                        chunks.append(
                            {
                                "id": str(uuid5(NAMESPACE_URL, key)),
                                "text": passage,
                                "section": section,
                                "page": page_number,
                            }
                        )
        if not chunks:
            raise ValueError("Document has no extractable text")
        vectors = self.embedder.encode([c["text"] for c in chunks])
        self.store.put_document(doc_id, name, source, self.embedder.name, chunks, vectors)
        return {"document_id": doc_id, "chunks": len(chunks)}

    def search(self, query, k=4, mode="hybrid"):
        if not query.strip():
            return []
        if mode not in {"dense", "hybrid", "lexical"}:
            raise ValueError("Unknown retrieval mode")
        k = max(1, min(k, 8))
        dense = []
        lexical = []
        if mode != "lexical":
            vector = np.asarray(self._query_vector(query), dtype=np.float32)
            dense = self.store.dense(vector, self.embedder.name, k * 3)
        if mode != "dense":
            rows = self.store.all_chunks(self.embedder.name)
            if rows:
                corpus = [tokens(r["text"]) or ["_"] for r in rows]
                scores = BM25Okapi(corpus).get_scores(tokens(query))
                lexical = [
                    dict(rows[i], score=float(scores[i])) for i in np.argsort(-scores) if scores[i] > 0
                ][: k * 3]
        if mode == "dense":
            return [dict(r, score=float(r["score"])) for r in dense[:k]]
        if mode == "lexical":
            return lexical[:k]
        fused = {}
        for candidates in [lexical, dense]:
            for rank, hit in enumerate(candidates, 1):
                item = fused.setdefault(hit["id"], dict(hit, score=0.0))
                item["score"] += 1.0 / (60 + rank)
        return sorted(fused.values(), key=lambda r: (-r["score"], r["id"]))[:k]

    def _query_vector(self, query):
        return self.embedder.encode([query], query=True)[0]
