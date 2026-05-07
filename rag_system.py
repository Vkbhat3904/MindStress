import os
import json
import re
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

LIST_METADATA_KEYS = ("applicable_conditions", "stress_level_mapping", "steps_to_perform")


def _chroma_safe_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be str, int, float, or bool."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if k in LIST_METADATA_KEYS and isinstance(v, (list, tuple)):
            out[k] = json.dumps(list(v))
        elif isinstance(v, (list, tuple, dict)):
            out[k] = json.dumps(v)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _denormalize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {}
    m = dict(meta)
    for key in LIST_METADATA_KEYS:
        if key not in m:
            continue
        val = m[key]
        if isinstance(val, str):
            try:
                m[key] = json.loads(val)
            except json.JSONDecodeError:
                m[key] = [val] if val else []
        elif val is None:
            m[key] = []
    return m


def _denormalize_query_results(results: Dict[str, Any]) -> Dict[str, Any]:
    if not results.get("metadatas") or not results["metadatas"][0]:
        return results
    row = [_denormalize_metadata(m) for m in results["metadatas"][0]]
    results = dict(results)
    results["metadatas"] = [row]
    return results


def _technique_title(slug: str) -> str:
    return slug.replace("_", " ").strip().title()


class StressRAGSystem:
    def __init__(self, json_file="stress_knowledge_base.json", collection_name="stress_kb"):
        self.json_file = json_file
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self._ensure_collection_populated()

    def _ensure_collection_populated(self):
        """
        Ensure Chroma collection has retrievable knowledge.
        Priority:
        1) Existing collection data
        2) Structured WHO chunks JSON
        3) Curated stress_knowledge_base.json
        """
        existing = self.collection.count()
        if existing > 0:
            return

        chunk_paths = [
            "knowledge/who_mhgap_stress_chunks.json",
            "knowledge/who_stress_chunks.json",
        ]
        for chunk_path in chunk_paths:
            if not os.path.exists(chunk_path):
                continue
            try:
                with open(chunk_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if isinstance(records, list) and records:
                    prefix = "who_mhgap" if "mhgap" in chunk_path else "who"
                    self.ingest_structured_chunks(records, id_prefix=prefix, replace=False)
                    return
            except Exception:
                continue

        if os.path.exists(self.json_file):
            self.load_and_embed_knowledge_base()

    def load_and_embed_knowledge_base(self):
        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for i, entry in enumerate(data):
            text = (
                f"Technique: {entry['technique_name']}. Description: {entry['description']}. "
                f"Applicable Conditions: {', '.join(entry['applicable_conditions'])}. "
                f"Stress Levels: {', '.join(entry['stress_level_mapping'])}. "
                f"Duration: {entry['duration']}. "
                f"Steps: {'; '.join(entry['steps_to_perform'])}."
            )
            documents.append(text)
            embeddings.append(self.model.encode(text).tolist())
            meta = _chroma_safe_metadata(
                {
                    "technique_name": entry["technique_name"],
                    "description": entry["description"],
                    "applicable_conditions": entry["applicable_conditions"],
                    "stress_level_mapping": entry["stress_level_mapping"],
                    "duration": entry["duration"],
                    "steps_to_perform": entry["steps_to_perform"],
                    "source": entry.get("source", "curated_kb"),
                }
            )
            metadatas.append(meta)
            ids.append(f"technique_{i}")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Loaded {len(data)} techniques into ChromaDB.")

    def ingest_structured_chunks(
        self,
        records: Sequence[Dict[str, Any]],
        id_prefix: str = "who",
        replace: bool = True,
    ):
        """
        Ingest PDF-derived chunks. Each record should have at least:
        text, source, technique, conditions (list[str]), optional stress_level_mapping.
        Stored with planner-compatible fields: technique_name, description, steps_to_perform.
        """
        if replace:
            existing = self.collection.get(include=[])
            eids = existing.get("ids") or []
            drop = [i for i in eids if i.startswith(f"{id_prefix}_")]
            if drop:
                self.collection.delete(ids=drop)

        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for i, rec in enumerate(records):
            text = rec["text"]
            source = rec.get("source", "WHO")
            technique = rec.get("technique") or "stress management"
            conditions: List[str] = list(rec.get("conditions") or [])
            stress_map: List[str] = list(
                rec.get("stress_level_mapping") or ["low", "moderate", "high", "very_high"]
            )

            embed_text = (
                f"{text}\nTechnique: {technique}. Source: {source}. "
                f"Conditions: {', '.join(conditions)}."
            )

            technique_name = f"{source}: {_technique_title(technique)}"
            steps = rec.get("steps_to_perform")
            if not steps:
                parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
                steps = parts[:12] if parts else [text[:500]]

            meta = _chroma_safe_metadata(
                {
                    "technique_name": technique_name,
                    "description": text,
                    "applicable_conditions": conditions,
                    "stress_level_mapping": stress_map,
                    "duration": rec.get("duration", "as needed"),
                    "steps_to_perform": steps if isinstance(steps, list) else [str(steps)],
                    "source": source,
                    "technique_slug": technique,
                    "doc_slug": rec.get("doc_slug", ""),
                    "chunk_type": rec.get("type", "technique"),
                }
            )

            documents.append(embed_text)
            embeddings.append(self.model.encode(embed_text).tolist())
            metadatas.append(meta)
            ids.append(f"{id_prefix}_{i}")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Ingested {len(records)} chunks as {id_prefix}_*.")

    def search_similar(self, query, n_results=3):
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return _denormalize_query_results(results)


if __name__ == "__main__":
    rag = StressRAGSystem()
    rag.load_and_embed_knowledge_base()

    query = "techniques for high stress and anxiety"
    results = rag.search_similar(query)
    print("Search Results:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i + 1}. {doc[:200]}...")
        print(f"Metadata: {results['metadatas'][0][i]}")
        print("---")
