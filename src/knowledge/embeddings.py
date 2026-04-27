from pathlib import Path
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    def __init__(self, persist_directory: str | Path | None = None):
        if persist_directory is None:
            persist_directory = (
                Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
            )
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory)
        )
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_or_create_collection(self, agent_id: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=agent_id,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        agent_id: str,
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        collection = self.get_or_create_collection(agent_id)
        embeddings = self._model.encode(documents).tolist()
        ids = [str(uuid4()) for _ in documents]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    def query(
        self, agent_id: str, query_text: str, n_results: int = 5
    ) -> list[dict]:
        collection = self.get_or_create_collection(agent_id)

        if collection.count() == 0:
            return []

        n_results = min(n_results, collection.count())
        query_embedding = self._model.encode([query_text]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return output

    def delete_collection(self, agent_id: str) -> None:
        self._client.delete_collection(name=agent_id)
