"""
向量索引模块 - 文本向量化与 ChromaDB 存储，支持创建、保存、加载和相似度检索

Embedding 方案：
  - tfidf  : sklearn TfidfVectorizer（默认，纯本地，稳定可靠，中文 char-ngram）
  - api    : HuggingFace Inference API（需网络 + HF_TOKEN）
  - local  : HuggingFaceEmbeddings（需 torch/sentence-transformers）

注意：DeepSeek 不提供 Embedding API，不可用。
可通过环境变量 EMBEDDING_PROVIDER 切换
"""
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import jieba

load_dotenv()

DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "tfidf")


# ══════════════════════════════════════════════════════════════
#  TF-IDF 向量化器（零外部依赖，兼容 ChromaDB Embeddings 接口）
# ══════════════════════════════════════════════════════════════

def _split_tokens(text: str) -> list:
    """按空格分割已分词文本"""
    return text.split()


class TfidfEmbeddings(Embeddings):
    """基于 sklearn TfidfVectorizer + jieba 中文分词的轻量 Embedding，支持持久化"""

    def __init__(self, max_features: int = 2048, cache_path: Optional[str] = None):
        self.max_features = max_features
        self.cache_path = cache_path
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> str:
        """用 jieba 分词后用空格连接，供 TfidfVectorizer word analyzer 使用"""
        return " ".join(jieba.cut(text))

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                analyzer="word",
                tokenizer=_split_tokens,
                token_pattern=None,
            )
        return self._vectorizer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        tokenized = [self._tokenize(t) for t in texts]
        if not self._fitted:
            self.vectorizer.fit(tokenized)
            self._fitted = True
            self._save_cache()
        else:
            self._load_cache()
        vectors = self.vectorizer.transform(tokenized)
        return vectors.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            self._load_cache()
        if not self._fitted:
            raise RuntimeError("TF-IDF 未拟合，请先调用 embed_documents()")
        tokenized = self._tokenize(text)
        vec = self.vectorizer.transform([tokenized])
        return vec.toarray()[0].tolist()

    def _save_cache(self):
        if self.cache_path and self._vectorizer:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "vectorizer": self._vectorizer,
                    "max_features": self.max_features,
                }, f)

    def _load_cache(self):
        if self.cache_path and Path(self.cache_path).exists():
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self._vectorizer = data["vectorizer"]
            self._fitted = True


def _get_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDING_PROVIDER", "tfidf")

    if provider == "api":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
        hf_token = os.getenv("HF_TOKEN") or None
        return HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=hf_token,
        )

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # 默认 TF-IDF（DeepSeek 不提供 Embedding API）
    return TfidfEmbeddings(
        max_features=int(os.getenv("TFIDF_MAX_FEATURES", 2048)),
        cache_path="knowledge_base/tfidf_vectorizer.pkl",
    )


# ══════════════════════════════════════════════════════════════
#  VectorIndex 向量索引
# ══════════════════════════════════════════════════════════════

class VectorIndex:
    """向量索引：将文档片段向量化后存入 ChromaDB，支持持久化和检索"""

    def __init__(
        self,
        persist_dir: str = "knowledge_base/chroma_db",
        collection_name: str = "laike_rag",
        embedding: Optional[Embeddings] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding = embedding or _get_embeddings()
        self._store: Optional[Chroma] = None

    # ─── 创建 / 重建 ──────────────────────────────────────

    def build_index(self, chunks: List[Document], clear_existing: bool = True) -> Chroma:
        """将文本片段向量化并存入向量库"""
        if not chunks:
            raise ValueError("chunks 列表为空，无法构建索引")

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if clear_existing:
            self._delete_existing()
            self.embedding = _get_embeddings()  # 重建 embedding，清除旧缓存

        self._store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
        )

        self._print_build_stats(chunks)
        return self._store

    # ─── 保存 / 加载 ──────────────────────────────────────

    def save(self):
        """ChromaDB 自动持久化到磁盘（保留接口以兼容调用方）"""
        pass

    def load(self) -> Chroma:
        """从磁盘加载已持久化的向量库"""
        if not self._exists_on_disk():
            raise FileNotFoundError(
                f"向量库不存在: {self.persist_dir}，请先调用 build_index()"
            )

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=str(self.persist_dir),
        )
        return self._store

    # ─── 检索 ─────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Document]:
        """相似度检索，返回 Top-K 文档片段"""
        store = self._ensure_loaded()
        results = store.similarity_search_with_score(query, k=k)
        return [self._enrich_result(doc, score) for doc, score in results]

    def search_with_scores(self, query: str, k: int = 5) -> List[dict]:
        """检索并返回结构化结果，含内容、元数据和分数"""
        store = self._ensure_loaded()
        results = store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(float(score), 4),
            }
            for doc, score in results
        ]

    # ─── 内部方法 ──────────────────────────────────────────

    def _ensure_loaded(self) -> Chroma:
        if self._store is None:
            return self.load()
        return self._store

    def _exists_on_disk(self) -> bool:
        return (self.persist_dir / "chroma.sqlite3").exists()

    def _delete_existing(self):
        if self._exists_on_disk():
            import chromadb
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._store = None

    @staticmethod
    def _enrich_result(doc: Document, score: float) -> Document:
        doc.metadata["similarity_score"] = round(float(score), 4)
        doc.metadata["relevance"] = f"{max(0.0, min(1.0, 1.0 - score / 4.0)):.2%}"
        return doc

    # ─── 统计 ──────────────────────────────────────────────

    def _print_build_stats(self, chunks: List[Document]):
        store = self._store
        count = store._collection.count() if store and store._collection else len(chunks)
        emb_name = type(self.embedding).__name__

        print(f"\n{'='*60}")
        print(f"  向量索引构建完成")
        print(f"{'='*60}")
        print(f"  持久化目录  : {self.persist_dir.resolve()}")
        print(f"  集合名称    : {self.collection_name}")
        print(f"  Embedding   : {emb_name}")
        print(f"  文档片段数  : {count}")
        print(f"{'='*60}\n")

    def stats(self) -> dict:
        store = self._store
        count = store._collection.count() if store and store._collection else 0
        return {
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir.resolve()),
            "document_count": count,
        }


# ─── 命令行入口 ───────────────────────────────────────────────

if __name__ == "__main__":
    from document_splitter import DocumentSplitter, SplitterConfig

    # 1. 分片
    splitter = DocumentSplitter(config=SplitterConfig.from_env())
    chunks = splitter.load_and_split()

    if not chunks:
        print("[信息] 没有文档可索引，退出。")
        exit(0)

    # 2. 构建向量索引
    index = VectorIndex()
    index.build_index(chunks)

    # 3. 测试检索
    test_query = "DeepSeek API 怎么接入"
    print(f"检索测试  Query: {test_query}\n")
    for r in index.search_with_scores(test_query, k=3):
        print(f"  [score={r['score']:.4f}] [{r['metadata'].get('file_name', '?')}]")
        print(f"  {r['content'][:120]}...\n")
