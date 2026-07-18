"""
RAG 问答引擎 - 串联「召回 → 重排 → 生成」三个环节

- 召回：从向量库检索 Top-N 相关片段
- 重排：基于关键词密度对召回结果二次排序，精选 Top-K
- 生成：拼接上下文，调用 DeepSeek 大模型生成答案
"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

from vector_index import VectorIndex

load_dotenv()


# ══════════════════════════════════════════════════════════════
#  重排器
# ══════════════════════════════════════════════════════════════

class KeywordReranker:
    """基于关键词密度的轻量重排器"""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        对召回片段按关键词命中密度重新打分排序，返回 Top-K
        """
        if not documents:
            return []

        keywords = self._extract_keywords(query)

        scored = []
        for doc in documents:
            score = self._keyword_density_score(doc.page_content, keywords)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:self.top_k]

        for rank, (score, doc) in enumerate(selected):
            doc.metadata["rerank_score"] = round(score, 4)
            doc.metadata["rerank_rank"] = rank + 1

        return [doc for _, doc in selected]

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """从查询中提取中文关键词（2-4 字词组）"""
        cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        tokens = cleaned.split()
        keywords = []
        for t in tokens:
            if len(t) >= 2:
                keywords.append(t.lower())
            for i in range(len(t) - 1):
                bigram = t[i:i+2]
                if re.search(r"[\u4e00-\u9fa5]", bigram):
                    keywords.append(bigram)
        return list(dict.fromkeys(keywords))

    @staticmethod
    def _keyword_density_score(content: str, keywords: List[str]) -> float:
        """计算关键词在内容中的命中密度"""
        content_lower = content.lower()
        hits = sum(1 for kw in keywords if kw in content_lower)
        if not hits:
            return 0.0
        density = hits / max(len(content), 1) * 1000
        return round(density, 4)


# ══════════════════════════════════════════════════════════════
#  RAG 引擎
# ══════════════════════════════════════════════════════════════

@dataclass
class RAGConfig:
    """RAG 流程参数"""
    recall_top_n: int = 10      # 召回阶段取的片段数
    rerank_top_k: int = 3       # 重排后保留的片段数
    max_context_chars: int = 3000  # 拼接上下文最大字符数
    temperature: float = 0.3

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls(
            recall_top_n=int(os.getenv("RECALL_TOP_N", 10)),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", 3)),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", 3000)),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.3)),
        )


SYSTEM_PROMPT = """你是一个知识问答助手。

回答规则：
1. 如果提供了"参考文档"，优先基于文档内容回答，回答简洁准确
2. 如果没有提供参考文档，或文档不包含问题答案，直接用你的知识回答
3. 不要添加"根据文档"等前缀，直接给出答案"""


class RAGEngine:
    """RAG 问答引擎"""

    def __init__(self, index: Optional[VectorIndex] = None, config: Optional[RAGConfig] = None):
        self.index = index or VectorIndex()
        self.config = config or RAGConfig.from_env()
        self.reranker = KeywordReranker(top_k=self.config.rerank_top_k)
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        return self._client

    def ask(self, question: str, history: Optional[List[dict]] = None, verbose: bool = True) -> dict:
        """
        执行 RAG 问答全流程，返回结构化结果

        history: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
                  最多保留最近 10 条消息（5 轮对话）

        返回: {
            "question": str,
            "answer": str,
            "stages": { "recall": {...}, "rerank": {...}, "generate": {...} },
            "sources": [...]
        }
        """
        # 只保留最近 10 条（5 轮对话）
        if history:
            history = history[-10:]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  RAG 问答流程")
            print(f"{'='*60}")
            print(f"  问题: {question}")
            if history:
                print(f"  历史: {len(history)} 条消息")

        # ── 阶段 1: 召回 ──
        recall_result = self._stage_recall(question, verbose)

        # ── 阶段 2: 重排 ──
        rerank_result = self._stage_rerank(question, recall_result["documents"], verbose)

        # ── 阶段 3: 生成 ──
        generate_result = self._stage_generate(question, rerank_result["documents"], history, verbose)

        result = {
            "question": question,
            "answer": generate_result["answer"],
            "stages": {
                "recall": {k: v for k, v in recall_result.items() if k != "documents"},
                "rerank": {k: v for k, v in rerank_result.items() if k != "documents"},
                "generate": generate_result,
            },
            "sources": [
                {
                    "file": doc.metadata.get("file_name", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "rerank_score": doc.metadata.get("rerank_score", 0),
                    "preview": doc.page_content[:200],
                }
                for doc in rerank_result["documents"]
            ],
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"  最终答案")
            print(f"{'='*60}")
            print(f"  {generate_result['answer']}")
            print(f"{'='*60}\n")

        return result

    # ─── 阶段 1: 召回 ────────────────────────────────────────

    def _stage_recall(self, question: str, verbose: bool) -> dict:
        """从向量库召回 Top-N 相关片段"""
        t0 = time.time()

        try:
            self.index.load()
        except FileNotFoundError:
            return {"documents": [], "count": 0, "elapsed": 0, "error": "向量库不存在，请先构建索引"}

        docs = self.index.search(question, k=self.config.recall_top_n)
        elapsed = round((time.time() - t0) * 1000, 1)

        if verbose:
            print(f"\n  ┌─ [阶段 1] 向量召回 ─────────────────────────────")
            print(f"  │  检索数量: {self.config.recall_top_n}")
            print(f"  │  命中数量: {len(docs)}")
            print(f"  │  耗时    : {elapsed} ms")
            for i, doc in enumerate(docs):
                score = doc.metadata.get("similarity_score", 0)
                source = doc.metadata.get("file_name", "?")
                print(f"  │  [{i+1}] score={score:.4f}  {source}")
                print(f"  │      {doc.page_content[:80]}...")
            print(f"  └──────────────────────────────────────────────────")

        return {"documents": docs, "count": len(docs), "elapsed_ms": elapsed}

    # ─── 阶段 2: 重排 ────────────────────────────────────────

    def _stage_rerank(self, question: str, documents: List[Document], verbose: bool) -> dict:
        """关键词密度重排，精选最优片段"""
        t0 = time.time()

        reranked = self.reranker.rerank(question, documents)
        # 过滤掉 rerank 得分为 0 的不相关片段
        reranked = [doc for doc in reranked if doc.metadata.get("rerank_score", 0) > 0]
        elapsed = round((time.time() - t0) * 1000, 1)

        if verbose:
            print(f"\n  ┌─ [阶段 2] 关键词重排 ───────────────────────────")
            print(f"  │  输入片段: {len(documents)}")
            print(f"  │  有效片段: {len(reranked)} (已过滤 rerank=0)")
            print(f"  │  耗时    : {elapsed} ms")
            for i, doc in enumerate(reranked):
                score = doc.metadata.get("rerank_score", 0)
                source = doc.metadata.get("file_name", "?")
                print(f"  │  [{i+1}] rerank={score:.4f}  {source}")
            print(f"  └──────────────────────────────────────────────────")

        return {"documents": reranked, "count": len(reranked), "elapsed_ms": elapsed}

    # ─── 阶段 3: 生成 ────────────────────────────────────────

    def _stage_generate(self, question: str, documents: List[Document],
                        history: Optional[List[dict]] = None, verbose: bool = True) -> dict:
        """拼接上下文与对话历史，调用 DeepSeek 生成答案。无文档时直接联网回答"""
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # 构建 messages：system + 对话历史 + 当前问题
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)

        if documents:
            # 有文档：拼接文档上下文
            context_parts = []
            total_chars = 0
            for doc in documents:
                content = doc.page_content
                if total_chars + len(content) > self.config.max_context_chars:
                    remaining = self.config.max_context_chars - total_chars
                    if remaining > 50:
                        content = content[:remaining]
                    else:
                        break
                source = doc.metadata.get("file_name", "未知")
                context_parts.append(f"[来源: {source}]\n{content}")
                total_chars += len(content)
            context_text = "\n\n---\n\n".join(context_parts)
            messages.append({"role": "user",
                             "content": f"参考文档：\n\n{context_text}\n\n问题：{question}\n\n请回答："})
            context_chunks = len(context_parts)
            context_chars = total_chars
        else:
            # 无文档：直接提问，LLM 用自身知识回答
            messages.append({"role": "user", "content": question})
            context_chunks = 0
            context_chars = 0

        t0 = time.time()
        response = self.client.chat.completions.create(
            model=model,
            temperature=self.config.temperature,
            messages=messages,
        )
        elapsed = round((time.time() - t0) * 1000, 1)

        if verbose:
            print(f"\n  ┌─ [阶段 3] LLM 生成 ─────────────────────────────")
            print(f"  │  模型    : {model}")
            print(f"  │  上下文  : {context_chunks} 个片段, {context_chars:,} 字符")
            print(f"  │  耗时    : {elapsed} ms")
            print(f"  └──────────────────────────────────────────────────")

        return {
            "answer": response.choices[0].message.content,
            "context_chars": context_chars,
            "context_chunks": context_chunks,
            "model": model,
            "elapsed_ms": elapsed,
        }

    # ─── 批量 ──────────────────────────────────────────────────

    def ask_batch(self, questions: List[str], verbose: bool = False) -> List[dict]:
        """批量问答"""
        return [self.ask(q, verbose=verbose) for q in questions]


# ─── 命令行入口 ───────────────────────────────────────────────

if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.ask("DeepSeek API 怎么接入")
