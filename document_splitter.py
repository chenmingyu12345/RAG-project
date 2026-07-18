"""
文档分片模块 - 加载 docs/ 下的 txt/pdf 文档并切割为检索用文本片段

纯 Python 实现，零外部 DL 依赖（不依赖 torch/sentence-transformers）
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document


@dataclass
class SplitterConfig:
    """分片参数配置"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", "，"])

    @classmethod
    def from_env(cls) -> "SplitterConfig":
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
        )


class RecursiveTextSplitter:
    """递归文本切割器（纯 Python 实现，替代 langchain_text_splitters）"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，"]

    def split_text(self, text: str) -> List[str]:
        """递归按分隔符切分文本"""
        if not text.strip():
            return []
        return self._split_recursive(text, self.separators)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档列表，保留元数据"""
        chunks = []
        for doc in documents:
            texts = self.split_text(doc.page_content)
            for i, text in enumerate(texts):
                chunks.append(Document(
                    page_content=text,
                    metadata={**doc.metadata, "chunk_index": i},
                ))
        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._split_by_size(text)

        sep = separators[0]
        remaining = separators[1:]

        if sep:
            parts = text.split(sep)
        else:
            parts = list(text)

        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    result.append(part)
            else:
                result.extend(self._split_recursive(part, remaining))
        return self._merge_chunks(result)

    def _split_by_size(self, text: str) -> List[str]:
        """按固定大小切分"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """合并过短的相邻片段"""
        if not chunks:
            return []
        merged = []
        current = chunks[0]
        for chunk in chunks[1:]:
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current += " " + chunk
            else:
                if current.strip():
                    merged.append(current)
                current = chunk
        if current.strip():
            merged.append(current)
        return merged


class DocumentSplitter:
    """文档加载与分片器"""

    SUPPORTED_SUFFIXES = {".txt", ".pdf"}

    def __init__(self, docs_dir: str = "docs", config: Optional[SplitterConfig] = None):
        self.docs_dir = Path(docs_dir)
        self.config = config or SplitterConfig()
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    def _iter_files(self) -> List[Path]:
        """遍历 docs_dir 下所有支持的文件"""
        if not self.docs_dir.exists():
            print(f"[警告] 目录不存在: {self.docs_dir}")
            return []

        files = []
        for f in self.docs_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_SUFFIXES:
                files.append(f)
        return sorted(files)

    def _load_file(self, file_path: Path) -> List[Document]:
        """根据文件类型加载文档"""
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".txt":
                return self._load_txt(file_path)
            elif suffix == ".pdf":
                return self._load_pdf(file_path)
            else:
                print(f"[跳过] 不支持的文件类型: {file_path}")
                return []
        except Exception as e:
            print(f"[错误] 加载文件失败 {file_path.name}: {e}")
            return []

    def _load_txt(self, file_path: Path) -> List[Document]:
        """加载纯文本文件"""
        for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
            try:
                text = file_path.read_text(encoding=encoding)
                return [Document(
                    page_content=text,
                    metadata={"source": str(file_path), "file_type": ".txt"}
                )]
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法识别文件编码: {file_path}")

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载 PDF 文件（使用 pypdf 库）"""
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_type": ".pdf",
                        "page": i + 1,
                    }
                ))
        return docs

    def load_and_split(self) -> List[Document]:
        """加载所有文档并切分为片段"""
        files = self._iter_files()
        if not files:
            print("[信息] docs/ 目录下没有支持的文档文件（.txt / .pdf）")
            return []

        all_chunks = []
        stats: dict[str, dict] = {}

        for file_path in files:
            docs = self._load_file(file_path)
            if not docs:
                continue

            chunks = self.splitter.split_documents(docs)
            all_chunks.extend(chunks)
            stats[file_path.name] = {
                "pages": len(docs),
                "chunks": len(chunks),
                "total_chars": sum(len(doc.page_content) for doc in docs),
            }

            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["file_name"] = file_path.name

        self._print_stats(files, stats, all_chunks)
        return all_chunks

    def _print_stats(self, files: List[Path], stats: dict, all_chunks: List[Document]):
        """打印分片统计信息"""
        total_chunks = len(all_chunks)
        total_chars = sum(s["total_chars"] for s in stats.values())
        total_pages = sum(s["pages"] for s in stats.values())

        print(f"\n{'='*60}")
        print(f"  文档分片统计")
        print(f"{'='*60}")
        print(f"  文档目录    : {self.docs_dir.resolve()}")
        print(f"  分片大小    : {self.config.chunk_size} 字符")
        print(f"  重叠大小    : {self.config.chunk_overlap} 字符")
        print(f"  文件总数    : {len(files)}")
        print(f"  文档页数    : {total_pages}")
        print(f"  总字符数    : {total_chars:,}")
        print(f"  总片段数    : {total_chunks}")
        if total_chunks > 0:
            avg_size = sum(len(c.page_content) for c in all_chunks) / total_chunks
            print(f"  平均片段大小: {avg_size:.0f} 字符")
        print(f"{'='*60}")

        if stats:
            print(f"\n  各文件详情:")
            print(f"  {'文件名':<30} {'页数':>5} {'字符数':>10} {'片段数':>6}")
            print(f"  {'-'*55}")
            for name, s in stats.items():
                display_name = name if len(name) <= 28 else name[:25] + "..."
                print(f"  {display_name:<30} {s['pages']:>5} {s['total_chars']:>10,} {s['chunks']:>6}")
            print()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    splitter = DocumentSplitter(config=SplitterConfig.from_env())
    chunks = splitter.load_and_split()
