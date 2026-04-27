from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def load_document(file_path: str | Path) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = path.suffix.lower()
    loaders = {
        ".pdf": _load_pdf,
        ".txt": _load_txt,
        ".docx": _load_docx,
        ".md": _load_markdown,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Extensão '{ext}' não suportada. Use: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    text = loader(path)
    return split_text(text)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
