import hashlib
import io
import os
from dataclasses import dataclass

from pypdf import PdfReader

from app.core.config import settings
from app.core.exceptions import EmptyDocumentError, FileTooLargeError, InvalidFileTypeError

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@dataclass
class ProcessedDocument:
    filename: str
    file_type: str
    content_text: str
    file_size: int
    file_hash: str


class DocumentLoaderService:
    def process_bytes(self, filename: str, content_bytes: bytes) -> ProcessedDocument:
        file_size = len(content_bytes)
        if file_size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise FileTooLargeError(max_mb)

        # reject by extension first
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError(f"File extension '{ext}' is not allowed. Only .pdf and .txt are accepted.")

        file_hash = hashlib.sha256(content_bytes).hexdigest()

        # cross-check real content bytes against declared extension
        is_pdf_bytes = content_bytes.startswith(b"%PDF-")

        if ext == ".pdf" and not is_pdf_bytes:
            raise InvalidFileTypeError("File has .pdf extension but content is not a valid PDF.")
        if ext == ".txt" and is_pdf_bytes:
            raise InvalidFileTypeError("File has .txt extension but content appears to be a PDF.")

        if is_pdf_bytes:
            file_type = "pdf"
            content_text = self._extract_pdf_text(content_bytes)
        else:
            file_type = "txt"
            content_text = self._extract_txt_text(content_bytes)

        clean_text = content_text.strip()
        if not clean_text:
            raise EmptyDocumentError("Document contains no extractable text")

        return ProcessedDocument(
            filename=filename,
            file_type=file_type,
            content_text=clean_text,
            file_size=file_size,
            file_hash=file_hash,
        )

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text: list[str] = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            return "\n\n".join(pages_text)
        except Exception as e:
            if isinstance(e, EmptyDocumentError):
                raise
            raise InvalidFileTypeError("Failed to parse PDF document structure") from e

    def _extract_txt_text(self, txt_bytes: bytes) -> str:
        try:
            return txt_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return txt_bytes.decode("latin-1")
            except Exception as e:
                raise InvalidFileTypeError("File content is not valid plain text") from e
