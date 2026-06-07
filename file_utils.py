import io
import mimetypes
from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image

MAX_TEXT_CHARS = 8000


def extract_file_content(uploaded_file) -> dict:
    """
    Extract content from uploaded files.
    Returns: {"text": str, "image_bytes": bytes|None, "mime_type": str, "is_image": bool}
    """
    mime_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or ""
    file_bytes = uploaded_file.read()
    result = {"text": "", "image_bytes": None, "mime_type": mime_type, "is_image": False}

    try:
        if "pdf" in mime_type:
            reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            result["text"] = text[:MAX_TEXT_CHARS]

        elif "wordprocessingml" in mime_type or uploaded_file.name.endswith(".docx"):
            doc = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
            result["text"] = text[:MAX_TEXT_CHARS]

        elif "text/plain" in mime_type or uploaded_file.name.endswith(".txt"):
            result["text"] = file_bytes.decode("utf-8", errors="replace")[:MAX_TEXT_CHARS]

        elif any(ext in mime_type for ext in ["image/png", "image/jpeg", "image/jpg"]):
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            result["image_bytes"] = file_bytes
            result["is_image"] = True
            result["text"] = "[IMAGE ATTACHED]"

        else:
            result["text"] = f"[Unsupported file type: {mime_type}]"

    except Exception as e:
        result["text"] = f"[Error extracting file: {str(e)}]"

    return result