import sys
import uuid
import textwrap
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = FastAPI()

# project root (pattern-app/)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))  # so "import extractor" works reliably

# now imports are safe
from grid_extraction.api import extract_instructions

PATTERN_DIR = BASE_DIR / "patterns"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/patterns", StaticFiles(directory=str(PATTERN_DIR)), name="patterns")


class GenerateRequest(BaseModel):
    pattern_id: str


def text_to_pdf(text: str, pdf_path: Path, title: str = "Pattern Instructions") -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    _, height = A4

    x = 50
    y = height - 60
    line_height = 14

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 24

    c.setFont("Helvetica", 10)

    for line in text.splitlines():
        for wline in (textwrap.wrap(line, width=110) or [""]):
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 60
            c.drawString(x, y, wline)
            y -= line_height

    c.save()


def run_extraction_to_pdf(image_path: Path, pdf_path: Path) -> None:
    instructions_text = extract_instructions(str(image_path))
    text_to_pdf(instructions_text, pdf_path)


@app.get("/api/patterns")
def list_patterns():
    items = []
    for p in sorted(PATTERN_DIR.glob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            items.append({"id": p.name, "title": p.stem, "imageUrl": f"/patterns/{p.name}"})
    return items


@app.post("/api/generate")
def generate(req: GenerateRequest):
    image_path = PATTERN_DIR / req.pattern_id
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Pattern not found")

    out_name = f"{image_path.stem}_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = OUTPUT_DIR / out_name

    run_extraction_to_pdf(image_path, pdf_path)

    return {"downloadUrl": f"/api/download/{out_name}"}


@app.get("/api/download/{filename}")
def download(filename: str):
    pdf_path = OUTPUT_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)

# To run 
# cd backend
# uvicorn main:app --reload
# Open: http://127.0.0.1:8000/api/patterns
# (After you put some images into patterns/.)main