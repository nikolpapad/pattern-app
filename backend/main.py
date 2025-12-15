import sys
import uuid
import textwrap
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = FastAPI()

# CORS (fine for local dev; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))  # so "import grid_extraction" works reliably

from grid_extraction.api import extract_instructions  # noqa: E402

PATTERN_DIR = BASE_DIR / "patterns"
OUTPUT_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

OUTPUT_DIR.mkdir(exist_ok=True)

# Static mounts
app.mount("/patterns", StaticFiles(directory=str(PATTERN_DIR)), name="patterns")

# IMPORTANT: mount the *frontend/static* directory at /static
FRONTEND_STATIC_DIR = FRONTEND_DIR / "static"
app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")


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


# Serve frontend at root
@app.get("/")
def read_root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing frontend file: {index_path}")
    return FileResponse(index_path)


# ✅ This is what your frontend calls
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

    try:
        run_extraction_to_pdf(image_path, pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    return {"downloadUrl": f"/api/download/{out_name}"}


@app.get("/api/download/{filename}")
def download(filename: str):
    pdf_path = OUTPUT_DIR / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)


@app.get("/__routes")
def __routes():
    out = []
    for r in app.router.routes:
        methods = sorted(getattr(r, "methods", []))
        out.append({"path": getattr(r, "path", str(r)), "methods": methods})
    return out