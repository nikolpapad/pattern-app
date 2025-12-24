from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.settings import settings
from app.services.pattern_registry import resolve_pattern_path, load_patterns
from app.services.extractor_service import extract_instructions_from_image
from app.services.pdf_service import instructions_to_pdf_bytes

from pathlib import Path

router = APIRouter(prefix="/extract", tags=["extract"])

class ExtractRequest(BaseModel):
    pattern_id: str
    start_corner: str = Field(default="bottom-left", pattern="^(bottom|top)-(left|right)$")
    crochet: bool = False  # alternate direction per row


@router.post("/pdf")
def extract_pdf(req: ExtractRequest):
    # 1) Load patterns and find the one requested
    patterns = load_patterns(settings.PATTERN_DIR)
    pattern = next((p for p in patterns if p.id == req.pattern_id), None)
    if pattern is None:
        raise HTTPException(status_code=404, detail="Unknown pattern_id")

    # 2) Build path directly from the chosen pattern
    img_path = Path(settings.PATTERN_DIR) / pattern.filename
    if not img_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pattern asset missing on server: {img_path.name}"
        )

    # 3) Extract instructions
    try:
        instructions = extract_instructions_from_image(
            img_path=img_path,
            start_corner=req.start_corner,
            crochet=req.crochet,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Extraction failed: {str(e)}")

    # 4) Create PDF
    pdf_bytes = instructions_to_pdf_bytes(
        title="Pattern Instructions",
        pattern_name=pattern.name,
        start_corner=req.start_corner,
        instructions=instructions,
    )

    filename = f"{pattern.id}_{req.start_corner}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
