from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

def instructions_to_pdf_bytes(
    title: str,
    pattern_name: str,
    start_corner: str,
    instructions: list[str],
) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    # Basic typography
    margin_x = 18 * mm
    margin_top = 18 * mm
    line_h = 6.0 * mm

    y = height - margin_top
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_x, y, title)
    y -= 10 * mm

    c.setFont("Helvetica", 10)
    c.drawString(margin_x, y, f"Pattern: {pattern_name}")
    y -= 6 * mm
    c.drawString(margin_x, y, f"Start corner: {start_corner}")
    y -= 10 * mm

    c.setFont("Helvetica", 10)

    for i, line in enumerate(instructions, start=1):
        # page break
        if y < 18 * mm:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin_top

        # Keep it simple: one line per instruction
        c.drawString(margin_x, y, line)
        y -= line_h

    c.showPage()
    c.save()
    return buf.getvalue()
