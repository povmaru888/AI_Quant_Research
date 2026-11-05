"""Render 研究成果.md as a polished interview-ready PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "研究成果.md"
OUTPUT = ROOT / "output/pdf/研究成果.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#176B91")
TEAL = colors.HexColor("#0E7490")
ORANGE = colors.HexColor("#C66B27")
SLATE = colors.HexColor("#475569")
MUTED = colors.HexColor("#64748B")
LIGHT = colors.HexColor("#F1F5F9")
PALE_BLUE = colors.HexColor("#EAF4F8")
PALE_ORANGE = colors.HexColor("#FFF4E8")
WHITE = colors.white


def normalize(text: str) -> str:
    return (
        text.replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("~", "-")
    )


def inline(text: str) -> str:
    value = html.escape(normalize(text), quote=False)
    value = re.sub(r"`([^`]+)`", r'<font name="NotoTCMono">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r'<font color="#176B91">\1</font>', value)
    return value


def register_fonts() -> None:
    regular = "C:/Windows/Fonts/msjh.ttc"
    bold = "C:/Windows/Fonts/msjhbd.ttc"
    mono = "C:/Windows/Fonts/msjh.ttc"
    pdfmetrics.registerFont(TTFont("NotoTC", regular, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("NotoTCBold", bold, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("NotoTCMono", mono, subfontIndex=0))
    pdfmetrics.registerFontFamily("NotoTC", normal="NotoTC", bold="NotoTCBold")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleTC", parent=base["Title"], fontName="NotoTCBold", fontSize=25,
            leading=31, textColor=WHITE, alignment=TA_LEFT, spaceAfter=7 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleTC", fontName="NotoTC", fontSize=10.2, leading=16,
            textColor=colors.HexColor("#D8EAF3"), spaceAfter=4 * mm,
        ),
        "h2": ParagraphStyle(
            "H2TC", fontName="NotoTCBold", fontSize=17, leading=23,
            textColor=NAVY, spaceBefore=2 * mm, spaceAfter=4 * mm,
            borderColor=BLUE, borderWidth=0, borderPadding=0,
        ),
        "h3": ParagraphStyle(
            "H3TC", fontName="NotoTCBold", fontSize=12.2, leading=17,
            textColor=BLUE, spaceBefore=4 * mm, spaceAfter=2.3 * mm,
        ),
        "body": ParagraphStyle(
            "BodyTC", fontName="NotoTC", fontSize=8.7, leading=13.8,
            textColor=NAVY, alignment=TA_LEFT, spaceAfter=2.4 * mm,
            splitLongWords=True,
        ),
        "small": ParagraphStyle(
            "SmallTC", fontName="NotoTC", fontSize=7.4, leading=11,
            textColor=MUTED, spaceAfter=1.5 * mm,
        ),
        "quote": ParagraphStyle(
            "QuoteTC", fontName="NotoTC", fontSize=8.2, leading=13,
            textColor=SLATE, backColor=PALE_BLUE, borderColor=BLUE,
            borderWidth=0.8, borderPadding=7, leftIndent=4, rightIndent=4,
            spaceAfter=3 * mm,
        ),
        "bullet": ParagraphStyle(
            "BulletTC", fontName="NotoTC", fontSize=8.4, leading=13.2,
            textColor=NAVY, leftIndent=12, firstLineIndent=-8,
            bulletIndent=2, spaceAfter=1.2 * mm,
        ),
        "number": ParagraphStyle(
            "NumberTC", fontName="NotoTC", fontSize=8.4, leading=13.2,
            textColor=NAVY, leftIndent=15, firstLineIndent=-11,
            spaceAfter=1.4 * mm,
        ),
        "code": ParagraphStyle(
            "CodeTC", fontName="NotoTCMono", fontSize=7.5, leading=12,
            textColor=NAVY, backColor=LIGHT, borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.6, borderPadding=8, spaceAfter=3 * mm,
        ),
        "table_head": ParagraphStyle(
            "TableHeadTC", fontName="NotoTCBold", fontSize=6.7, leading=9,
            textColor=WHITE, alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCellTC", fontName="NotoTC", fontSize=6.5, leading=9.2,
            textColor=NAVY, alignment=TA_LEFT,
        ),
        "table_cell_tiny": ParagraphStyle(
            "TableCellTinyTC", fontName="NotoTC", fontSize=5.6, leading=7.5,
            textColor=NAVY, alignment=TA_LEFT,
        ),
        "caption": ParagraphStyle(
            "CaptionTC", fontName="NotoTC", fontSize=7.2, leading=10,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=1 * mm,
            spaceAfter=3 * mm,
        ),
    }


def make_table(rows: list[list[str]], sty: dict, available_width: float):
    columns = len(rows[0])
    tiny = columns >= 8
    cell_style = sty["table_cell_tiny"] if tiny else sty["table_cell"]
    parsed = []
    for row_index, row in enumerate(rows):
        row_style = sty["table_head"] if row_index == 0 else cell_style
        parsed.append([Paragraph(inline(cell), row_style) for cell in row])
    weights = []
    for col in range(columns):
        maximum = max(len(normalize(row[col])) for row in rows)
        weights.append(max(7, min(maximum, 34)))
    total = sum(weights)
    widths = [available_width * weight / total for weight in weights]
    table = LongTable(parsed, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def architecture_box(lines: list[str], sty: dict, width: float):
    cleaned = [line.strip() for line in lines if line.strip() and line.strip() not in {"|", "v"}]
    cells = []
    for index, line in enumerate(cleaned):
        cells.append(
            [
                Paragraph(
                    f'<font color="#176B91"><b>{index + 1:02d}</b></font>  {inline(line)}',
                    sty["body"],
                )
            ]
        )
    table = Table(cells, colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#B9D7E5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def parse_markdown(text: str, sty: dict, content_width: float):
    lines = normalize(text).splitlines()
    story = []
    paragraph_buffer: list[str] = []
    table_rows: list[list[str]] = []
    code_lines: list[str] = []
    in_code = False
    first_title = True
    first_h2 = True

    def flush_paragraph():
        if paragraph_buffer:
            joined = " ".join(line.strip() for line in paragraph_buffer)
            paragraph_buffer.clear()
            style = sty["small"] if joined.startswith("來源：") else sty["body"]
            story.append(Paragraph(inline(joined), style))

    def flush_table():
        if table_rows:
            rows = [row for row in table_rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in row)]
            table_rows.clear()
            if rows:
                story.extend([make_table(rows, sty, content_width), Spacer(1, 3 * mm)])

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                if any(item.strip() in {"|", "v"} for item in code_lines):
                    story.extend([architecture_box(code_lines, sty, content_width), Spacer(1, 3 * mm)])
                else:
                    story.append(Paragraph("<br/>".join(inline(item) for item in code_lines), sty["code"]))
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("<div style=") and "page-break-after" in line:
            flush_paragraph(); flush_table(); story.append(PageBreak()); continue
        if line.startswith("# "):
            flush_paragraph(); flush_table()
            if first_title:
                story.append(Spacer(1, 14 * mm))
                story.append(Paragraph(inline(line[2:]), sty["title"]))
                story.append(HRFlowable(width="36%", thickness=2.5, color=ORANGE, hAlign="LEFT"))
                story.append(Spacer(1, 5 * mm))
                first_title = False
            continue
        if line.startswith("## "):
            flush_paragraph(); flush_table()
            heading_text = line[3:].strip()
            if first_h2:
                # Keep the opening summary below the dark cover band.
                story.append(Spacer(1, 22 * mm))
                first_h2 = False
            elif heading_text.startswith("附錄｜"):
                # Keep the appendix heading with its content.
                story.append(PageBreak())
            story.append(Paragraph(inline(heading_text), sty["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#B8D5E3")))
            story.append(Spacer(1, 2 * mm)); continue
        if line.startswith("### "):
            flush_paragraph(); flush_table(); story.append(Paragraph(inline(line[4:]), sty["h3"])); continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line.strip())
        if image_match:
            flush_paragraph(); flush_table()
            image_path = ROOT / image_match.group(2)
            if image_path.is_file():
                img = Image(str(image_path))
                scale = min(content_width / img.imageWidth, 76 * mm / img.imageHeight, 1.0)
                img.drawWidth = img.imageWidth * scale
                img.drawHeight = img.imageHeight * scale
                img.hAlign = "CENTER"
                story.extend([img, Paragraph(inline(image_match.group(1)), sty["caption"])])
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            table_rows.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        flush_table()
        if line.startswith("> "):
            flush_paragraph(); story.append(Paragraph(inline(line[2:]), sty["quote"])); continue
        if re.match(r"^- ", line):
            flush_paragraph(); story.append(Paragraph(inline(line[2:]), sty["bullet"], bulletText="•")); continue
        numbered = re.match(r"^(\d+)\.\s+(.*)", line)
        if numbered:
            flush_paragraph(); story.append(Paragraph(inline(numbered.group(2)), sty["number"], bulletText=numbered.group(1) + ".")); continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph_buffer.append(line)
    flush_paragraph(); flush_table()
    return story


def page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    page = canvas.getPageNumber()
    if page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 95 * mm, width, 95 * mm, stroke=0, fill=1)
        canvas.setFillColor(BLUE)
        canvas.rect(0, height - 98 * mm, width, 3 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#EAF4F8"))
        canvas.circle(width - 24 * mm, height - 22 * mm, 28 * mm, stroke=0, fill=1)
        canvas.setFillColor(ORANGE)
        canvas.circle(width - 24 * mm, height - 22 * mm, 12 * mm, stroke=0, fill=1)
    else:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 10 * mm, width, 10 * mm, stroke=0, fill=1)
        canvas.setFont("NotoTC", 7.2)
        canvas.setFillColor(WHITE)
        canvas.drawString(17 * mm, height - 6.5 * mm, "台股多因子機器學習選股｜研究成果")
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(17 * mm, 13 * mm, width - 17 * mm, 13 * mm)
    canvas.setFont("NotoTC", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 8.5 * mm, "OOS 2020-2025｜Factor v4｜XGBoost Pseudo-Huber")
    canvas.drawRightString(width - 17 * mm, 8.5 * mm, f"{page:02d}")
    canvas.restoreState()


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page_width, _ = A4
    margin = 17 * mm
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=margin, leftMargin=margin,
        topMargin=17 * mm, bottomMargin=18 * mm,
        title="台股多因子機器學習選股｜研究成果",
        author="AI Quant Research",
        subject="量化分析師與機器學習工程師面試研究成果",
    )
    sty = styles()
    story = parse_markdown(SOURCE.read_text(encoding="utf-8-sig"), sty, page_width - 2 * margin)
    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(f"created {OUTPUT}")


if __name__ == "__main__":
    main()
