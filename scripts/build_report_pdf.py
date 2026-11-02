"""Render a report markdown file to PDF, with no external toolchain.

Why this exists
---------------
A submission portal wants a document, not a repository of markdown. Pandoc and LaTeX are
not always available (and never on a marker's laptop), so this script renders the report
with reportlab, which is a single pip install.

Supported markdown subset -- deliberately small, and everything ``report/*.md`` uses:

  # h1   ## h2   ### h3   #### h4
  paragraphs with **bold**, *italic*, `code`
  - bullets and 1. numbered lists (one level of nesting)
  | pipe | tables | with an optional |---| separator row |
  ``` fenced code blocks ```
  > blockquote / callout paragraphs
  --- horizontal rules

Usage::

    python scripts/build_report_pdf.py                              # report/REPORT.md -> report/REPORT.pdf
    python scripts/build_report_pdf.py --input report/REPORT.md --output report/REPORT.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Sequence

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageTemplate,
        Paragraph,
        Preformatted,
        Spacer,
        Table,
        TableStyle,
    )
except ModuleNotFoundError:  # pragma: no cover
    print("ERROR: reportlab is required.  pip install reportlab", file=sys.stderr)
    raise SystemExit(1)

NAVY = colors.HexColor("#1F3864")
ACCENT = colors.HexColor("#2E5C8A")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#F2F4F8")
RULE = colors.HexColor("#C9D2E0")

INLINE = re.compile(r"(\*\*.+?\*\*|__.+?__|\*[^*]+?\*|`[^`]+?`|\[[^\]]+?\]\([^)]+?\))")
CODE_MARK = "\x00"

#: reportlab's built-in fonts use WinAnsi, which has no glyph for the math arrows and
#: comparison signs we use freely in markdown. Map them to readable ASCII so the PDF
#: never shows a black box where a symbol should be.
GLYPHS = {
    "\u2192": "->", "\u2190": "<-", "\u2194": "<->",
    "\u2299": "x", "\u2265": ">=", "\u2264": "<=", "\u2248": "~",
    "\u0394": "Delta", "\u2212": "-", "\u00d7": "x", "\u00b7": "-",
    "\u2260": "!=", "\u2211": "sum", "\u2208": "in", "\u211d": "R",
    "\u2705": "[OK]", "\u26a0\ufe0f": "[!]", "\u26a0": "[!]",
    "\u2713": "yes", "\u2022": "-", "\u00b1": "+/-", "\u2013": "-",
    "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    # box drawing / arrows used by the pipeline diagrams in the report
    "\u2500": "-", "\u2502": "|", "\u250c": "+", "\u2510": "+", "\u2514": "+",
    "\u2518": "+", "\u251c": "+", "\u2524": "+", "\u252c": "+", "\u2534": "+",
    "\u253c": "+", "\u25b6": ">", "\u25bc": "v", "\u25ba": ">", "\u27a4": ">",
    "\u2933": ">", "\u21b3": "->", "\u2026": "...",
}


def ascii_safe(text: str) -> str:
    for source, target in GLYPHS.items():
        text = text.replace(source, target)
    return text


# --------------------------------------------------------------------------- #
# Inline formatting
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    text = ascii_safe(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    """Convert the inline markdown subset to reportlab's mini-HTML."""
    out: List[str] = []
    # protect inline code first so its contents are never parsed as markdown
    codes: List[str] = []

    def stash(match: re.Match) -> str:
        codes.append(match.group(0)[1:-1])
        return f"{CODE_MARK}{len(codes) - 1}{CODE_MARK}"

    text = re.sub(r"`[^`]+?`", stash, text)
    for token in INLINE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            out.append(f"<b>{esc(token[2:-2])}</b>")
        elif token.startswith("__") and token.endswith("__"):
            out.append(f"<b>{esc(token[2:-2])}</b>")
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            out.append(f"<i>{esc(token[1:-1])}</i>")
        elif token.startswith("[") and "]" in token:
            label = token[1 : token.index("]")]
            out.append(f"<i>{esc(label)}</i>")
        else:
            out.append(esc(token))
    rendered = "".join(out)
    for i, code in enumerate(codes):
        rendered = rendered.replace(f"{CODE_MARK}{i}{CODE_MARK}", f'<font face="Courier">{esc(code)}</font>')
    return rendered


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
            leading=25, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11.5,
            leading=16, textColor=GREY, alignment=TA_CENTER, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15,
            leading=19, textColor=NAVY, spaceBefore=16, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
            leading=16, textColor=ACCENT, spaceBefore=12, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="Helvetica-BoldOblique", fontSize=11,
            leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=13, leftIndent=12, bulletIndent=3, spaceAfter=2.5,
        ),
        "bullet2": ParagraphStyle(
            "Bullet2", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
            leading=13, leftIndent=24, bulletIndent=14, spaceAfter=2.5,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=8,
            leading=10.5, backColor=LIGHT, borderPadding=5, leftIndent=6,
            textColor=colors.HexColor("#1A1A1A"),
        ),
        "quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=9.5,
            leading=13.5, leftIndent=10, textColor=colors.HexColor("#333333"),
            backColor=colors.HexColor("#FBF3D5"), borderPadding=6, spaceBefore=4, spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "Cell", fontName="Helvetica", fontSize=7.6, leading=9.6,
        ),
        "cell_head": ParagraphStyle(
            "CellHead", fontName="Helvetica-Bold", fontSize=7.8, leading=9.8,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8,
            leading=11, textColor=GREY, spaceAfter=6,
        ),
    }


# --------------------------------------------------------------------------- #
# Block parsing
# --------------------------------------------------------------------------- #
def split_row(line: str) -> List[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def is_separator(line: str) -> bool:
    stripped = line.strip().strip("|")
    if not stripped:
        return False
    return all(re.fullmatch(r":?-{2,}:?", part.strip()) for part in stripped.split("|"))


def make_table(rows: Sequence[Sequence[str]], styles: dict, available: float):
    headers = rows[0]
    body = [r for r in rows[1:] if not is_separator("|".join(r))]
    ncols = len(headers)
    body = [(r + [""] * ncols)[:ncols] for r in body]

    data = [[Paragraph(inline(cell), styles["cell_head"]) for cell in headers]]
    for row in body:
        data.append([Paragraph(inline(cell), styles["cell"]) for cell in row])

    # weight columns by their longest visible cell, so wide text columns get the room
    widths = []
    for ci in range(ncols):
        longest = max([len(headers[ci])] + [len(r[ci]) for r in body] or [4])
        widths.append(min(max(longest, 4), 60))
    total = sum(widths) or ncols
    col_widths = [available * w / total for w in widths]
    # reportlab dislikes very narrow columns next to very wide ones
    minimum = available / (ncols * 3.2)
    col_widths = [max(w, minimum) for w in col_widths]
    scale = available / sum(col_widths)
    col_widths = [w * scale for w in col_widths]

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for ri in range(1, len(data)):
        if ri % 2 == 0:
            style.append(("BACKGROUND", (0, ri), (-1, ri), LIGHT))
    table.setStyle(TableStyle(style))
    return table


def render(markdown: str, styles: dict, available: float) -> List:
    story: List = []
    lines = markdown.splitlines()
    i = 0
    first_heading = True

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- fenced code -------------------------------------------------
        if stripped.startswith("```"):
            i += 1
            block: List[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            story.append(Spacer(1, 3))
            story.append(Preformatted(ascii_safe("\n".join(block)), styles["code"]))
            story.append(Spacer(1, 6))
            continue

        # ---- tables ------------------------------------------------------
        if stripped.startswith("|") and "|" in stripped[1:]:
            rows: List[List[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = split_row(lines[i])
                if not is_separator(lines[i]):
                    rows.append(cells)
                i += 1
            if rows:
                story.append(Spacer(1, 3))
                story.append(make_table(rows, styles, available))
                story.append(Spacer(1, 7))
            continue

        # ---- headings ----------------------------------------------------
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = inline(heading.group(2))
            if first_heading and level == 1:
                story.append(Paragraph(text, styles["title"]))
                first_heading = False
            elif level == 2 and len(story) <= 2:
                story.append(Paragraph(text, styles["subtitle"]))
            elif level == 1:
                story.append(Spacer(1, 6))
                story.append(Paragraph(text, styles["h1"]))
            elif level == 2:
                story.append(Paragraph(text, styles["h2"]))
            else:
                story.append(Paragraph(text, styles["h3"]))
            i += 1
            continue

        # ---- horizontal rule ---------------------------------------------
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            story.append(Spacer(1, 4))
            story.append(Table([[""]], colWidths=[available], rowHeights=[0.6],
                               style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.8, RULE)])))
            story.append(Spacer(1, 6))
            i += 1
            continue

        # ---- blockquote --------------------------------------------------
        if stripped.startswith(">"):
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            text = " ".join(part for part in block if part)
            if text:
                story.append(Paragraph(inline(text), styles["quote"]))
            continue

        # ---- lists -------------------------------------------------------
        bullet_match = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        number_match = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if bullet_match or number_match:
            indent = len((bullet_match or number_match).group(1))
            style = styles["bullet2"] if indent >= 2 else styles["bullet"]
            while i < len(lines):
                current = lines[i]
                b = re.match(r"^(\s*)[-*+]\s+(.*)$", current)
                n = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", current)
                if not (b or n):
                    break
                marker = "\u2022" if b else f"{(n).group(2)}."
                text = (b.group(2) if b else n.group(3))
                story.append(
                    Paragraph(inline(text), style, bulletText=marker)
                )
                i += 1
            story.append(Spacer(1, 3))
            continue

        # ---- paragraph ---------------------------------------------------
        if stripped:
            paragraph: List[str] = []
            while i < len(lines):
                current = lines[i]
                if not current.strip():
                    break
                if (current.strip().startswith(("```", "|", ">", "#"))
                        or re.match(r"^\s*[-*+]\s+", current)
                        or re.match(r"^\s*\d+[.)]\s+", current)
                        or re.fullmatch(r"-{3,}", current.strip())):
                    break
                paragraph.append(current.strip())
                i += 1
            text = " ".join(paragraph)
            if text:
                story.append(Paragraph(inline(text), styles["body"]))
            continue

        i += 1

    return story


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 10 * mm, "From Patches to Prognosis — weakly supervised ISUP grading on PANDA")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a markdown report to PDF.")
    parser.add_argument("--input", default=str(root / "report" / "REPORT.md"))
    parser.add_argument("--output", default=str(root / "report" / "REPORT.pdf"))
    args = parser.parse_args(argv)

    source = Path(args.input)
    if not source.is_file():
        print(f"ERROR: input not found: {source}", file=sys.stderr)
        return 1

    markdown = source.read_text(encoding="utf-8")
    styles = build_styles()
    margin = 18 * mm
    available = A4[0] - 2 * margin

    doc = BaseDocTemplate(
        args.output, pagesize=A4,
        leftMargin=margin, rightMargin=margin, topMargin=16 * mm, bottomMargin=16 * mm,
        title="From Patches to Prognosis", author="prostate-mil-project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=add_page_number)])

    story = render(markdown, styles, available)
    doc.build(story)
    print(f"wrote {args.output}  ({len(markdown.splitlines())} markdown lines, {doc.page} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
