#!/usr/bin/env python3
"""Render the Chinese code-change Markdown report as a neutral A4 PDF.

Requires ReportLab. The Markdown file remains the single editable source.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "docs" / "CODE_CHANGES_REPORT.zh-CN.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "ActuateX_代码修改报告.pdf"
GITHUB_URL = "https://github.com/Functionhx/actuatex"


def _inline_markdown(text: str) -> str:
    value = html.escape(text.strip())
    value = re.sub(
        r"&lt;(https?://[^&]+)&gt;",
        r'<link href="\1" color="#1463a5">\1</link>',
        value,
    )
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font color="#7b2f8e">\1</font>', value)
    return value


def _styles():
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdfmetrics.registerFontFamily(
        "STSong-Light",
        normal="STSong-Light",
        bold="STSong-Light",
        italic="STSong-Light",
        boldItalic="STSong-Light",
    )
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "CJKBody",
        parent=base["BodyText"],
        fontName="STSong-Light",
        fontSize=10.2,
        leading=17,
        textColor=colors.HexColor("#202834"),
        spaceAfter=6,
        alignment=TA_LEFT,
        allowWidows=0,
        allowOrphans=0,
    )
    return {
        "title": ParagraphStyle(
            "Title",
            parent=body,
            fontSize=24,
            leading=34,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#10233f"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=body,
            fontSize=12,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4b6078"),
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=body,
            fontSize=16,
            leading=24,
            textColor=colors.HexColor("#113a63"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=body,
            fontSize=13,
            leading=20,
            textColor=colors.HexColor("#176aa1"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=body,
            fontSize=11.5,
            leading=18,
            textColor=colors.HexColor("#176aa1"),
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "Bullet",
            parent=body,
            leftIndent=13,
            firstLineIndent=-9,
            spaceAfter=2,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=body,
            leftIndent=10,
            rightIndent=10,
            borderColor=colors.HexColor("#53a7d8"),
            borderWidth=0,
            borderPadding=8,
            backColor=colors.HexColor("#eef7fc"),
            textColor=colors.HexColor("#29465b"),
        ),
        "code": ParagraphStyle(
            "Code",
            parent=body,
            fontSize=8.2,
            leading=12,
            leftIndent=7,
            rightIndent=7,
            borderPadding=7,
            backColor=colors.HexColor("#f2f5f8"),
            textColor=colors.HexColor("#1c2a38"),
        ),
        "table": ParagraphStyle(
            "Table",
            parent=body,
            fontSize=8.1,
            leading=11,
            spaceAfter=0,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=body,
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#607080"),
        ),
    }


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _table(lines: list[str], style, available_width: float):
    raw_rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
        if not _is_table_separator(line)
    ]
    column_count = max(len(row) for row in raw_rows)
    for row in raw_rows:
        row.extend([""] * (column_count - len(row)))
    rows = [
        [Paragraph(_inline_markdown(cell), style) for cell in row]
        for row in raw_rows
    ]
    widths = [available_width / column_count] * column_count
    table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173d63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b9c7d3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _parse_markdown(text: str, styles, available_width: float):
    lines = text.splitlines()
    story = []
    index = 0
    first_title = True
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            story.append(Spacer(1, 3))
            index += 1
            continue

        if line.startswith("```"):
            language = line[3:].strip()
            index += 1
            code_lines = []
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index].rstrip("\n"))
                index += 1
            index += 1
            label = f"[{language}]\n" if language else ""
            story.append(Preformatted(label + "\n".join(code_lines), styles["code"]))
            story.append(Spacer(1, 5))
            continue

        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(_table(table_lines, styles["table"], available_width))
            story.append(Spacer(1, 7))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            if first_title and level == 1:
                story.extend(
                    [
                        Spacer(1, 34 * mm),
                        Paragraph(_inline_markdown(title), styles["title"]),
                        Paragraph("Learn. Act. Control.", styles["subtitle"]),
                        Spacer(1, 6 * mm),
                        Paragraph(
                            f'<link href="{GITHUB_URL}" color="#1463a5">{GITHUB_URL}</link>',
                            styles["subtitle"],
                        ),
                        Spacer(1, 18 * mm),
                    ]
                )
                first_title = False
            else:
                story.append(Paragraph(_inline_markdown(title), styles[f"h{level}"]))
            index += 1
            continue

        if line.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].startswith(">"):
                quote_lines.append(lines[index].lstrip("> "))
                index += 1
            story.append(Paragraph(_inline_markdown(" ".join(quote_lines)), styles["quote"]))
            continue

        if re.match(r"^-\s+", line):
            while index < len(lines) and re.match(r"^-\s+", lines[index]):
                story.append(
                    Paragraph(
                        "— " + _inline_markdown(lines[index][2:]),
                        styles["bullet"],
                    )
                )
                index += 1
            continue

        if re.match(r"^\d+\.\s+", line):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index]):
                item_text = re.sub(r"^\d+\.\s+", "", lines[index])
                items.append(ListItem(Paragraph(_inline_markdown(item_text), styles["body"])))
                index += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=20, bulletFontName="STSong-Light"))
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if (
                not candidate
                or candidate.startswith(("#", ">", "```", "|", "- "))
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        story.append(Paragraph(_inline_markdown(" ".join(paragraph_lines)), styles["body"]))
    return story


def _page_decor(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#d4dde5"))
    canvas.setLineWidth(0.5)
    canvas.line(document.leftMargin, 16 * mm, width - document.rightMargin, 16 * mm)
    canvas.setFont("STSong-Light", 7.5)
    canvas.setFillColor(colors.HexColor("#607080"))
    canvas.drawString(document.leftMargin, 10.5 * mm, "ActuateX - 代码修改报告")
    canvas.drawRightString(width - document.rightMargin, 10.5 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def build(source: Path, output: Path) -> None:
    styles = _styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title="ActuateX 强化学习控制平台——代码修改报告",
        author="ActuateX",
        subject="TinyMal multi-simulator reinforcement learning code changes",
    )
    available_width = A4[0] - document.leftMargin - document.rightMargin
    story = _parse_markdown(source.read_text(encoding="utf-8"), styles, available_width)
    document.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
