from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "opening_report_trie_cache.md"
OUTPUT = ROOT / "opening_report_trie_cache.pdf"


def register_fonts() -> tuple[str, str]:
    candidates = {
        "body": [
            Path(r"C:\Windows\Fonts\simfang.ttf"),
            Path(r"C:\Windows\Fonts\simkai.ttf"),
            Path(r"C:\Windows\Fonts\NotoSerifSC-VF.ttf"),
        ],
        "bold": [
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        ],
    }

    body_path = next((path for path in candidates["body"] if path.exists()), None)
    bold_path = next((path for path in candidates["bold"] if path.exists()), None)
    if body_path is None or bold_path is None:
        raise FileNotFoundError("No suitable Chinese fonts were found in C:\\Windows\\Fonts")

    pdfmetrics.registerFont(TTFont("ChineseBody", str(body_path)))
    pdfmetrics.registerFont(TTFont("ChineseBold", str(bold_path)))
    return "ChineseBody", "ChineseBold"


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline_markup(text: str) -> str:
    escaped = xml_escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    return escaped


def is_meta_line(text: str) -> bool:
    return bool(re.match(r"^(学生姓名|学号|专业|导师|学院|日期)：", text))


def build_styles(body_font: str, bold_font: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CnTitle",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=18,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnSubtitle",
            parent=styles["Normal"],
            fontName=bold_font,
            fontSize=14,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=24,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnMeta",
            parent=styles["Normal"],
            fontName=body_font,
            fontSize=12,
            leading=22,
            leftIndent=5.2 * cm,
            spaceAfter=5,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnHeading1",
            parent=styles["Heading1"],
            fontName=bold_font,
            fontSize=16,
            leading=24,
            alignment=TA_LEFT,
            spaceBefore=14,
            spaceAfter=10,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnHeading2",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=13.5,
            leading=22,
            spaceBefore=10,
            spaceAfter=7,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnBody",
            parent=styles["BodyText"],
            fontName=body_font,
            fontSize=11.2,
            leading=20,
            firstLineIndent=22.4,
            spaceAfter=6,
            wordWrap="CJK",
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CnRef",
            parent=styles["BodyText"],
            fontName=body_font,
            fontSize=10.2,
            leading=17,
            firstLineIndent=0,
            leftIndent=0,
            spaceAfter=4,
            wordWrap="CJK",
        )
    )
    return styles


def parse_markdown(markdown_text: str, styles) -> list:
    story = []
    lines = markdown_text.splitlines()
    title_done = False
    meta_done = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("# "):
            story.append(Spacer(1, 3.2 * cm))
            story.append(Paragraph(inline_markup(line[2:].strip()), styles["CnTitle"]))
            title_done = True
            continue

        if title_done and not meta_done and line == "硕士学位论文开题报告初稿":
            story.append(Paragraph(inline_markup(line), styles["CnSubtitle"]))
            continue

        if title_done and is_meta_line(line):
            story.append(Paragraph(inline_markup(line), styles["CnMeta"]))
            continue

        if title_done and not meta_done and line.startswith("## "):
            story.append(PageBreak())
            meta_done = True

        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:].strip()), styles["CnHeading1"]))
            continue

        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:].strip()), styles["CnHeading2"]))
            continue

        if re.match(r"^\[\d+\]", line):
            story.append(Paragraph(inline_markup(line), styles["CnRef"]))
        else:
            story.append(Paragraph(inline_markup(line), styles["CnBody"]))

    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setFont("ChineseBody", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(width / 2, 1.05 * cm, f"- {doc.page} -")
    canvas.restoreState()


def main() -> None:
    body_font, bold_font = register_fonts()
    styles = build_styles(body_font, bold_font)
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), styles)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=2.4 * cm,
        leftMargin=2.4 * cm,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm,
        title="面向多轮大模型服务的学习增强型 Prefix-Tree KV Cache 淘汰策略研究",
        author="Cache-Coliseum",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT)


if __name__ == "__main__":
    main()
