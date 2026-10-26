"""Render the single theory Markdown source to a Japanese, typeset PDF."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import tempfile
from functools import partial
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class TheoryRenderer:
    """Translate the limited prose, math and tables used by the theory paper."""

    def __init__(self, font: Path, bold_font: Path, cache: Path) -> None:
        self.flow = importlib.import_module("reportlab.platypus")
        styles = importlib.import_module("reportlab.lib.styles")
        fonts = importlib.import_module("reportlab.pdfbase.pdfmetrics")
        truetype = importlib.import_module("reportlab.pdfbase.ttfonts")
        fonts.registerFont(truetype.TTFont("Japanese", str(font)))
        fonts.registerFont(truetype.TTFont("JapaneseBold", str(bold_font)))
        fonts.registerFontFamily(
            "Japanese",
            normal="Japanese",
            bold="JapaneseBold",
            italic="Japanese",
            boldItalic="JapaneseBold",
        )
        self.styles = {
            name: styles.ParagraphStyle(
                name,
                fontName=face,
                fontSize=size,
                leading=leading,
                spaceBefore=before,
                spaceAfter=after,
                splitLongWords=True,
                keepWithNext=keep,
                textColor=color,
            )
            for name, face, size, leading, before, after, keep, color in (
                ("Body", "Japanese", 9.6, 16, 0, 6, False, "#202a34"),
                ("H1", "JapaneseBold", 20, 29, 0, 14, True, "#17384c"),
                ("H2", "JapaneseBold", 14, 21, 14, 8, True, "#17384c"),
                ("H3", "JapaneseBold", 11, 17, 10, 6, True, "#17665e"),
                ("Cell", "Japanese", 8.1, 12, 0, 0, False, "#202a34"),
                ("Caption", "Japanese", 8.5, 13, 2, 10, False, "#40515f"),
                ("Code", "Courier", 7.2, 10.5, 5, 9, False, "#202a34"),
            )
        }
        self.cache = cache
        self.width = 467.3
        self.math_count = 0
        self.figure_count = 0
        self.mathtext = importlib.import_module("matplotlib.mathtext")
        self.font_properties = importlib.import_module(
            "matplotlib.font_manager"
        ).FontProperties
        self.images = importlib.import_module("PIL.Image")

    def math_image(
        self, expression: str, *, display: bool
    ) -> tuple[Path, float, float]:
        expression = expression.strip()[2:-2].strip()
        expression = re.sub(r"\\tag\{[^}]+\}", "", expression)
        expression = " ".join(expression.split())
        digest = hashlib.sha256(f"{display}:{expression}".encode()).hexdigest()
        path = self.cache / f"{digest}.png"
        if not path.exists():
            self.mathtext.math_to_image(
                f"${expression}$",
                path,
                prop=self.font_properties(size=11 if display else 9.6),
                dpi=240,
                format="png",
            )
            self.math_count += 1
        with self.images.open(path) as bitmap:
            width, height = bitmap.size
        return path, width * 72 / 240, height * 72 / 240

    def inline(self, node: Any) -> str:
        if node.get("class") == "arithmatex":
            path, width, height = self.math_image(
                "".join(node.itertext()), display=False
            )
            return (
                f'<img src="{escape(str(path), quote=True)}" '
                f'width="{width}" height="{height}" valign="-3"/>'
            )
        text = escape(node.text or "")
        for child in node:
            content = self.inline(child)
            content = self._inline_tag(child, content)
            text += content + escape(child.tail or "")
        return _typography(text.replace("\n", " "))

    @staticmethod
    def _inline_tag(node: Any, content: str) -> str:
        if node.tag in {"strong", "b"}:
            return f"<b>{content}</b>"
        if node.tag in {"em", "i"}:
            return f"<i>{content}</i>"
        if node.tag == "code":
            return f'<font name="Courier" size="7.8">{content}</font>'
        if node.tag == "a":
            return _anchor(node, content)
        return content

    def display_math(self, node: Any) -> Any:
        expression = "".join(node.itertext())
        number = re.search(r"\\tag\{([^}]+)\}", expression)
        path, width, height = self.math_image(expression, display=True)
        scale = min(1.0, (self.width - 37) / width)
        image = self.flow.Image(str(path), width=width * scale, height=height * scale)
        table = self.flow.Table(
            [[image, f"({number[1]})" if number else ""]],
            colWidths=[self.width - 32, 32],
        )
        table.setStyle(
            self.flow.TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Japanese"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return table

    def table(self, node: Any) -> Any:
        rows = [
            [
                self.flow.Paragraph(self.inline(cell), self.styles["Cell"])
                for cell in row
            ]
            for row in node.findall(".//tr")
        ]
        columns = len(rows[0])
        widths = _column_widths(columns, self.width)
        table = self.flow.Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            self.flow.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), "#e6eef2"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, "#456778"),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, "#ccd6dc"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        table.spaceBefore, table.spaceAfter = 6, 12
        return table

    def figure(self, node: Any, source: Path) -> Any:
        path = (source.parent / str(node.get("src"))).resolve()
        with self.images.open(path) as bitmap:
            width, height = bitmap.size
        scale = min(self.width / width, 335 / height)
        figure = self.flow.Image(str(path), width=width * scale, height=height * scale)
        figure.keepWithNext = True
        self.figure_count += 1
        return figure

    def blocks(self, document: Any, source: Path) -> list[Any]:
        result = []
        for node in document:
            result.extend(self.block(node, source))
        return result

    def block(self, node: Any, source: Path) -> list[Any]:
        if node.get("class") == "arithmatex":
            return [self.display_math(node)]
        if node.tag == "table":
            return [self.table(node)]
        if node.tag == "pre":
            return [
                self.flow.Preformatted("".join(node.itertext()), self.styles["Code"])
            ]
        if node.tag in {"ol", "ul"}:
            return self.list_items(node)
        image = node.find("img")
        if node.tag == "p" and image is not None:
            return [self.figure(image, source)]
        return [self.paragraph(node)]

    def paragraph(self, node: Any) -> Any:
        style = {"h1": "H1", "h2": "H2", "h3": "H3"}.get(node.tag, "Body")
        if (node.text or "").startswith("図"):
            style = "Caption"
        return self.flow.Paragraph(self.inline(node), self.styles[style])

    def list_items(self, node: Any) -> list[Any]:
        return [
            self.flow.Paragraph(
                self.inline(item),
                self.styles["Body"],
                bulletText=f"{index}." if node.tag == "ol" else "•",
            )
            for index, item in enumerate(node, 1)
        ]


def _anchor(node: Any, content: str) -> str:
    identity = node.get("id")
    if identity:
        return f'<a name="{escape(identity, quote=True)}"/>{content}'
    link = node.get("href", "")
    if link.startswith(("https://", "http://", "#")):
        return (
            f'<link href="{escape(link, quote=True)}" color="#175a80">{content}</link>'
        )
    return content


def _typography(text: str) -> str:
    superscript = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    subscript = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐᵦₘ", "0123456789abm")

    def as_super(match: re.Match[str]) -> str:
        return f"<super>{match[0].translate(superscript)}</super>"

    def as_sub(match: re.Match[str]) -> str:
        return f"<sub>{match[0].translate(subscript)}</sub>"

    text = re.sub(
        "[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+",
        as_super,
        text,
    )
    return re.sub(
        "[₀₁₂₃₄₅₆₇₈₉ₐᵦₘ]+",
        as_sub,
        text,
    )


def _column_widths(columns: int, width: float) -> list[float]:
    fractions = {
        2: [0.39, 0.61],
        3: [0.27, 0.30, 0.43],
        4: [0.34, 0.23, 0.23, 0.20],
        5: [0.24, 0.20, 0.20, 0.19, 0.17],
    }
    return [width * value for value in fractions.get(columns, [1 / columns] * columns)]


def _page(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Japanese", 8)
    canvas.setFillColor("#526773")
    canvas.drawString(
        42, 813, "Plasma Global Model 0.5.0 | Theory and Numerical Verification"
    )
    canvas.drawRightString(552, 25, str(document.page))
    canvas.restoreState()


def _outline(document: Any, flowable: Any) -> None:
    style = getattr(flowable, "style", None)
    if style is None or style.name != "H2":
        return
    count = getattr(document, "theory_outline_count", 0) + 1
    document.theory_outline_count = count
    identity = f"theory-section-{count}"
    document.canv.bookmarkPage(identity)
    document.canv.addOutlineEntry(flowable.getPlainText(), identity, level=0)


def render(source: Path, output: Path, font: Path, bold_font: Path) -> tuple[int, int]:
    markdown = importlib.import_module("markdown")
    html = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    safe_xml = importlib.import_module("defusedxml.ElementTree")
    document = safe_xml.fromstring(f"<document>{html}</document>")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="theory-math-") as temp:
        renderer = TheoryRenderer(font, bold_font, Path(temp))
        story = renderer.blocks(document, source)
        page = renderer.flow.SimpleDocTemplate(
            str(output),
            pagesize=(595.3, 841.9),
            leftMargin=64,
            rightMargin=64,
            topMargin=49,
            bottomMargin=44,
            title="体積平均プラズマモデルの理論と数値検証",
            author="Plasma Global Model technical documentation",
        )
        page.afterFlowable = partial(_outline, page)
        page.build(story, onFirstPage=_page, onLaterPages=_page)
        return renderer.math_count, renderer.figure_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs/THEORY.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, required=True, help="Japanese TrueType font"
    )
    parser.add_argument("--bold-font", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    figures = render(
        args.source.resolve(),
        args.output.resolve(),
        args.font.resolve(),
        (args.bold_font or args.font).resolve(),
    )
    print(f"{args.output}: {figures[0]} math images, {figures[1]} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
