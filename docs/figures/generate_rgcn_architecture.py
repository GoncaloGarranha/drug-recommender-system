from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


OUTPUT_DIR = Path(__file__).resolve().parent
WIDTH = 1000
HEIGHT = 980


FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

pdfmetrics.registerFont(TTFont("FigureSans", FONT_REGULAR))
pdfmetrics.registerFont(TTFont("FigureSans-Bold", FONT_BOLD))


INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#526075")
BLUE = colors.HexColor("#2E5F8A")
BLUE_LIGHT = colors.HexColor("#EAF2F8")
SLATE_LIGHT = colors.HexColor("#F4F6F8")
GOLD_LIGHT = colors.HexColor("#FFF4D6")
GREEN_LIGHT = colors.HexColor("#EAF6EE")
WHITE = colors.white


def add_text(
    drawing: Drawing | Group,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 18,
    bold: bool = False,
    colour=INK,
    anchor: str = "middle",
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName="FigureSans-Bold" if bold else "FigureSans",
            fontSize=size,
            fillColor=colour,
            textAnchor=anchor,
        )
    )


def add_box(
    drawing: Drawing | Group,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill,
    stroke=BLUE,
    radius: float = 12,
    stroke_width: float = 1.6,
) -> None:
    drawing.add(
        Rect(
            x,
            y,
            width,
            height,
            rx=radius,
            ry=radius,
            fillColor=fill,
            strokeColor=stroke,
            strokeWidth=stroke_width,
        )
    )


def add_down_arrow(drawing: Drawing | Group, x: float, y_top: float, y_bottom: float) -> None:
    drawing.add(Line(x, y_top, x, y_bottom + 11, strokeColor=BLUE, strokeWidth=2.2))
    drawing.add(
        Polygon(
            [x - 7, y_bottom + 13, x, y_bottom, x + 7, y_bottom + 13],
            fillColor=BLUE,
            strokeColor=BLUE,
        )
    )


def add_right_arrow(drawing: Drawing | Group, x_left: float, x_right: float, y: float) -> None:
    drawing.add(Line(x_left, y, x_right - 11, y, strokeColor=BLUE, strokeWidth=2.2))
    drawing.add(
        Polygon(
            [x_right - 13, y - 7, x_right, y, x_right - 13, y + 7],
            fillColor=BLUE,
            strokeColor=BLUE,
        )
    )


def build_figure() -> Drawing:
    drawing = Drawing(WIDTH, HEIGHT)

    # Training graph
    add_box(drawing, 70, 830, 860, 115, fill=BLUE_LIGHT)
    add_text(drawing, 500, 912, "Grafo utilizado no treino", size=23, bold=True)
    add_text(
        drawing,
        500,
        881,
        "INDICATION (treino) · CONTRAINDICATION · HAS_INGREDIENT",
        size=16,
    )
    add_text(
        drawing,
        500,
        855,
        "ASSOCIATED_WITH_GENE · INTERACTS_WITH_GENE · relações inversas",
        size=16,
    )

    add_down_arrow(drawing, 500, 830, 800)

    # Initial embeddings
    add_box(drawing, 245, 728, 510, 72, fill=SLATE_LIGHT, stroke=colors.HexColor("#77859A"))
    add_text(drawing, 500, 770, "Embeddings iniciais aprendidos", size=20, bold=True)
    add_text(drawing, 500, 744, "10 380 entidades × 128 dimensões", size=16, colour=MUTED)

    add_down_arrow(drawing, 500, 728, 697)

    # Encoder container
    add_box(drawing, 70, 368, 860, 329, fill=WHITE, stroke=BLUE, radius=14, stroke_width=2.2)
    add_text(drawing, 500, 661, "Encoder R-GCN", size=23, bold=True, colour=BLUE)

    # Layer 1
    add_box(drawing, 105, 438, 350, 184, fill=BLUE_LIGHT, stroke=BLUE)
    add_text(drawing, 280, 587, "Camada R-GCN 1", size=20, bold=True)
    add_text(drawing, 280, 554, "Transformações por relação · 5 bases", size=15)
    add_text(drawing, 280, 528, "Pesos por relação (gates) · self-loop", size=15)
    drawing.add(Line(135, 504, 425, 504, strokeColor=colors.HexColor("#B8C9D8"), strokeWidth=1))
    add_text(drawing, 280, 477, "LayerNorm → ReLU", size=15, bold=True)
    add_text(drawing, 280, 451, "→ ligação residual", size=15, bold=True)

    # Layer 2
    add_box(drawing, 545, 438, 350, 184, fill=BLUE_LIGHT, stroke=BLUE)
    add_text(drawing, 720, 587, "Camada R-GCN 2", size=20, bold=True)
    add_text(drawing, 720, 554, "Transformações por relação · 5 bases", size=15)
    add_text(drawing, 720, 528, "Pesos por relação (gates) · self-loop", size=15)
    drawing.add(Line(575, 504, 865, 504, strokeColor=colors.HexColor("#B8C9D8"), strokeWidth=1))
    add_text(drawing, 720, 477, "LayerNorm → ligação residual", size=15, bold=True)
    add_text(drawing, 720, 451, "sem ReLU na saída", size=15, colour=MUTED)

    add_right_arrow(drawing, 455, 545, 530)
    add_text(
        drawing,
        500,
        397,
        "Sem normalização explícita pelo grau · regularização por dropout",
        size=15,
        colour=MUTED,
    )

    add_down_arrow(drawing, 500, 368, 337)

    # Final embeddings
    add_box(drawing, 245, 269, 510, 68, fill=SLATE_LIGHT, stroke=colors.HexColor("#77859A"))
    add_text(drawing, 500, 308, "Embeddings finais das entidades", size=20, bold=True)
    add_text(drawing, 500, 283, "representação relacional aprendida", size=15, colour=MUTED)

    add_down_arrow(drawing, 500, 269, 238)

    # Decoder
    add_box(drawing, 120, 132, 760, 106, fill=GOLD_LIGHT, stroke=colors.HexColor("#9A7423"))
    add_text(drawing, 500, 198, "Pontuação DistMult", size=22, bold=True)
    add_text(
        drawing,
        500,
        157,
        "fármaco candidato m  ×  INDICATION  ×  patologia fixa d",
        size=17,
    )

    add_down_arrow(drawing, 500, 132, 102)

    # Ranking output
    add_box(drawing, 235, 38, 530, 64, fill=GREEN_LIGHT, stroke=colors.HexColor("#43845A"))
    add_text(drawing, 500, 74, "Ranking dos fármacos", size=20, bold=True)
    add_text(drawing, 500, 51, "2 347 candidatos ordenados por pontuação", size=15, colour=MUTED)

    return drawing


if __name__ == "__main__":
    figure = build_figure()
    renderPDF.drawToFile(figure, str(OUTPUT_DIR / "rgcn_architecture.pdf"))
    renderSVG.drawToFile(figure, str(OUTPUT_DIR / "rgcn_architecture.svg"))
