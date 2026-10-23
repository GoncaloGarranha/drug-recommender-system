from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


OUTPUT_DIR = Path(__file__).resolve().parent
WIDTH = 1600
HEIGHT = 920


FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

pdfmetrics.registerFont(TTFont("FigureSans", FONT_REGULAR))
pdfmetrics.registerFont(TTFont("FigureSans-Bold", FONT_BOLD))


INK = colors.HexColor("#222222")
GREEN = colors.HexColor("#C9F0D0")
YELLOW = colors.HexColor("#FFE8B1")
PURPLE = colors.HexColor("#D9C8FA")
PINK = colors.HexColor("#FFC5C1")
BLUE = colors.HexColor("#CDE8F7")
WHITE = colors.white


def add_text(
    drawing: Drawing | Group,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 25,
    bold: bool = False,
    anchor: str = "middle",
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName="FigureSans-Bold" if bold else "FigureSans",
            fontSize=size,
            fillColor=INK,
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
    stroke_width: float = 5,
) -> None:
    drawing.add(
        Rect(
            x,
            y,
            width,
            height,
            rx=13,
            ry=13,
            fillColor=fill,
            strokeColor=INK,
            strokeWidth=stroke_width,
        )
    )


def add_group_box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    drawing.add(
        Rect(
            x,
            y,
            width,
            height,
            rx=18,
            ry=18,
            fillColor=WHITE,
            fillOpacity=0,
            strokeColor=INK,
            strokeWidth=5,
            strokeDashArray=[16, 12],
        )
    )


def add_right_arrow(drawing: Drawing, x1: float, x2: float, y: float) -> None:
    drawing.add(Line(x1, y, x2 - 20, y, strokeColor=INK, strokeWidth=6))
    drawing.add(
        Polygon(
            [x2 - 24, y - 15, x2, y, x2 - 24, y + 15],
            fillColor=INK,
            strokeColor=INK,
        )
    )


def add_up_arrow(drawing: Drawing, x: float, y1: float, y2: float) -> None:
    drawing.add(Line(x, y1, x, y2 - 20, strokeColor=INK, strokeWidth=6))
    drawing.add(
        Polygon(
            [x - 15, y2 - 24, x, y2, x + 15, y2 - 24],
            fillColor=INK,
            strokeColor=INK,
        )
    )


def add_graph_icon(drawing: Drawing, x: float, y: float) -> None:
    nodes = [
        (x, y + 45),
        (x + 55, y + 80),
        (x + 115, y + 55),
        (x + 30, y),
        (x + 100, y - 5),
    ]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2), (1, 4)]
    for a, b in edges:
        drawing.add(
            Line(
                nodes[a][0],
                nodes[a][1],
                nodes[b][0],
                nodes[b][1],
                strokeColor=INK,
                strokeWidth=3,
            )
        )
    for nx, ny in nodes:
        drawing.add(
            Rect(
                nx - 9,
                ny - 9,
                18,
                18,
                rx=9,
                ry=9,
                fillColor=WHITE,
                strokeColor=INK,
                strokeWidth=3,
            )
        )


def build_figure() -> Drawing:
    drawing = Drawing(WIDTH, HEIGHT)

    # Input graph
    add_box(drawing, 55, 505, 250, 250, fill=GREEN)
    add_graph_icon(drawing, 120, 650)
    add_text(drawing, 180, 610, "Grafo de treino", size=28, bold=True)
    add_text(drawing, 180, 570, "INDICATION de treino", size=18)
    add_text(drawing, 180, 540, "+ 4 relações de contexto", size=18)
    add_text(drawing, 180, 514, "+ relações inversas", size=18)

    add_right_arrow(drawing, 305, 385, 637)

    # R-GCN encoder
    add_group_box(drawing, 365, 365, 700, 470)
    add_text(drawing, 715, 786, "Codificador R-GCN", size=34, bold=True)

    add_box(drawing, 410, 565, 180, 145, fill=YELLOW)
    add_text(drawing, 500, 652, "Embeddings", size=25, bold=True)
    add_text(drawing, 500, 618, "iniciais", size=25, bold=True)
    add_text(drawing, 500, 582, "dimensão 128", size=20)

    add_right_arrow(drawing, 590, 645, 637)

    add_box(drawing, 645, 565, 165, 145, fill=PURPLE)
    add_text(drawing, 727, 650, "Camada", size=25, bold=True)
    add_text(drawing, 727, 616, "R-GCN 1", size=25, bold=True)
    add_text(drawing, 727, 582, "com ReLU", size=20)

    add_right_arrow(drawing, 810, 865, 637)

    add_box(drawing, 865, 565, 165, 145, fill=PURPLE)
    add_text(drawing, 947, 650, "Camada", size=25, bold=True)
    add_text(drawing, 947, 616, "R-GCN 2", size=25, bold=True)
    add_text(drawing, 947, 582, "saída final", size=20)

    add_text(drawing, 715, 510, "Em cada camada", size=23, bold=True)
    add_text(
        drawing,
        715,
        477,
        "Transformações por relação com 5 bases e relation gates",
        size=20,
    )
    add_text(
        drawing,
        715,
        444,
        "Ligações residuais, Layer Normalization e dropout",
        size=20,
    )
    add_text(drawing, 715, 411, "Sem normalização explícita pelo grau", size=20)

    add_right_arrow(drawing, 1065, 1135, 637)

    # Decoder
    add_box(drawing, 1135, 565, 190, 145, fill=PINK)
    add_text(drawing, 1230, 650, "DistMult", size=28, bold=True)
    add_text(drawing, 1230, 614, "pontuação de cada", size=19)
    add_text(drawing, 1230, 584, "par candidato", size=19)

    # Query and candidates enter the decoder from below.
    add_box(drawing, 1095, 250, 270, 170, fill=BLUE)
    add_text(drawing, 1230, 367, "Consulta", size=26, bold=True)
    add_text(drawing, 1230, 330, "uma patologia", size=21)
    add_text(drawing, 1230, 295, "+", size=24, bold=True)
    add_text(drawing, 1230, 266, "2 347 fármacos candidatos", size=19)
    add_up_arrow(drawing, 1230, 420, 565)

    add_right_arrow(drawing, 1325, 1390, 637)

    # Output ranking
    add_box(drawing, 1390, 535, 165, 205, fill=GREEN)
    add_text(drawing, 1472, 670, "Ranking", size=28, bold=True)
    add_text(drawing, 1472, 632, "dos fármacos", size=24, bold=True)
    add_text(drawing, 1472, 584, "ordenação por", size=19)
    add_text(drawing, 1472, 553, "pontuação", size=19)

    add_text(
        drawing,
        1230,
        184,
        "A pontuação é utilizada apenas para ordenar os candidatos.",
        size=21,
        bold=True,
    )

    return drawing


if __name__ == "__main__":
    figure = build_figure()
    renderPDF.drawToFile(figure, str(OUTPUT_DIR / "rgcn_architecture_simple.pdf"))
    renderSVG.drawToFile(figure, str(OUTPUT_DIR / "rgcn_architecture_simple.svg"))
