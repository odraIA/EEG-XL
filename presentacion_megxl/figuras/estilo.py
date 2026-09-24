"""Estilo de figuras para la presentacion MEG-XL.

Adaptacion de la casa figures4papers (espinas minimas, leyendas sin marco,
semantica de color azul/verde/rojo, export vectorial) a la paleta y la
tipografia de la plantilla Beamer del TFM.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# --- Paleta hibrida ---------------------------------------------------------
# Verdes de la plantilla como color principal; semantica figures4papers
# (azul = propuesto, rojo = contraste/baseline, neutro = fondo) donde hay
# comparacion explicita.
PAL = dict(
    deepgreen="#0B3D2E",
    greenish="#2A9D8F",
    softgreen="#EAF5F0",
    midgreen="#8BCF8B",
    palegreen="#DDF3DE",
    accent="#E76F51",
    softred="#F9D6CA",
    warning="#F4A261",
    softblue="#DDEAF7",
    softyellow="#FFF0C7",
    muted="#6C757D",
    line="#6E7D77",
    # semantica figures4papers
    blue_main="#0F4D92",
    blue_secondary="#3775BA",
    red_strong="#B64342",
    neutral="#CFCECE",
)

# Ancho de texto de la presentacion (398.34pt = 5.512 in).
FIG_W = 5.51
COL_W = 2.62          # una columna de 0.48\linewidth
BASE = 7.5            # ~\footnotesize del cuerpo (10pt base)
SMALL = 6.3           # ~\scriptsize
TINY = 5.4


def apply_style(font_size=BASE, axes_linewidth=1.0):
    rcParams.update({
        "font.family": ["Latin Modern Sans", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 0.5,
        "xtick.labelsize": font_size - 0.7,
        "ytick.labelsize": font_size - 0.7,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "legend.frameon": False,
        "legend.fontsize": font_size - 0.7,
        "lines.linewidth": 1.1,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 300,
    })


def canvas(w=FIG_W, h=2.2, xlim=(0, 100), ylim=(0, 40)):
    """Lienzo sin ejes, en coordenadas comodas para esquemas."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def save(fig, name, layout=False, pad=0.25):
    """Guarda con el tamano exacto de figsize.

    Nada de bbox_inches="tight": el PDF debe medir lo que dice figsize para
    que \includegraphics[width=\linewidth] no reescale la tipografia.
    """
    if layout:
        fig.tight_layout(pad=pad)
    fig.savefig(f"{name}.pdf")
    plt.close(fig)
    print("  ->", name + ".pdf")


# --- Primitivas de esquema --------------------------------------------------

def box(ax, x, y, w, h, text="", fc="white", ec=None, fs=SMALL, lw=0.9,
        tc="black", weight="normal", pad=0.012, va="center", zorder=3):
    ec = ec or PAL["line"]
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.18}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va=va, fontsize=fs,
                color=tc, fontweight=weight, zorder=zorder + 1, linespacing=1.35)
    return p


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.0, ls="-", mut=5.0, zorder=4):
    color = color or PAL["line"]
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=mut, linewidth=lw, linestyle=ls,
                        color=color, shrinkA=0, shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def strip(ax, x, y, w, h, values, cmap, vmin=None, vmax=None, ec="white",
          lw=0.25, zorder=3, border=None):
    """Fila de celdas coloreadas que representa un vector."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    norm = (values - vmin) / (vmax - vmin + 1e-12)
    cm = plt.get_cmap(cmap)
    cw = w / n
    for i, v in enumerate(norm):
        ax.add_patch(Rectangle((x + i * cw, y), cw, h, facecolor=cm(v),
                               edgecolor=ec, linewidth=lw if n < 60 else 0,
                               zorder=zorder))
    if border:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none",
                               edgecolor=border, linewidth=0.8, zorder=zorder + 1))


def dim_label(ax, x0, x1, y, text, color=None, fs=TINY, tick=0.6):
    """Llave horizontal con la dimension debajo."""
    color = color or PAL["muted"]
    ax.plot([x0, x1], [y, y], color=color, lw=0.7, zorder=2)
    ax.plot([x0, x0], [y - tick, y + tick], color=color, lw=0.7, zorder=2)
    ax.plot([x1, x1], [y - tick, y + tick], color=color, lw=0.7, zorder=2)
    ax.text((x0 + x1) / 2, y - tick * 2.1, text, ha="center", va="top",
            fontsize=fs, color=color)


def caption(ax, x, y, text, fs=TINY, color=None, ha="center"):
    ax.text(x, y, text, ha=ha, va="top", fontsize=fs,
            color=color or PAL["muted"], linespacing=1.3)
