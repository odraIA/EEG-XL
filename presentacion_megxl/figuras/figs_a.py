# -*- coding: utf-8 -*-
"""Figuras A: recorrido completo, VQ, RVQ y formas del tokenizador."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from estilo import (PAL, FIG_W, COL_W, BASE, SMALL, TINY,
                    apply_style, canvas, save, box, arrow, dim_label)


# ---------------------------------------------------------------- 1. pipeline
def fig_pipeline():
    apply_style()
    fig, ax = canvas(FIG_W, 2.05, xlim=(0, 100), ylim=(0, 40))

    etapas = [
        ("MEG crudo",            "$306 \\times 150\\,000$",       "white",          PAL["muted"]),
        ("Preproceso",           "$306 \\times 7\\,500$",         PAL["softyellow"], PAL["warning"]),
        ("BioCodec",             "$306 \\times 625 \\times 6$",   PAL["softblue"],   PAL["blue_secondary"]),
        ("Embeddings",           "$306 \\times 625 \\times 512$", PAL["softgreen"],  PAL["greenish"]),
        ("Enmascarado",          "40 % tapado",                   PAL["softred"],    PAL["accent"]),
        ("Transformer\ncriss-cross", "$306 \\times 625 \\times 512$", PAL["palegreen"], PAL["deepgreen"]),
        ("Cabezal",              "$306{\\times}625{\\times}6{\\times}256$", "white",  PAL["muted"]),
    ]
    n = len(etapas)
    gap, x0 = 1.7, 1.2
    w = (100 - 2 * x0 - (n - 1) * gap) / n
    y, h = 14.5, 11.5
    centros = []
    for i, (titulo, forma, fc, ec) in enumerate(etapas):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, titulo, fc=fc, ec=ec, fs=SMALL, weight="bold", lw=1.0)
        ax.text(x + w / 2, y - 1.6, forma, ha="center", va="top",
                fontsize=TINY, color=PAL["muted"])
        centros.append((x, x + w))
        if i:
            arrow(ax, centros[i - 1][1] + 0.25, y + h / 2, x - 0.25, y + h / 2, lw=0.9)

    # perdida: del cabezal al transformer
    ytop = y + h + 3.0
    ax.plot([centros[6][0] + w / 2, centros[6][0] + w / 2], [y + h, ytop],
            color=PAL["accent"], lw=0.9)
    ax.plot([centros[5][0] + w / 2, centros[6][0] + w / 2], [ytop, ytop],
            color=PAL["accent"], lw=0.9)
    arrow(ax, centros[5][0] + w / 2, ytop, centros[5][0] + w / 2, y + h + 0.3,
          color=PAL["accent"], lw=0.9)
    ax.text((centros[5][0] + centros[6][1]) / 2 - w / 2, ytop + 1.0,
            "gradiente de la pérdida", ha="center", va="bottom",
            fontsize=TINY, color=PAL["accent"])

    # etiquetas: del tokenizador al cabezal
    ybot = y - 7.2
    ax.plot([centros[2][0] + w / 2, centros[2][0] + w / 2], [y - 5.2, ybot],
            color=PAL["deepgreen"], lw=0.9, ls=(0, (3, 2)))
    ax.plot([centros[2][0] + w / 2, centros[6][0] + w / 2], [ybot, ybot],
            color=PAL["deepgreen"], lw=0.9, ls=(0, (3, 2)))
    arrow(ax, centros[6][0] + w / 2, ybot, centros[6][0] + w / 2, y - 5.0,
          color=PAL["deepgreen"], lw=0.9, ls=(0, (3, 2)))
    ax.text((centros[2][0] + centros[6][0]) / 2 + w / 2, ybot - 1.2,
            "los tokens originales son la respuesta correcta",
            ha="center", va="top", fontsize=TINY, color=PAL["deepgreen"])
    save(fig, "fig_pipeline")


# --------------------------------------------------------------------- 2. VQ
def fig_vq():
    apply_style(font_size=SMALL)
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    proto = rng.uniform(0.10, 0.90, size=(13, 2))
    # se elige un prototipo central y se coloca la entrada a una distancia
    # visible, para que la flecha del residuo se lea bien
    k = int(np.argmin(np.linalg.norm(proto - 0.5, axis=1)))
    entrada = proto[k] + np.array([0.155, 0.135])

    ax.scatter(proto[:, 0], proto[:, 1], s=16, color=PAL["greenish"],
               zorder=3, linewidths=0)
    ax.scatter(*proto[k], s=36, color=PAL["deepgreen"], zorder=4, linewidths=0)
    ax.scatter(*entrada, s=32, color=PAL["accent"], zorder=5, marker="D",
               linewidths=0)
    ax.annotate("", xy=proto[k], xytext=entrada, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=PAL["accent"],
                                lw=1.1, shrinkA=3, shrinkB=3))
    ax.text(entrada[0] + 0.035, entrada[1] + 0.035, "vector\nde entrada",
            fontsize=TINY, color=PAL["accent"], ha="left", va="bottom",
            linespacing=1.25)
    ax.text(proto[k][0] - 0.035, proto[k][1] - 0.035, "prototipo #137",
            fontsize=TINY, color=PAL["deepgreen"], ha="right", va="top")
    mid = (entrada + proto[k]) / 2
    ax.text(mid[0] - 0.025, mid[1] + 0.025, "residuo", fontsize=TINY,
            color=PAL["muted"], ha="right", va="bottom")
    ax.text(0.035, 0.035, "diccionario: $V = 256$ prototipos", fontsize=TINY,
            color=PAL["muted"], ha="left", va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(PAL["neutral"]); s.set_linewidth(0.8)
    ax.set_xlabel("espacio de características", fontsize=TINY,
                  color=PAL["muted"], labelpad=2)
    save(fig, "fig_vq", layout=True)


# -------------------------------------------------------------------- 3. RVQ
def _lloyd(x, K=4, iters=60):
    c = np.quantile(x, np.linspace(0.08, 0.92, K))
    for _ in range(iters):
        idx = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        for k in range(K):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
    idx = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
    return c[idx], c


def fig_rvq():
    apply_style(font_size=SMALL)
    rng = np.random.default_rng(3)
    t = np.linspace(0, 3, 360)
    sig = (1.00 * np.sin(2 * np.pi * 0.85 * t)
           + 0.42 * np.sin(2 * np.pi * 3.1 * t + 0.7)
           + 0.18 * np.sin(2 * np.pi * 7.5 * t + 1.9)
           + 0.05 * rng.standard_normal(t.size))

    # RVQ escalar real: 6 niveles, 4 codigos por nivel (demo legible)
    Q, K = 6, 4
    residuo = sig.copy()
    recon = np.zeros_like(sig)
    niveles, rmse = [], [np.sqrt(np.mean(sig ** 2))]
    for q in range(Q):
        aprox, _ = _lloyd(residuo, K)
        niveles.append((residuo.copy(), aprox.copy()))
        recon = recon + aprox
        residuo = residuo - aprox
        rmse.append(np.sqrt(np.mean(residuo ** 2)))

    fig = plt.figure(figsize=(FIG_W, 2.00))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.35, 1.0], hspace=0.55,
                          wspace=0.28)
    filas = [
        (0, "Nivel 1: aproximar la señal", sig, niveles[0][1], PAL["deepgreen"], PAL["accent"]),
        (1, "Nivel 2: aproximar el residuo", niveles[1][0], niveles[1][1], PAL["muted"], PAL["warning"]),
        (2, "Nivel 3: aproximar el nuevo residuo", niveles[2][0], niveles[2][1], PAL["muted"], PAL["blue_secondary"]),
    ]
    for r, titulo, base, aprox, cb, ca in filas:
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(t, base, color=cb, lw=0.8, alpha=0.85)
        ax.step(t, aprox, where="mid", color=ca, lw=1.0)
        ax.set_ylim(-1.75, 1.75)
        ax.set_yticks([])
        ax.set_xticks([] if r < 2 else [0, 1, 2, 3])
        if r == 2:
            ax.set_xlabel("tiempo (s)", fontsize=TINY, labelpad=1)
        ax.set_title(titulo, fontsize=TINY, loc="left", pad=1.5,
                     color=PAL["deepgreen"], fontweight="bold")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(PAL["neutral"])

    axr = fig.add_subplot(gs[:, 1])
    axr.plot(np.arange(Q + 1), rmse, marker="o", ms=2.6,
             color=PAL["blue_main"], lw=1.1)
    axr.set_yscale("log")
    axr.set_xlabel("niveles acumulados", fontsize=TINY, labelpad=1)
    axr.set_ylabel("error RMS", fontsize=TINY, labelpad=1)
    axr.set_xticks(range(0, Q + 1))
    axr.tick_params(labelsize=TINY, length=2, pad=1)
    axr.text(0.97, 0.94, "cada nivel\ndivide el error", transform=axr.transAxes,
             fontsize=TINY, ha="right", va="top", color=PAL["muted"])
    for s in ("left", "bottom"):
        axr.spines[s].set_color(PAL["neutral"])
    save(fig, "fig_rvq", layout=True)


# ------------------------------------------------------- 4. formas del tokenizador
def _bloque3d(ax, x, y, w, h, prof, fc, ec, lw=0.9, z=3):
    """Prisma en proyeccion isometrica sencilla."""
    dx, dy = prof * 0.62, prof * 0.55
    ax.add_patch(Polygon([(x, y + h), (x + dx, y + h + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=fc, edgecolor=ec, lw=lw,
                         alpha=0.55, zorder=z))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, facecolor=fc, edgecolor=ec, lw=lw,
                         alpha=0.75, zorder=z))
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                           zorder=z + 1))


def fig_tokenizer_shapes():
    apply_style()
    fig, ax = canvas(FIG_W, 1.92, xlim=(0, 100), ylim=(0, 44))

    # bloque de entrada
    _bloque3d(ax, 4, 16, 26, 18, 0, PAL["softyellow"], PAL["warning"])
    ax.text(17, 25, "señal preprocesada", ha="center", va="center",
            fontsize=SMALL, fontweight="bold", color=PAL["deepgreen"])
    dim_label(ax, 4, 30, 14.0, "$T = 7\\,500$ muestras (150 s a 50 Hz)")
    ax.text(2.4, 25, "$C = 306$ sensores", rotation=90, ha="center",
            va="center", fontsize=TINY, color=PAL["muted"])

    arrow(ax, 32, 25, 42, 25, lw=1.1, mut=6)
    ax.text(37, 27.0, "BioCodec", ha="center", va="bottom", fontsize=SMALL,
            fontweight="bold", color=PAL["blue_main"])
    ax.text(37, 23.2, "canal a canal,\ncongelado", ha="center", va="top",
            fontsize=TINY, color=PAL["muted"])

    # bloque de salida: Q laminas
    for q in range(6):
        _bloque3d(ax, 50 + q * 1.9, 16 + q * 1.15, 22, 18, 0,
                  PAL["softblue"], PAL["blue_secondary"], lw=0.7, z=3 + q)
    ax.text(61, 25, "tokens", ha="center", va="center", fontsize=SMALL,
            fontweight="bold", color=PAL["blue_main"], zorder=20)
    dim_label(ax, 50, 72, 14.0, "$T' = 625$ pasos")
    ax.text(85.5, 26.5, "$Q = 6$ niveles\nresiduales", ha="left", va="center",
            fontsize=TINY, color=PAL["blue_main"])
    arrow(ax, 84.5, 24.0, 79.0, 18.5, color=PAL["blue_main"], lw=0.7, mut=4)

    # regla de compresion
    ybase = 6.0
    for i in range(12):
        ax.add_patch(Rectangle((4 + i * 1.6, ybase), 1.4, 2.2,
                               facecolor=PAL["softyellow"],
                               edgecolor=PAL["warning"], lw=0.5))
    arrow(ax, 25, ybase + 1.1, 31, ybase + 1.1, lw=0.8, mut=4)
    ax.add_patch(Rectangle((32, ybase), 4.6, 2.2, facecolor=PAL["softblue"],
                           edgecolor=PAL["blue_secondary"], lw=0.7))
    ax.text(38.5, ybase + 1.1, "12 muestras $\\rightarrow$ 1 paso de token "
            "($r = 12$), es decir 0,24 s por token",
            ha="left", va="center", fontsize=TINY, color=PAL["muted"])
    save(fig, "fig_tokenizer_shapes")


if __name__ == "__main__":
    fig_pipeline(); fig_vq(); fig_rvq(); fig_tokenizer_shapes()
