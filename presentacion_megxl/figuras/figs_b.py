# -*- coding: utf-8 -*-
"""Figuras B: indices->vectores, rasgos de Fourier, suma de embeddings, criss-cross."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from estilo import (PAL, FIG_W, COL_W, BASE, SMALL, TINY,
                    apply_style, canvas, save, box, arrow, dim_label, strip)

CM_GREEN = LinearSegmentedColormap.from_list("g", ["#FFFFFF", PAL["greenish"], PAL["deepgreen"]])
CM_BLUE = LinearSegmentedColormap.from_list("b", ["#FFFFFF", PAL["blue_secondary"], PAL["blue_main"]])
CM_ORANGE = LinearSegmentedColormap.from_list("o", ["#FFFFFF", PAL["warning"], PAL["accent"]])
CM_VIOLET = LinearSegmentedColormap.from_list("v", ["#FFFFFF", "#C9A8D4", "#6D3B80"])


# ------------------------------------------------- 5. de indices a vectores
def fig_indices_to_vectors():
    apply_style()
    rng = np.random.default_rng(11)
    fig, ax = canvas(FIG_W, 2.02, xlim=(0, 100), ylim=(0, 48))

    indices = [137, 12, 201, 88, 45, 9]
    ys = [38.0, 32.6, 27.2, 21.8, 16.4, 11.0]
    hb = 4.2
    cms = [CM_GREEN, CM_BLUE, CM_ORANGE, CM_VIOLET, CM_GREEN, CM_BLUE]

    ax.text(11.5, 45.6, "1. Consulta", ha="center", fontsize=SMALL,
            fontweight="bold", color=PAL["deepgreen"])
    ax.text(47.0, 45.6, "2. Concatena", ha="center", fontsize=SMALL,
            fontweight="bold", color=PAL["deepgreen"])
    ax.text(72.0, 45.6, "3. Proyecta", ha="center", fontsize=SMALL,
            fontweight="bold", color=PAL["deepgreen"])
    ax.text(91.0, 45.6, "resultado", ha="center", fontsize=SMALL,
            fontweight="bold", color=PAL["deepgreen"])

    vecs = []
    for q, (z, y, cm) in enumerate(zip(indices, ys, cms)):
        box(ax, 0.5, y, 7.2, hb, f"$z_{{{q+1}}}$ = {z}", fc="white",
            ec=PAL["muted"], fs=TINY)
        arrow(ax, 8.2, y + hb / 2, 10.6, y + hb / 2, lw=0.7, mut=4)
        v = rng.random(14)
        vecs.append(v)
        strip(ax, 11.0, y, 15.0, hb, v, cm, 0, 1, border=PAL["muted"])
    ax.text(18.5, 8.6, "un vector por nivel residual ($Q = 6$)",
            ha="center", va="top", fontsize=TINY, color=PAL["muted"])
    ax.text(-1.2, 24.5, "6 índices", rotation=90, ha="center", va="center",
            fontsize=TINY, color=PAL["muted"])

    # concatenacion
    xc, wc, yc, hc = 33.0, 28.0, 21.8, 4.6
    seg = wc / 6
    for q, (v, cm) in enumerate(zip(vecs, cms)):
        strip(ax, xc + q * seg, yc, seg, hc, v, cm, 0, 1, lw=0)
    # limites entre los 6 trozos
    for i in range(1, 6):
        ax.plot([xc + i * wc / 6] * 2, [yc, yc + hc], color="white", lw=0.8,
                zorder=5)
    ax.add_patch(Rectangle((xc, yc), wc, hc, facecolor="none",
                           edgecolor=PAL["muted"], lw=0.8, zorder=6))
    for q, y in enumerate(ys):
        ax.plot([26.6, xc - 0.6], [y + hb / 2, yc + hc / 2],
                color=PAL["neutral"], lw=0.5, zorder=1)
    dim_label(ax, xc, xc + wc, yc - 2.2, "$6 \\times d_{\\mathrm{cb}}$ números")

    # proyeccion
    arrow(ax, xc + wc + 1.0, yc + hc / 2, 64.5, yc + hc / 2, lw=0.9, mut=5)
    mat = rng.random((9, 9))
    xm, ym, wm, hm = 65.5, 15.5, 13.0, 17.0
    for i in range(9):
        strip(ax, xm, ym + i * hm / 9, wm, hm / 9, mat[i], CM_BLUE, 0, 1, lw=0)
    ax.add_patch(Rectangle((xm, ym), wm, hm, facecolor="none",
                           edgecolor=PAL["blue_main"], lw=0.9, zorder=6))
    ax.text(xm + wm / 2, ym - 2.4, "$W_{\\mathrm{proj}}$\n(se aprende)",
            ha="center", va="top", fontsize=TINY, color=PAL["blue_main"])
    arrow(ax, xm + wm + 1.0, yc + hc / 2, 82.0, yc + hc / 2, lw=0.9, mut=5)

    # salida
    xo, wo = 83.0, 16.0
    strip(ax, xo, yc, wo, hc, rng.random(64), CM_GREEN, 0, 1, lw=0,
          border=PAL["deepgreen"])
    dim_label(ax, xo, xo + wo, yc - 2.2, "$d_{\\mathrm{model}} = 512$",
              color=PAL["deepgreen"])
    ax.text(xo + wo / 2, yc + hc + 1.4, "$\\mathbf{h}^{(0)}_{c,t}$",
            ha="center", va="bottom", fontsize=SMALL, color=PAL["deepgreen"])
    save(fig, "fig_indices_to_vectors")


# ------------------------------------------------------ 6. rasgos de Fourier
def fig_fourier():
    apply_style(font_size=SMALL)
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 1.82),
                             gridspec_kw=dict(wspace=0.28, width_ratios=[1, 1]))

    # (a) componentes de gamma(x)
    ax = axes[0]
    x = np.linspace(0, 1, 500)
    for i, (f, c) in enumerate(zip([1.4, 4.6, 11.0],
                                   [PAL["deepgreen"], PAL["accent"], PAL["blue_main"]])):
        ax.plot(x, np.cos(2 * np.pi * f * x) * 0.9 - i * 2.3, color=c, lw=0.9)
    for xv, lab in [(0.44, None), (0.50, None)]:
        ax.axvline(xv, color=PAL["muted"], lw=0.6, ls=(0, (2, 2)))
    ax.annotate("dos sensores próximos", xy=(0.47, 1.45), fontsize=TINY,
                color=PAL["muted"], ha="center", va="bottom")
    ax.set_xlim(0, 1); ax.set_ylim(-5.9, 2.5)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlabel("coordenada del sensor", fontsize=TINY, labelpad=2)
    ax.set_title("(a) $\\gamma(v)$: muchas frecuencias a la vez",
                 fontsize=TINY, loc="left", pad=2, color=PAL["deepgreen"],
                 fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PAL["neutral"])

    # (b) similitud frente a distancia, calculada de verdad
    ax = axes[1]
    n = 400
    u = rng.normal(size=(n, 3))
    pos = u / np.linalg.norm(u, axis=1, keepdims=True)   # casco esferico
    pos = pos * rng.uniform(0.95, 1.0, size=(n, 1))
    sigma, dfour = 1.8, 128
    B = rng.normal(0, sigma, size=(dfour // 2, 3))
    proj = 2 * np.pi * pos @ B.T
    gam = np.concatenate([np.cos(proj), np.sin(proj)], axis=1)

    def cos_sim(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return Mn @ Mn.T

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    iu = np.triu_indices(n, 1)
    dd = d[iu]
    for M, c, lab in [(pos, PAL["red_strong"], "coordenadas crudas"),
                      (gam, PAL["blue_main"], "rasgos de Fourier")]:
        ss = cos_sim(M)[iu]
        bins = np.linspace(0, dd.max(), 26)
        k = np.digitize(dd, bins) - 1
        m = np.array([ss[k == j].mean() if (k == j).any() else np.nan
                      for j in range(len(bins) - 1)])
        ax.plot((bins[:-1] + bins[1:]) / 2, m, color=c, lw=1.2, label=lab)
    ax.axhline(0, color=PAL["neutral"], lw=0.6)
    ax.set_xlabel("distancia entre sensores", fontsize=TINY, labelpad=2)
    ax.set_ylabel("similitud coseno", fontsize=TINY, labelpad=2)
    ax.tick_params(labelsize=TINY, length=2, pad=1)
    ax.legend(fontsize=TINY, loc="upper right", handlelength=1.2,
              borderaxespad=0.2)
    ax.set_title("(b) sensores vecinos, ya distinguibles",
                 fontsize=TINY, loc="left", pad=2, color=PAL["deepgreen"],
                 fontweight="bold")
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["neutral"])
    save(fig, "fig_fourier", layout=True)


# -------------------------------------------------- 7. suma de embeddings
def fig_sum_embeddings():
    apply_style()
    rng = np.random.default_rng(23)
    fig, ax = canvas(FIG_W, 2.05, xlim=(0, 100), ylim=(0, 40))

    filas = [
        ("tokens",      "qué forma de onda hay", CM_GREEN,  PAL["deepgreen"]),
        ("posición",    "dónde está el sensor",  CM_BLUE,   PAL["blue_main"]),
        ("orientación", "hacia dónde mira",      CM_ORANGE, PAL["accent"]),
        ("tipo",        "mag / grad (aprendido)", CM_VIOLET, "#6D3B80"),
    ]
    x0, w, h = 26.0, 34.0, 4.4
    ys = [31.0, 23.4, 15.8, 8.2]
    for (nom, desc, cm, col), y in zip(filas, ys):
        ax.text(24.0, y + h / 2 + 0.9, nom, ha="right", va="center",
                fontsize=TINY, color=col, fontweight="bold")
        ax.text(24.0, y + h / 2 - 1.5, desc, ha="right", va="center",
                fontsize=TINY, color=PAL["muted"])
        strip(ax, x0, y, w, h, rng.random(64), cm, 0, 1, lw=0, border=col)
    for y in ys[:-1]:
        ax.text(x0 + w / 2, y - 1.7, "$+$", ha="center", va="center",
                fontsize=SMALL + 1, color=PAL["muted"])

    ax.text(x0 + w + 3.2, 19.8, "$=$", ha="center", va="center",
            fontsize=SMALL + 2, color=PAL["muted"])
    xo, wo = x0 + w + 7.0, 28.0
    suma = rng.random(64)
    strip(ax, xo, 17.6, wo, 5.0, suma, CM_GREEN, 0, 1, lw=0,
          border=PAL["deepgreen"])
    ax.text(xo + wo / 2, 23.4, "entrada al transformer",
            ha="center", va="bottom", fontsize=TINY, fontweight="bold",
            color=PAL["deepgreen"])
    dim_label(ax, xo, xo + wo, 15.6, "512 números por sensor y por instante",
              color=PAL["deepgreen"])
    dim_label(ax, x0, x0 + w, 5.6, "los cuatro son de 512 dimensiones")
    save(fig, "fig_sum_embeddings")


# ----------------------------------------------------------- 8. criss-cross
def _rejilla(ax, ox, oy, nc, nt, cw, ch, fill="#F3F5F4", ec=PAL["neutral"]):
    for c in range(nc):
        for t in range(nt):
            ax.add_patch(Rectangle((ox + t * cw, oy + c * ch), cw, ch,
                                   facecolor=fill, edgecolor=ec, lw=0.3))


def fig_crisscross():
    apply_style()
    fig, ax = canvas(FIG_W, 1.98, xlim=(0, 100), ylim=(0, 44))
    nc, nt = 7, 14
    cw, ch = 1.75, 2.6
    qc, qt = 3, 7
    paneles = [
        (2.0, "Atención temporal", "fila", PAL["greenish"],
         "cada sensor recorre su propio\ntiempo (625 posiciones)"),
        (35.5, "Atención espacial", "col", PAL["blue_secondary"],
         "cada instante mira todos los\nsensores (306 posiciones)"),
        (69.0, "Tras dos capas", "all", PAL["palegreen"],
         "cualquier sensor en cualquier\ninstante ya es alcanzable"),
    ]
    for ox, titulo, modo, col, pie in paneles:
        oy = 15.0
        _rejilla(ax, ox, oy, nc, nt, cw, ch)
        if modo in ("all",):
            for c in range(nc):
                for t in range(nt):
                    ax.add_patch(Rectangle((ox + t * cw, oy + c * ch), cw, ch,
                                           facecolor=col, edgecolor="white",
                                           lw=0.3, alpha=0.75))
        if modo in ("fila", "all"):
            for t in range(nt):
                ax.add_patch(Rectangle((ox + t * cw, oy + qc * ch), cw, ch,
                                       facecolor=PAL["greenish"],
                                       edgecolor="white", lw=0.3, alpha=0.9))
        if modo in ("col", "all"):
            for c in range(nc):
                ax.add_patch(Rectangle((ox + qt * cw, oy + c * ch), cw, ch,
                                       facecolor=PAL["blue_secondary"],
                                       edgecolor="white", lw=0.3, alpha=0.9))
        ax.add_patch(Rectangle((ox + qt * cw, oy + qc * ch), cw, ch,
                               facecolor=PAL["accent"], edgecolor="white",
                               lw=0.4, zorder=5))
        ax.text(ox + nt * cw / 2, oy + nc * ch + 2.6, titulo, ha="center",
                va="bottom", fontsize=SMALL, fontweight="bold",
                color=PAL["deepgreen"])
        ax.text(ox + nt * cw / 2, oy - 4.4, pie, ha="center", va="top",
                fontsize=TINY, color=PAL["muted"], linespacing=1.3)
    ax.text(0.6, 15.0 + nc * ch / 2, "sensores", rotation=90, ha="center",
            va="center", fontsize=TINY, color=PAL["muted"])
    ax.text(2.0 + nt * cw / 2, 13.4, "tiempo", ha="center", va="top",
            fontsize=TINY, color=PAL["muted"])
    ax.add_patch(Rectangle((41.5, 1.2), 2.1, 2.1, facecolor=PAL["accent"],
                           edgecolor="none"))
    ax.text(44.6, 2.25, "posición consultada", ha="left", va="center",
            fontsize=TINY, color=PAL["accent"])
    save(fig, "fig_crisscross")


if __name__ == "__main__":
    fig_indices_to_vectors(); fig_fourier(); fig_sum_embeddings(); fig_crisscross()
