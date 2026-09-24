# -*- coding: utf-8 -*-
"""Figuras C: coste de la atencion, enmascaramiento, porcentaje de mascara y formas."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from estilo import (PAL, FIG_W, COL_W, BASE, SMALL, TINY,
                    apply_style, canvas, save, box, arrow, dim_label)
from figs_a import _bloque3d

C, TP = 306, 625


# --------------------------------------------------- 9. coste de la atencion
def fig_cost():
    apply_style(font_size=SMALL)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.05),
                             gridspec_kw=dict(wspace=0.32, width_ratios=[0.85, 1.15]))

    completa = (C * TP) ** 2
    cross = C * TP ** 2 + TP * C ** 2

    ax = axes[0]
    barras = ax.bar([0, 1], [completa, cross], width=0.55,
                    color=[PAL["red_strong"], PAL["deepgreen"]],
                    edgecolor="black", linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["atención\ncompleta", "criss-cross"], fontsize=TINY)
    ax.set_ylabel("pares de posiciones", fontsize=TINY, labelpad=1)
    ax.set_ylim(1e7, 6e11)
    ax.tick_params(labelsize=TINY, length=2, pad=1)
    def sci(v):
        e = int(np.floor(np.log10(v)))
        return f"${v / 10 ** e:.2f}\\times 10^{{{e}}}$"

    for x, v, mem in [(0, completa, "73 GB"), (1, cross, "0,36 GB")]:
        ax.text(x, v * 1.8, sci(v), ha="center", va="bottom", fontsize=TINY,
                color="black")
        ax.text(x, v * 0.42, mem, ha="center", va="top", fontsize=TINY,
                color="white", fontweight="bold")
    ax.annotate("", xy=(1, cross * 4.5), xytext=(0, completa * 0.30),
                arrowprops=dict(arrowstyle="-|>", color=PAL["accent"], lw=1.0))
    ax.text(0.52, np.sqrt(completa * cross) * 0.14, r"$\approx 205\times$",
            ha="center", va="center", fontsize=SMALL + 1, color=PAL["accent"],
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.set_title("(a) con 150 s y 306 sensores", fontsize=TINY, loc="left",
                 pad=2, color=PAL["deepgreen"], fontweight="bold")
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["neutral"])

    ax = axes[1]
    dur = np.linspace(3, 150, 200)
    tp = dur * 50 / 12
    ax.plot(dur, (C * tp) ** 2, color=PAL["red_strong"], lw=1.2,
            label="atención completa")
    ax.plot(dur, C * tp ** 2 + tp * C ** 2, color=PAL["deepgreen"], lw=1.2,
            label="criss-cross")
    ax.axvline(150, color=PAL["muted"], lw=0.6, ls=(0, (2, 2)))
    ax.text(146, 4e11, "150 s", fontsize=TINY, color=PAL["muted"],
            ha="right", va="top", rotation=90)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("contexto (s)", fontsize=TINY, labelpad=1)
    ax.set_ylabel("pares de posiciones", fontsize=TINY, labelpad=1)
    ax.tick_params(labelsize=TINY, length=2, pad=1)
    ax.legend(fontsize=TINY, loc="upper left", handlelength=1.3,
              borderaxespad=0.2)
    ax.set_title("(b) la distancia crece con el contexto", fontsize=TINY,
                 loc="left", pad=2, color=PAL["deepgreen"], fontweight="bold")
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PAL["neutral"])
    save(fig, "fig_cost", layout=True)


# ------------------------------------------------------ 10. enmascaramiento
def _senal_autocorrelada(nc, nt, rng, lt=9.0, lc=2.2):
    """Ruido suavizado en tiempo y en sensores (FFT, sin scipy)."""
    x = rng.standard_normal((nc, nt))
    ft = np.fft.rfftfreq(nt)
    fc = np.fft.fftfreq(nc)
    X = np.fft.rfft(x, axis=1) * np.exp(-0.5 * (ft * lt * 2 * np.pi) ** 2)
    x = np.fft.irfft(X, n=nt, axis=1)
    X = np.fft.fft(x, axis=0) * np.exp(-0.5 * (fc * lc * 2 * np.pi) ** 2)[:, None]
    x = np.real(np.fft.ifft(X, axis=0))
    return (x - x.mean()) / (x.std() + 1e-12)


def fig_masking():
    apply_style()
    rng = np.random.default_rng(17)
    nc, nt = 64, 200
    datos = _senal_autocorrelada(nc, nt, rng)

    # 20 bloques de 3 s -> 4 columnas de pantalla cada uno = 40 % del total
    blk, nblk = 4, 20
    libres = list(range(nt // blk))
    rng.shuffle(libres)
    elegidos = sorted(libres[:nblk])
    mascara = np.zeros(nt, dtype=bool)
    for b in elegidos:
        mascara[b * blk:(b + 1) * blk] = True

    cm = LinearSegmentedColormap.from_list(
        "meg", ["#0B3D2E", "#6FA894", "#FFFFFF", "#F0B49C", "#8C3B20"])
    fig, ax = plt.subplots(figsize=(FIG_W, 1.85))
    ax.imshow(datos, cmap=cm, aspect="auto", vmin=-2.6, vmax=2.6,
              interpolation="nearest", extent=[0, nt, nc, 0])
    for b in elegidos:
        ax.add_patch(Rectangle((b * blk, 0), blk, nc, facecolor=PAL["accent"],
                               edgecolor="none", alpha=0.92, zorder=3))
    ax.set_xticks([0, nt / 3, 2 * nt / 3, nt])
    ax.set_xticklabels(["0 s", "50 s", "100 s", "150 s"], fontsize=TINY)
    ax.set_yticks([])
    ax.set_ylabel("306 sensores", fontsize=TINY, labelpad=2)
    ax.tick_params(length=2, pad=1)
    b0 = elegidos[3]
    ax.annotate("bloque de 3 s", xy=((b0 + 0.5) * blk, 4.0),
                xytext=((b0 + 6.5) * blk, 11.0), fontsize=TINY,
                color=PAL["accent"], ha="left", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                arrowprops=dict(arrowstyle="-|>", color=PAL["accent"], lw=0.7))
    ax.set_title("20 bloques de 3 s tapados $=$ 40 % de la secuencia",
                 fontsize=TINY, color=PAL["accent"], loc="right", pad=2)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(PAL["neutral"]); s.set_linewidth(0.7)
    save(fig, "fig_masking", layout=True)


# --------------------------------------------------- 11. porcentaje de mascara
def fig_mask_ratio():
    apply_style(font_size=SMALL)
    fig, ax = plt.subplots(figsize=(FIG_W, 1.35))
    datos = [
        ("BERT", 15, "texto", "#767676", 1),
        ("MEG-XL", 40, "", PAL["deepgreen"], -1),
        ("wav2vec 2.0", 49, "audio", PAL["blue_main"], 1),
        ("MAE", 75, "imagen", PAL["red_strong"], 1),
    ]
    ax.axhline(0, color=PAL["line"], lw=1.0, zorder=1)
    for nom, v, dom, col, lado in datos:
        grande = nom == "MEG-XL"
        ax.plot([v, v], [0, 0.55 * lado], color=col, lw=1.0, zorder=2)
        ax.scatter([v], [0], s=46 if grande else 26, color=col, zorder=3,
                   linewidths=0)
        ax.text(v, (0.68 if lado > 0 else 1.00) * lado, f"{nom}\n{v} %",
                ha="center", va="bottom" if lado > 0 else "top",
                fontsize=TINY, color=col,
                fontweight="bold" if grande else "normal", linespacing=1.25)
        if dom:
            ax.text(v + 2.2, 0.27 * lado, dom, ha="left",
                    va="bottom" if lado > 0 else "top", fontsize=TINY,
                    color=PAL["muted"])
    ax.set_xlim(0, 100); ax.set_ylim(-2.15, 1.80)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=TINY)
    ax.tick_params(length=2, pad=1)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PAL["neutral"])
    ax.text(50, -2.08, "porcentaje de la entrada que se tapa durante el preentrenamiento",
            ha="center", va="bottom", fontsize=TINY, color=PAL["muted"])
    save(fig, "fig_mask_ratio", layout=True)


# --------------------------------------------------------- 12. formas finales
def fig_tensor_shapes():
    apply_style()
    fig, ax = canvas(FIG_W, 2.02, xlim=(0, 100), ylim=(0, 46))

    # (alto = C fijo, ancho ~ log10(T), profundidad ~ canal de caracteristicas)
    etapas = [
        ("MEG crudo",      "$306 \\times 150\\,000$",      14.5, 0.0,  PAL["neutral"],    PAL["muted"], ""),
        ("Preprocesado",   "$306 \\times 7\\,500$",        11.0, 0.0,  PAL["softyellow"], PAL["warning"], ""),
        ("Tokens",         "$306 \\times 625 \\times 6$",   7.2, 2.8,  PAL["softblue"],   PAL["blue_secondary"], ""),
        ("Embeddings",     "$306 \\times 625 \\times 512$", 7.2, 7.2,  PAL["softgreen"],  PAL["greenish"], ""),
        ("Tras 8 capas",   "$306 \\times 625 \\times 512$", 7.2, 7.2,  PAL["palegreen"],  PAL["deepgreen"], ""),
        ("Predicciones",   "$306{\\times}625{\\times}6{\\times}256$", 7.2, 4.0, "white", PAL["muted"], ""),
    ]
    x = 3.0
    h = 17.0
    y = 17.0
    centros = []
    for i, (nom, forma, w, prof, fc, ec, _) in enumerate(etapas):
        _bloque3d(ax, x, y, w, h, prof, fc, ec, lw=0.9, z=3)
        ax.text(x + w / 2 + prof * 0.3, y + h + 4.2, nom, ha="center",
                va="bottom", fontsize=TINY, fontweight="bold",
                color=PAL["deepgreen"], rotation=0)
        ax.text(x + w / 2 + prof * 0.3, y - 1.8, forma, ha="center", va="top",
                fontsize=TINY, color=PAL["muted"])
        centros.append((x, x + w + prof * 0.62))
        x = centros[-1][1] + 3.7
    for i in range(1, len(centros)):
        arrow(ax, centros[i - 1][1] + 0.8, y + h / 2, centros[i][0] - 0.8,
              y + h / 2, lw=0.8, mut=4.5)

    # anotaciones de los dos cambios de escala
    ax.annotate("", xy=(centros[2][0], 12.0), xytext=(centros[1][1], 12.0),
                arrowprops=dict(arrowstyle="-|>", color=PAL["blue_main"], lw=0.8))
    ax.text((centros[1][1] + centros[2][0]) / 2, 10.6,
            "$\\div 12$ en\nel tiempo", ha="center", va="top", fontsize=TINY,
            color=PAL["blue_main"], linespacing=1.25)
    ax.annotate("", xy=(centros[3][0], 12.0), xytext=(centros[2][1], 12.0),
                arrowprops=dict(arrowstyle="-|>", color=PAL["greenish"], lw=0.8))
    ax.text((centros[2][1] + centros[3][0]) / 2, 10.6,
            "$6 \\rightarrow 512$\ncaracterísticas", ha="center", va="top",
            fontsize=TINY, color=PAL["greenish"], linespacing=1.25)
    ax.text(1.0, y + h / 2, "$C = 306$", rotation=90, ha="center", va="center",
            fontsize=TINY, color=PAL["muted"])
    save(fig, "fig_tensor_shapes")


if __name__ == "__main__":
    fig_cost(); fig_masking(); fig_mask_ratio(); fig_tensor_shapes()
