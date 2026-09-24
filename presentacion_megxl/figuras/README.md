# Figuras de la presentación MEG-XL

Figuras matplotlib en el estilo *figures4papers* adaptado a la plantilla Beamer
del TFM (paleta verde de la presentación + semántica azul/verde/rojo para las
comparaciones; tipografía Latin Modern Sans, la misma del texto).

## Regenerar

```bash
python3 figs_a.py && python3 figs_b.py && python3 figs_c.py
```

Cada PDF se guarda con **el tamaño exacto de `figsize`** (sin `bbox_inches="tight"`),
de modo que `\includegraphics[width=\linewidth]` no reescala la tipografía:
`FIG_W = 5.51` in es el `\textwidth` de la presentación.

## Contenido

| Fichero | Diapositiva |
|---|---|
| `fig_pipeline.pdf` | El recorrido completo de una muestra |
| `fig_vq.pdf` | Cuantización vectorial |
| `fig_rvq.pdf` | RVQ (RVQ escalar real + caída del error RMS) |
| `fig_tokenizer_shapes.pdf` | BioCodec: las cifras exactas |
| `fig_indices_to_vectors.pdf` | De índices a vectores |
| `fig_fourier.pdf` | Características de Fourier gaussianas |
| `fig_sum_embeddings.pdf` | La suma final |
| `fig_crisscross.pdf` | La idea de la atención criss-cross |
| `fig_cost.pdf` | El ahorro (205x) y su escalado |
| `fig_masking.pdf` | Enmascaramiento en bloques de 3 s |
| `fig_mask_ratio.pdf` | Por qué el 40 % |
| `fig_tensor_shapes.pdf` | Resumen de formas del tensor |

`estilo.py` contiene la paleta, los ajustes de `rcParams` y las primitivas de
esquema (`box`, `arrow`, `strip`, `dim_label`).
