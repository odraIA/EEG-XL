# Presentación TFM

Proyecto LaTeX Beamer para una defensa de aproximadamente 15 minutos del TFM:

**Transferencia de modelos MEG de contexto largo a EEG para clasificación de palabras en lectura natural**.

## Archivos

- `main.tex`: presentación Beamer 16:9, visual y con poco texto.
- `guion.md`: guion cronometrado diapositiva por diapositiva.
- `Makefile`: comandos de compilación y limpieza.

La presentación reutiliza recursos de la memoria cuando están disponibles:

- `../memoria/imagenes/eeg.png` para la diapositiva del casco EEG.
- `../memoria/figuras/fig_5_2_ds004408_val_balanced_top10_250.pdf` para la diapositiva de curvas de validación.

Si alguno de esos archivos no existe en el entorno de compilación, `main.tex` contiene figuras TikZ de reserva para que la presentación siga compilando.

## Compilación

Desde la raíz de la repo:

```bash
cd presentacion_tfm
make
```

O manualmente:

```bash
pdflatex main.tex
pdflatex main.tex
```

El resultado será `main.pdf`.

## Enfoque de la presentación

La estructura está pensada para una defensa académica con tono divulgativo:

1. problema y motivación;
2. explicación intuitiva de EEG/MEG;
3. propuesta EEG-XL;
4. datasets y preprocesamiento;
5. entrenamiento y evaluación;
6. resultados;
7. conclusión y trabajo futuro.

## Sobre llevar el casco EEG

La presentación incluye una diapositiva específica para integrarlo. La recomendación es llevarlo como recurso físico breve, no como demostración técnica dependiente de software. La explicación debe reforzar tres ideas:

- el EEG no lee pensamientos;
- mide señales débiles en microvoltios;
- precisamente por el ruido hacen falta preprocesamiento, preentrenamiento y evaluación cuidadosa.
