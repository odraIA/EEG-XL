# Guion de defensa - 15 minutos

Presentación: `presentacion_tfm/main.tex`.

La presentación está pensada para tener poco texto visible y apoyarse en una explicación oral clara. El estilo sigue la segunda presentación de referencia: tema Beamer `Madrid`, verde oscuro, bloques nativos, tablas sobrias y notas de presentador integradas con `\note{...}`.

## 0. Portada - 0:00-0:20

**Qué decir**

Buenas, soy Ricardo Díaz Peris y voy a presentar mi TFM, titulado *Transferencia de modelos MEG de contexto largo a EEG para clasificación contextual de palabras*. La idea central es estudiar si una arquitectura aprendida con MEG puede servir como punto de partida para EEG, que es más ruidoso pero mucho más viable para aplicaciones portables.

## 1. El problema: decodificar el cerebro no es "leer la mente" - 0:20-1:15

**Qué decir**

Antes de entrar en el modelo, conviene aclarar qué problema estamos resolviendo. No estamos leyendo pensamientos de forma directa. Lo que se intenta aprender es una relación estadística entre señales cerebrales no invasivas y estímulos lingüísticos. El objetivo a largo plazo de las BCI y del brain-to-text es traducir señales cerebrales en lenguaje útil para comunicación, pero eso sigue siendo difícil porque la señal es ruidosa, variable, depende del contexto y los datos alineados cerebro-texto son escasos.

## 2. Intrusivo vs. no intrusivo - 1:15-2:00

**Qué decir**

Los métodos invasivos suelen ofrecer mejor señal, pero requieren cirugía. Eso los hace muy potentes en contextos clínicos concretos, pero poco escalables. En cambio, EEG y MEG son no invasivos. MEG tiene mejor calidad de señal, pero es caro y poco portable. EEG es más barato y portable, aunque mucho más ruidoso. Por eso tiene sentido preguntarse si podemos aprender de MEG y transferir algo útil hacia EEG.

## 3. Comparativa de métodos no intrusivos - 2:00-2:45

**Qué decir**

La tabla resume el compromiso entre modalidades. EEG y MEG tienen alta resolución temporal, que es importante para lenguaje porque las palabras ocurren en escalas de milisegundos. fMRI tiene buena resolución espacial, pero peor resolución temporal. El TFM se centra en EEG y MEG porque el objetivo es clasificar palabras alineadas en el tiempo, no reconstruir semántica global a partir de imágenes cerebrales lentas.

## 4. Qué capta realmente un casco de EEG - 2:45-3:45

**Qué decir**

Aquí se puede enseñar el casco. Un casco EEG mide diferencias de potencial en el cuero cabelludo. La señal útil está mezclada con parpadeos, movimiento ocular, tensión muscular, mala impedancia y ruido eléctrico. Por eso el casco no lee pensamientos: mide señales de microvoltios muy contaminadas. Esta es precisamente la razón de que el preprocesamiento, el contexto largo y el preentrenamiento sean necesarios.

**Cómo usar el casco**

Enséñalo durante menos de un minuto. Señala electrodos, regiones aproximadas y cables. No hagas una demo en directo salvo que esté muy probada. Si quieres mostrar señal, mejor llevar una captura preparada.

## 5. Pregunta de investigación - 3:45-4:30

**Qué decir**

La pregunta formal del trabajo es si un modelo MEG de contexto largo puede adaptarse a EEG para clasificación contextual de palabras. El punto de partida es MEG-XL, la adaptación es EEG-XL, y la evaluación se realiza como recuperación Top-10 de palabras en ds004408. Es importante recalcar que no se genera texto libre: se evalúa si la palabra correcta aparece entre las candidatas mejor puntuadas.

## 6. Propuesta: de MEG-XL a EEG-XL - 4:30-5:30

**Qué decir**

La propuesta conserva la idea central de MEG-XL: representar señales cerebrales largas como tokens, añadir información física de sensores y usar un Transformer criss-cross para modelar tiempo y sensores. La adaptación a EEG consiste en soportar distinto número de canales, distintas posiciones y sensores ausentes mediante máscaras y embeddings de sensor.

## 7. Datasets: lectura para adaptar, escucha para acercarse a la tarea final - 5:30-6:20

**Qué decir**

El entrenamiento se organiza en una progresión. Primero se usan datasets de lectura, como ZuCo y Nieuwland, porque aportan EEG lingüístico. Después se pasa a escucha, con SparrKULee y ds007808, porque se acerca más al dominio de palabras habladas. ds004408 se reserva para fine-tuning y evaluación, evitando contaminar el preentrenamiento con el dataset final.

## 8. Preprocesamiento común - 6:20-7:10

**Qué decir**

Para combinar datasets distintos hay que normalizar la entrada. Se filtra la señal, se remuestrea a 50 Hz, se construyen ventanas de 150 segundos y se incorporan posiciones y máscaras de sensores. Un detalle importante es Nyquist: aunque el filtro inicial llegue a 40 Hz, al remuestrear a 50 Hz la banda efectiva queda limitada aproximadamente a 25 Hz.

## 9. Preentrenamiento autosupervisado - 7:10-8:00

**Qué decir**

Durante el preentrenamiento el modelo no predice palabras. BioCodec discretiza la señal por canal, se ocultan bloques temporales y el Transformer aprende a reconstruir esos tokens ocultos. Esto obliga al modelo a aprender regularidades generales del EEG antes de usar etiquetas lingüísticas.

## 10. Fine-tuning: recuperación contextual de palabras - 8:00-8:55

**Qué decir**

En el ajuste supervisado se alinean ventanas EEG con palabras. Cada palabra tiene un inicio temporal, se extrae una ventana alrededor de ella y se trabaja con secuencias de 50 palabras. La salida se proyecta al espacio de embeddings de T5-large, y la evaluación pregunta si la palabra correcta aparece entre las 10 más cercanas por similitud coseno.

## 11. Diseño experimental: separar efectos - 8:55-9:50

**Qué decir**

El diseño experimental compara cuatro condiciones para separar efectos. La condición sin preentrenamiento mide el control. El modelo EEG desde cero mide cuánto aporta el preentrenamiento autosupervisado. Las dos condiciones inicializadas desde MEG-XL permiten medir si hay transferencia desde MEG y si conviene mantener o redefinir el embedding de tipo de sensor.

## 12. Resultado principal - 9:50-11:10

**Qué decir**

El azar en Top-10 con 250 candidatos es un 4 %. Sin preentrenamiento el modelo queda prácticamente en azar, con 4,05 %. Al preentrenar con EEG se sube a 19,95 %. Con inicialización MEG-XL y embedding EEG se alcanza 20,56 %. El mejor resultado es 22,32 % con inicialización MEG-XL y embedding MEG reutilizado. La conclusión principal es que el preentrenamiento EEG produce el salto dominante, y la transferencia desde MEG añade una mejora moderada.

## 13. Dinámica de validación - 11:10-11:55

**Qué decir**

Las curvas muestran que los mejores checkpoints aparecen pronto. Esto es coherente con una tarea de EEG: al hacer fine-tuning durante demasiado tiempo, el modelo puede sobreajustar. La representación aprendida es útil, pero el ajuste supervisado tiene que controlarse cuidadosamente.

## 14. Interpretación de los resultados - 11:55-12:55

**Qué decir**

La lectura honesta es que la adaptación es viable y queda claramente por encima del azar. El resultado no demuestra que el problema de brain-to-text esté resuelto, pero sí que una arquitectura de contexto largo puede transferirse a EEG en una tarea controlada. También hay que ser prudente porque faltan más semillas, más particiones y comparaciones bajo protocolos idénticos.

## 15. Dónde encaja el casco EEG en la defensa - 12:55-13:35

**Qué decir**

Llevar el casco tiene sentido como recurso divulgativo. Sirve para que el tribunal vea físicamente qué tipo de señal se está usando y por qué es difícil. Pero debe ser breve: enseñar electrodos, explicar ruido y volver al modelo. No conviene que el casco se coma la defensa.

## 16. Conclusiones - 13:35-14:45

**Qué decir**

Como conclusión, se ha adaptado una arquitectura inspirada en MEG-XL a EEG, se ha entrenado con datasets de lectura y escucha y se ha evaluado en recuperación de palabras. El mejor modelo alcanza 22,32 % Top-10 balanceada, unas 5,58 veces el azar. La contribución principal es mostrar una vía viable de transferencia MEG-EEG, aunque todavía hacen falta más semillas, más datasets, más análisis de bandas y registros propios con casco EEG.

## 17. Gracias - 14:45-15:00

**Qué decir**

Con esto termino la presentación. Muchas gracias, y quedo abierto a preguntas.

## Preguntas probables del tribunal

### ¿Por qué no hacer directamente generación de texto?

Porque el objetivo del TFM es una tarea controlada y medible. La generación libre requiere muchos más datos, una evaluación más compleja y protocolos diferentes. Este trabajo se centra en comprobar si la transferencia MEG-EEG ayuda en clasificación contextual de palabras.

### ¿El casco EEG puede leer pensamientos?

No. Registra potenciales eléctricos débiles en el cuero cabelludo, mezclados con múltiples artefactos. El modelo no lee pensamientos; aprende asociaciones estadísticas entre señal cerebral y palabras en una tarea experimental concreta.

### ¿Cuál es la contribución principal?

La adaptación de una arquitectura de contexto largo inspirada en MEG-XL a EEG, junto con una evaluación experimental que separa el efecto del preentrenamiento EEG, la inicialización desde MEG y el embedding de sensor.

### ¿Qué resultado es más importante?

El salto de 4,05 % sin preentrenamiento a 19,95 % con preentrenamiento EEG. Ese es el efecto dominante. La transferencia desde MEG aporta una mejora adicional, pero más moderada.

### ¿Qué falta para hacerlo más sólido?

Más semillas, más sujetos, más datasets, comparaciones bajo protocolos idénticos y análisis específicos de bandas, tokenización y generalización entre sesiones.
