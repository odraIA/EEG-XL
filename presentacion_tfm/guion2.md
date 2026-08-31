# Guion de defensa — versión ideas clave

## 0. Portada — 0:00-0:20

* Presentación personal y título del TFM.
* Idea central: transferir una arquitectura aprendida en MEG a EEG.
* Motivación: EEG es más ruidoso, pero más viable, barato y portable.

## 1. ¿Cómo de difícil es leer la mente? — 0:20-1:15

* No se trata de “leer pensamientos”.
* Se intenta aprender una relación estadística entre señales cerebrales y lenguaje.
* Dificultades principales:

  * señal no invasiva muy ruidosa;
  * la respuesta a una palabra no está aislada en el tiempo;
  * pocos datos;
  * mucha variabilidad entre sujetos.

## 2. Intrusivo vs. no intrusivo — 1:15-2:00

* Punto de partida: sistemas invasivos como ECoG consiguen resultados potentes.
* Problema: requieren cirugía, por tanto son poco escalables.
* Métodos no invasivos: EEG, MEG y fMRI.
* EEG destaca por portabilidad y coste.
* Ejemplo visual: aplicaciones llamativas como jugar con EEG, pero el problema científico es mucho más difícil.

## 3. Comparativa de métodos no intrusivos — 2:00-2:45

* Comparar resolución temporal y espacial.
* EEG y MEG:

  * muy buena resolución temporal;
  * adecuados para lenguaje por escala de milisegundos.
* fMRI:

  * buena resolución espacial;
  * mala resolución temporal.
* El TFM se centra en EEG y MEG porque la tarea es clasificar palabras alineadas temporalmente.

## 4. MEG vs EEG — 2:45-3:30

* Ambas captan actividad neuronal relacionada, pero con sensores distintos.
* EEG mide potenciales eléctricos en el cuero cabelludo.
* MEG mide campos magnéticos desde sensores externos.
* EEG es más barato y reproducible.
* MEG aporta modelos y arquitecturas ya entrenadas.
* No basta con aplicar MEG-XL directamente a EEG: hay que adaptar sensores, geometría, ruido y entrada.

## 5. Qué capta realmente un casco de EEG — 3:30-4:15

* El EEG no mide pensamientos.
* Mide diferencias de potencial muy pequeñas.
* La señal útil llega mezclada con:

  * parpadeos;
  * movimiento ocular;
  * tensión muscular;
  * impedancias;
  * movimiento del casco;
  * ruido eléctrico.
* Consecuencia: hacen falta preprocesamiento, contexto largo y preentrenamiento.

## 6. Momento casco — 4:15-4:55

* Enseñar casco.
* Señalar electrodos y contacto con cuero cabelludo.
* Recalcar:

  * no invasivo;
  * portable;
  * barato frente a MEG;
  * pero con señal atenuada y ruidosa.
* Mensaje clave: no es lectura directa del cerebro, sino patrones estadísticos en una tarea controlada.

## 7. Pregunta de investigación — 4:55-5:35

* Punto de partida: MEG-XL.
* Adaptación propuesta: EEG-XL.
* Evaluación: recuperación Top-10 de palabras en OpenNeuro ds004408.
* Pregunta central:
  ¿puede un modelo MEG de contexto largo adaptarse a EEG para clasificación contextual de palabras?
* Aclaración importante:

  * no se generan frases;
  * se evalúa si la palabra correcta aparece entre las 10 mejores candidatas.

## 8. Propuesta: de MEG-XL a EEG-XL — 5:35-6:35

* MEG-XL aporta:

  * contexto largo;
  * aprendizaje autosupervisado;
  * pocos parámetros.
* EEG-XL adapta la entrada a EEG heterogéneo.
* Pipeline:

  * señal EEG continua;
  * ventanas de 150 segundos;
  * BioCodec tokeniza por canal;
  * posiciones 3D, tipo de sensor y máscaras;
  * Transformer criss-cross;
  * fine-tuning con embeddings T5.
* Cambio clave: aceptar distintos cascos, canales ausentes y geometrías diferentes.

## 9. Datasets — 6:35-7:25

* Problema: hay pocos datasets EEG de escucha continua.
* Estrategia de entrenamiento progresivo:

  * primero lectura: ZuCo 2.0 y Nieuwland;
  * después escucha: SparrKULee y ds007808;
  * ds004408 se reserva para fine-tuning y test.
* Razón metodológica:

  * evitar contaminación de la evaluación;
  * ordenar datasets de lo más general a lo más parecido a la tarea final.

## 10. Preprocesamiento común — 7:25-8:10

* Objetivo: convertir datasets EEG distintos en una entrada compatible.
* Dos frecuencias distintas:

  * frecuencia de la señal EEG: lo que se conserva con el filtrado;
  * frecuencia de muestreo: muestras por segundo.
* Filtrado: 0,1-40 Hz.
* Remuestreo: 50 Hz.
* Por Nyquist, la banda efectiva queda limitada a unos 25 Hz.
* Compromiso: se pierde banda alta, pero se reduce coste y se permiten ventanas largas.
* Ventanas completas de 150 segundos con posiciones 3D y máscaras.

## 11. Preentrenamiento autosupervisado — 8:10-9:05

* Aquí el modelo aprende señal cerebral, no palabras.
* BioCodec convierte la señal en tokens discretos.
* Ventana de 150 segundos dividida en bloques de 3 segundos.
* Se enmascara aproximadamente el 40 %.
* El Transformer debe reconstruir los tokens ocultos.
* Ventaja:

  * no requiere etiquetas lingüísticas;
  * aprende patrones temporales, relaciones entre sensores y estructura EEG.
* Contexto largo: permite capturar información más estable de sesión, sujeto y entorno.

## 12. Fine-tuning: recuperación contextual de palabras — 9:05-9:55

* Ahora sí se usa alineación EEG-palabra.
* Para cada palabra:

  * inicio anotado;
  * ventana EEG de 3 segundos;
  * 0,5 s antes y 2,5 s después.
* Se agrupan 50 palabras consecutivas → 150 segundos.
* Salida proyectada a embeddings T5-large de 1024 dimensiones.
* Evaluación por similitud coseno.
* Tarea: comprobar si la palabra correcta está en el Top-10.
* No es palabra aislada: es clasificación contextual.

## 13. Diseño experimental: separar efectos — 9:55-10:50

* Objetivo: no entrenar un único modelo, sino separar efectos.
* Experimento 1:

  * sin preentrenamiento;
  * inicialización aleatoria;
  * control.
* Experimento 2:

  * preentrenamiento EEG desde cero.
* Experimentos 3 y 4:

  * parten de checkpoint MEG-XL;
  * uno redefine embedding como EEG;
  * otro reutiliza embedding MEG.
* Preguntas que responde la tabla:

  * ¿qué pasa sin preentrenar?
  * ¿qué aporta el preentrenamiento EEG?
  * ¿qué añade MEG-XL?
  * ¿importa el tipo de embedding de sensor?

## 14. Resultado principal — 10:50-12:05

* Métrica: Top-10 balanceada sobre 250 palabras frecuentes.
* Azar uniforme: 4 %.
* Resultados:

  * sin preentrenamiento: 4,05 % → azar;
  * EEG preentrenado desde cero: 19,95 %;
  * MEG-XL + embedding EEG: 20,56 %;
  * MEG-XL + embedding MEG: 22,32 %.
* Comparación directa:

  * d'Ascoli et al. reportan aproximadamente 20 % en Broderick/ds004408;
  * el mejor modelo alcanza 22,32 %, unos +2,32 pp;
  * presentarlo como comparación descriptiva, no superioridad definitiva.
* Lectura principal:

  * el preentrenamiento EEG es el salto decisivo;
  * MEG-XL ayuda, pero de forma más moderada;
  * no venderlo como “MEG resuelve EEG”, sino como una inicialización útil.

## 15. Dinámica de validación — 12:05-12:50

* Los mejores checkpoints aparecen pronto.
* La curva sube rápido y luego puede estancarse o degradarse.
* Explicación:

  * EEG ruidoso;
  * poco dato supervisado;
  * riesgo de sobreajuste.
* Mensaje: entrenar más no siempre mejora.
* La selección de checkpoint y la validación son parte clave del protocolo.

## 16. Interpretación de los resultados — 12:50-13:45

* Interpretación positiva pero prudente.
* Viabilidad:

  * la arquitectura se adapta a EEG;
  * supera claramente el azar;
  * el contexto largo es compatible con la tarea.
* Factor dominante:

  * preentrenamiento EEG.
* Aporte adicional:

  * inicialización desde MEG-XL.
* Limitaciones:

  * una única semilla;
  * protocolos no idénticos a trabajos previos;
  * no hay generación libre de texto.
* Conclusión interpretativa: vía viable y medible, no problema resuelto.

## 17. Conclusiones — 13:45-14:45

* Se ha adaptado una arquitectura inspirada en MEG-XL a EEG heterogéneo.
* Se ha entrenado con datasets de lectura y escucha.
* Se ha evaluado en recuperación contextual de palabras sobre ds004408.
* Resultado central:

  * 22,32 % Top-10 balanceada;
  * 5,58 veces el azar.
* Mensaje más importante:

  * sin preentrenamiento → azar;
  * con preentrenamiento EEG → salto principal;
  * MEG-XL suma.
* Trabajo futuro:

  * más semillas;
  * más particiones;
  * más sujetos;
  * estudiar bandas, tokenizadores y generalización;
  * probar registros propios con casco EEG.
* Cierre: una arquitectura de contexto largo pensada para MEG puede adaptarse a EEG en una tarea controlada de clasificación contextual de palabras.

## 18. Gracias — 14:45-15:00

* Cerrar presentación.
* Agradecer atención.
* Abrir turno de preguntas.

# Preguntas probables del tribunal — ideas clave

## ¿Por qué no hacer directamente generación de texto?

* Es una tarea mucho más difícil.
* Requiere más datos y evaluación más compleja.
* Este TFM busca una tarea controlada y medible.
* La recuperación Top-10 permite aislar mejor el efecto de la transferencia MEG-EEG.

## ¿El casco EEG puede leer pensamientos?

* No.
* Registra potenciales eléctricos débiles y ruidosos.
* El modelo aprende asociaciones estadísticas dentro de una tarea concreta.

## ¿Cuál es la contribución principal?

* Adaptar una arquitectura de contexto largo inspirada en MEG-XL a EEG heterogéneo.
* Evaluarla en clasificación contextual de palabras.

## ¿Qué resultado es más importante?

* El salto de 4,05 % a 19,95 %.
* Muestra que el preentrenamiento EEG es el factor más importante.
* MEG-XL aporta mejora adicional, pero más moderada.

## ¿Por qué el mejor modelo reutiliza el embedding MEG?

* Puede actuar como una inicialización útil.
* No significa que la señal siga siendo MEG.
* Hacen falta más experimentos para confirmarlo.

## ¿Qué falta para que el resultado sea más sólido?

* Más semillas.
* Más particiones.
* Más sujetos.
* Protocolos comparables.
* Análisis de bandas, sensores, tokenización y generalización entre sesiones.
