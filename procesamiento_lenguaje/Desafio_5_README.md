# Procesamiento de Lenguaje Natural – Desafíos 1 a 4

**Autor:** Rodrigo Mesa Marchi  
**Institución:** FIUBA – Especialización en Inteligencia Artificial

Este conjunto de notebooks recorre distintas etapas del procesamiento de lenguaje natural (PLN), aplicando herramientas y modelos para entender cómo representar, comparar y generar texto con métodos basados en embeddings y redes neuronales.  
Cada desafío se enfoca en una parte distinta del proceso: desde vectorizar documentos hasta entrenar un traductor automático con LSTM.

---

## Desafío 1 – Vectorización y clasificación por similitud

En este primer desafío se trabajó con representaciones vectoriales de documentos y la idea de medir **similaridad semántica**.  
Se tomó un conjunto de textos, generó sus embeddings y analizó cómo se relacionaban entre sí en función de su contenido y etiquetas. A partir de eso, se buscó identificar similitud entre documentos en base a las palabras que contienen. Se utilizaron dos tipos de clasificadores: zero shot y naive bayes. En base a la comparativa de ambos, pudimos observar el desempeño de ambos, y evaluar también las ventajas de la utilización de modelos de machine learning para resolver este tipo de problemas.

Por último, invertimos el problema, realizando clasificación de palabras mas similares, en base al corpus de textos brindados al entrenamiento.

---

## Desafío 2 – Embeddings personalizados con Gensim

En este desafío se entrenaron embeddings usando **Gensim** para un dataset de noticias.  
El objetivo fue construir un espacio de embeddings propio, donde las palabras que comparten contexto queden más cerca unas de otras.

Al analizar los resultados de estas pruebas, se observó cómo las tecnologías de vectorización nos permiten representar las palabras mediante estructuras numéricas. Esto facilita el análisis detallado de distintos aspectos relacionados con ellas. Gracias a estos vectores, es posible identificar las palabras más relacionadas entre sí, teniendo en cuenta el contexto específico en el que fueron utilizadas.

Estas herramientas ofrecen capacidades muy robustas, ya que permiten generar representaciones vectoriales que incorporan información contextual, posibilitando así extraer conclusiones más precisas según el ámbito de uso. En otras palabras, el significado y las relaciones entre palabras pueden variar en función de los documentos utilizados para el entrenamiento.

---

## Desafío 3 – Modelo de lenguaje con tokenización por caracteres

En el tercer desafío implementé un **modelo de lenguaje** que trabaja a nivel de caracteres en lugar de palabras.  
La idea fue experimentar con un enfoque más granular: tokenizar el texto carácter por carácter, preparar el dataset y entrenar una red recurrente (LSTM o GRU) que aprenda a predecir la siguiente secuencia.

El modelo se entrenó sobre un corpus elegido y separado en entrenamiento y validación. De estos entrenamientos, se observó como el modelo adoptaba patrones propios del libro con el que se entrenó, lo que para este caso particular lo volvía algo poco generalizado. De este desfío se observó la necesidad de tener un dataset de gran calidad para realizar los entrenamiento, para así poder obtener un modelo que generaliza mejor.

---

## Desafío 4 – Traducción automática con LSTM

El último desafío fue desarrollar un **modelo de traducción automática** usando una arquitectura **encoder-decoder** con LSTM.  
Se preparó un dataset de pares de oraciones (origen–destino), se tokenizó con _spaCy_ y generaron las secuencias de entrada y salida necesarias para entrenar el modelo.

El proceso incluyó construir los embeddings, definir la arquitectura y entrenar sin limitar el tamaño del dataset.  
El objetivo fue ver cómo el modelo aprende a “entender” una oración en un idioma y reconstruirla en otro. Se hizo uso de una combinación de herramientas de pytorch y pytorch lighting para entrenar de forma eficiente el modelo, llegando a resultados bastante positivos.

---

## Conclusión

Estos cuatro desafíos representan un recorrido completo por distintas etapas del PLN: desde representar texto como vectores hasta generar secuencias y traducirlas con redes neuronales.  
Este desarrollo genera el ciclo de las distintas partes de un proyecto de procesamiento de lenguaje: preprocesamiento, representación, entrenamiento y evaluación.  
Cada notebook aporta una mirada diferente del mismo problema general: cómo transformar el lenguaje en algo que una máquina pueda entender, comparar y generar.
