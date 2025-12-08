# TP2
Para el tp2 se generaron dos archivos principales: 
- TP2_embeddings.ipynb
- TP2_chatbot.py

El primero es un notebook donde se realiza el preprocesado de los CVs, se obtienen los embeddings y se cargan a Pinecone. 

En el segundo archivo se genera el RAG, el cual utiliza los embeddings de Pinecone para mayor contexto, alimentando a un LLM de Groq. Se grabó un video explicando las funcionalidades principales del desarrollo, como tambien comentando los criterios de diseño elegidos. 
**Link al Video:** https://drive.google.com/file/d/1MDzem3Lp3EBK0aYgxLRe7Ny13lffSV8l/view?usp=sharing 

# TP3

Para el tp3, se generó un archivo: TP3_web.py, en donde se define la estructura del agente que realiza las tareas solicitadas en la consigna, como también la pequeña app de streamlit que muestra los resultados y permite interactuar con el agente. Para una mejor explicación, se grabó un video donde se presentan los resultados, y se explican las principales elecciones de diseño.
**Link al Video:** https://drive.google.com/file/d/1ob2dP60B1fSQofQmOt7n7wcgtKzOMG8-/view?usp=sharing
Se agregó además un pequeño video donde se muestra el caso donde no se especifica la persona: 
**Link**: https://drive.google.com/file/d/14esc1x02oDtlkiICyR0JLgfxwZjrYYNK/view?usp=sharing
Como puntos principales de la elección de diseño, se optó por utilizar un agente con dos usos de LLM:
- Parser y estructuración de contenido: se encargar de analizar la solicitud del usuario, formateando cada una de las consultas del texto en una entrada de json. Esto permite obtener una respuesta estructurada, donde por ejemplo si se consulta: "En que universidad estudió Rodrigo Mesa, y donde trabaja Juan García?" La respuesta sería: [{"persona":"rodrigo-mesa", "motivo":"Universidad"}, {"persona":"juan-garcia", "motivo":"Trabajo"}], lo cual procesamos como json y convertimos en un diccionario. 
- Agente de respuesta: en base a la consulta del usuario, y al contexto obtenido desde la base de datos vectorial (Pinecone), el agente da la respuesta.

Para mayor robustez, se generan tres indices diferentes, uno por cada cv. En base a la respuesta del primer modelo de lenguaje, se procesa cada uno de los json correspondientes, y se realizan las busquedas en el indice que corresponde. 

Estas explicaciones de diseño se comentan en mayor profundidad en el video previamente mencionado.