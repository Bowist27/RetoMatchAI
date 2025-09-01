# RetoMatchAI

# Proyecto ETL - YouTube México Trending Videos

La primera parte de este reto/ proyecto consiste en seguir y aplicar el proceso de **Extracción, Transformación y Limpieza (ETL)** al dataset escogido, en este caso (Videos en tendencia de YouTube México).  
El objetivo es preparar los datos para un análisis posterior que permita identificar patrones de popularidad y construir un modelo de predicción de visualizaciones.

---

## 1. Extracción
- Se cargó el dataset `MXvideos` desde Kaggle/Google Sheets utilizando pandas.  
- El dataset contiene **40,451 filas y 16 columnas** con información sobre videos en tendencia en México.  
- Decidimos cargar el Dataset en una hoja en Google Sheets para mantener trazabilidad sin descargas manuales del archivo.
- Se realizó una revisión inicial de dimensiones (`shape`), estructura (`info`) y estadísticas descriptivas (`describe`), analizamos las variables que si podiamos usar y cuales no serian para nuestro provecho.  

---

## 2. Limpieza de datos
Acciones realizadas:
- **Duplicados**: se detectaron videos repetidos (`video_id`), investigamos el porque estos aparecian así, decidimos quedarnos de esos videos repetidos con la fila con más vistas (representa el impacto máximo del video). 


- **Columnas eliminadas**: se descartaron columnas irrelevantes para el análisis (`video_id`, `title`, `channel_title`, `tags`, `description`, `thumbnail_link`). Algunos de estos aptos para un análisis categórico pero no de regresión.

- **Valores nulos**: se contabilizaron los valores faltantes en cada columna.  
- **Valores negativos**: se verificó que no hubiera métricas con valores inválidos en columnas numéricas.  

Pendiente por mejorar:
- Manejo de valores nulos en `description`.  
- Reemplazo de `"[none]"` en `tags`.  
- Conversión de fechas a tipo `datetime` (`trending_date`, `publish_time`).  
- Validación de consistencia lógica (ejemplo: si `comments_disabled=True`, entonces `comment_count` debe ser 0).  

---

## 3. Transformación de variables
Acciones realizadas:
- Conversión de columnas booleanas (`comments_disabled`, `ratings_disabled`, `video_error_or_removed`) a enteros (0/1).  
- Creación de la métrica `engagement_rate = (likes + comment_count) / views`.  
- Uso de `groupby` para calcular promedios por categoría.  
- Visualización de `engagement_rate` promedio por categoría mediante gráfico de barras.  
- Exploración de correlaciones entre variables numéricas con un scatter matrix.  

Pendiente por mejorar:
- Extraer variables temporales: año, mes, día, hora de `publish_time`.  
- Crear nuevas métricas:  
  - `likes_ratio` = likes / (likes + dislikes).  
  - `comments_per_view` = comment_count / views.  
  - `days_to_trending` = trending_date – publish_time.  
- Mapear `category_id` a nombres de categorías de YouTube.  

---

## 4. Análisis exploratorio
Acciones realizadas:
- Histogramas de distribución de `views`, `likes`, `dislikes`, `comment_count` en escala logarítmica.  
- Cálculo de métricas promedio por categoría.  
- Visualización de engagement por categoría.  

Pendiente por mejorar:
- Identificación y tratamiento de outliers.  
- Heatmap de correlaciones entre variables.  
- Análisis de categorías con más vistas totales, no solo por engagement promedio.  

---

## 5. Conclusiones
Hasta el momento, se logró limpiar duplicados, depurar columnas irrelevantes, generar una primera métrica de engagement y explorar la distribución de variables clave.  
El dataset ya se encuentra parcialmente transformado, pero todavía requiere la integración de nuevas variables derivadas, la normalización de fechas y un análisis exploratorio más profundo para completar el ciclo ETL.
