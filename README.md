# RetoMatchAI

## Equipo
- A01710550 - Maxime Vilcocq Parra
- A01710791 - Galo Alejandro del Rio Viggiano
- A01369687 - Ana Karen Toscano Díaz
- A01710367 - José Antonio López Saldaña

# Documentación del Proyecto

## Introducción

El presente proyecto aborda el problema de la **segmentación semántica en imágenes médicas**, específicamente en **tomografías computarizadas (TAC) de pulmón**. Este reto surge en el contexto de la pandemia de **COVID-19**, donde la identificación automática de lesiones pulmonares es de gran relevancia clínica para asistir en el diagnóstico, el seguimiento de la enfermedad y la evaluación de la respuesta al tratamiento.  

El objetivo principal de la competencia de Kaggle es **desarrollar un modelo robusto de segmentación multiclase**, capaz de distinguir entre diferentes estructuras y lesiones pulmonares. En particular, la tarea se centra en identificar:  

- **Fondo**  
- **Pulmón sano**  
- **Lesiones (opacidades, consolidaciones, etc.) asociadas a COVID-19**  

La arquitectura utilizada es una **UNet++**, una variante de la U-Net que introduce conexiones densas entre codificador y decodificador, mejorando la propagación de información multiescala. Además, se emplea una función de pérdida compuesta (**Cross-Entropy + Dice + Focal**) para balancear la influencia de clases mayoritarias y minoritarias, lo cual es crucial dado el desbalance natural en este tipo de datos.  

El código está implementado en **Jupyter Notebook dentro de la plataforma Kaggle**, haciendo uso de librerías clave como `segmentation-models-pytorch`, `albumentations` y `torchvision` para modelado, aumentación de datos y visualización.

---

## Análisis de Datos

### ETL / EDA

El conjunto de datos utilizado proviene del repositorio de **Kaggle COVID-19 CT segmentation**, el cual contiene:  

- **Número de muestras:** Aproximadamente 1000 imágenes de TAC junto con sus máscaras.  
- **Dimensiones:** Cada imagen tiene resolución variable, aunque para el entrenamiento se estandarizan a **256×256 píxeles**.  
- **Clases en las máscaras:**  
  - Clase 0: Fondo  
  - Clase 1: Pulmón  
  - Clase 2: Lesión 1 (ej. opacidades)  
  - Clase 3: Lesión 2 (ej. consolidaciones)  

Durante el análisis exploratorio, se visualizaron imágenes y sus máscaras de segmentación correspondientes, lo que permitió identificar:  

- La gran variabilidad en intensidades y contraste de las TAC.  
- Diferencias en la extensión de las lesiones (desde mínimas hasta bilaterales extensas).  
- Desbalance en la frecuencia de aparición de lesiones, lo cual justifica el uso de **muestreo ponderado** durante el entrenamiento.  

### Preprocesamiento de Datos

Los pasos principales de preprocesamiento fueron:  

1. **Lectura y conversión:**  
   - Las imágenes se cargaron en formato `float32`, dado que contienen intensidades continuas en **Unidades Hounsfield (HU)**.  
   - Las máscaras se procesaron en formato `int16`, representando cada clase como un entero.  

2. **Clipping o windowing:**  
   - Se recortaron los valores de HU al rango `[-1500, 500]`, el más relevante para estructuras pulmonares y lesiones.  

3. **Normalización:**  
   - Se calcularon la media y desviación estándar del dataset para aplicar **estandarización por canal**, asegurando estabilidad en el entrenamiento.  

4. **Redimensionamiento:**  
   - Todas las imágenes y máscaras se ajustaron a **256×256 píxeles**, manteniendo coherencia en los lotes de datos.  

5. **Aumento de datos (Data Augmentation):**  
   Se aplicaron transformaciones de `albumentations` para mejorar la generalización del modelo:  
   - Rotaciones aleatorias  
   - Giros horizontales  
   - Crops aleatorios  
   - Variaciones de brillo y contraste  

6. **Balanceo de clases:**  
   - Se implementó un **WeightedRandomSampler** que asigna mayor probabilidad de muestreo a imágenes con lesiones, evitando que el modelo se sesgue hacia clases mayoritarias.

## Modelo

### Arquitectura del Modelo

El modelo seleccionado para este proyecto es **U-Net++**, una variante mejorada de la **U-Net** clásica, ampliamente utilizada en tareas de segmentación médica. La U-Net original (Ronneberger et al., 2015) introdujo la idea de un **encoder-decoder** con **conexiones de salto (skip connections)**, permitiendo combinar información de bajo y alto nivel para segmentar con precisión estructuras anatómicas.

La **U-Net++** (Zhou et al., 2018) extiende esta idea añadiendo **conexiones densas y convoluciones intermedias** entre capas de codificación y decodificación. Esto reduce la brecha semántica entre características de distinta profundidad y mejora la segmentación en regiones complejas como lesiones pulmonares.

#### Descripción de la Arquitectura

1. **Encoder (contracción):**
   - Compuesto por una red convolucional preentrenada como backbone (e.g., ResNet34, EfficientNet, o VGG16).  
   - Cada bloque del encoder aplica:
     - Convoluciones 2D con kernel de `3x3`, stride `1`, padding `1`.  
     - Función de activación **ReLU**.  
     - Normalización por lotes (**BatchNorm**).  
     - **Max Pooling 2D (2x2, stride 2)** para reducir a la mitad la resolución espacial.  

2. **Decoder (expansión):**
   - Etapas de **upsampling** mediante interpolación bilineal o transposed convolution (stride 2).  
   - Concatenación con los mapas de características del encoder (**skip connections**).  
   - Convoluciones 2D `3x3` + ReLU + BatchNorm para refinar la reconstrucción espacial.  

3. **Conexiones densas (característica de U-Net++):**
   - Entre cada bloque encoder-decoder, se introducen **convoluciones adicionales** que permiten un refinamiento progresivo de las características antes de fusionarlas.  
   - Esto evita la discrepancia entre características profundas (semánticas) y superficiales (espaciales).  

4. **Capa de salida:**
   - Convolución final `1x1` para reducir el número de canales a **4 clases** (fondo, pulmón, dos tipos de lesión).  
   - Activación **Softmax** para obtener probabilidades por clase en cada píxel.  

#### Esquema Resumido

- **Encoder:** 4 bloques convolucionales + downsampling.  
- **Decoder:** 4 bloques convolucionales + upsampling.  
- **Conexiones densas:** entre cada etapa de encoder y decoder.  
- **Output:** tensor `(N, 4, 256, 256)` donde `N` es el tamaño del batch.  

---

### Justificación de la Selección

- La **U-Net** clásica es un estándar en segmentación médica, validada en múltiples estudios.  
- La **U-Net++** mejora el rendimiento en datasets desbalanceados y con estructuras poco definidas (como lesiones COVID-19) gracias a sus **skip connections densos**.  
- El backbone preentrenado (e.g., ResNet34) aprovecha características de redes entrenadas en **ImageNet**, acelerando la convergencia y mejorando la capacidad de generalización.  
- La combinación de pérdidas **Cross-Entropy + Dice + Focal** permite:  
  - Penalizar errores de clase mayoritaria/minoritaria.  
  - Aumentar sensibilidad en lesiones pequeñas.  
  - Mantener buena segmentación en estructuras más grandes.  

## Entrenamiento

### División del Conjunto de Datos

Para garantizar una evaluación justa del modelo, el dataset se dividió en tres subconjuntos:  

- **Entrenamiento (80%)**: utilizado para ajustar los parámetros de la red neuronal.  
- **Validación (20%)**: empleado para ajustar hyperparameters y monitorear el desempeño del modelo durante el entrenamiento.  
- **Pruebas (20% aprox)**: Como un archivo de pruebas externo.  

La partición se realizó de forma **estratificada** cuando fue posible, para asegurar una proporción similar de clases (fondo, pulmón y lesiones) en cada subconjunto.  

---

### Proceso de Entrenamiento

1. **Función de pérdida (criterion):**  
   Se empleó una combinación ponderada de:  
   - **Cross-Entropy Loss**: maneja la clasificación multiclase por píxel.  
   - **Dice Loss**: mejora la segmentación en clases minoritarias (lesiones pequeñas).  
   - **Focal Loss**: refuerza la atención en ejemplos difíciles, mitigando el desbalance de clases.   

2. **Optimizador y tasa de aprendizaje:**  
   - Optimizador: **Adam** (Adaptive Moment Estimation).  
   - Learning rate inicial: `1e-4`.  
   - Estrategia de ajuste: **ReduceLROnPlateau**, disminuyendo la tasa cuando la métrica de validación dejaba de mejorar.  

3. **Hiperparámetros de entrenamiento:**  
   - **Número de epochs:** 50 (con early stopping si no había mejora en validación después de 10 epochs).  
   - **Tamaño del minibatch (batch size):** 6 imágenes.  
   - **Weight decay:** `1e-5` como regularización para evitar overfitting.  

4. **Técnicas de regularización empleadas:**  
   - **Early stopping:** previno sobreajuste al detener el entrenamiento cuando la métrica de validación se estancaba.  
   - **Data augmentation:** actuó como regularizador al aumentar la diversidad del dataset.  
   - **WeightedRandomSampler:** aumentó la frecuencia de muestras con lesiones en cada batch, mejorando la sensibilidad en clases minoritarias.  

---

### Conclusiones sobre los Hiperparámetros

- El **batch size pequeño (6)** resultó adecuado dadas las limitaciones de memoria GPU y la resolución de las imágenes.  
- El **learning rate** tuvo un gran impacto: valores mayores (`1e-3`) causaban divergencia, mientras que valores más bajos (`1e-5`) ralentizaban demasiado el aprendizaje.  
- La combinación de pérdidas **CE + Dice + Focal** fue clave para lograr un equilibrio entre precisión global y sensibilidad a lesiones pequeñas.  
- El **ReduceLROnPlateau** permitió alcanzar una convergencia más estable, evitando caer en mínimos locales.  

---

### Consideraciones de Hardware

El entrenamiento se llevó a cabo en la infraestructura de **Kaggle Notebooks**, utilizando:  

- **GPU NVIDIA Tesla P100 (16 GB VRAM)**.  
- **CPU Intel Xeon** de soporte para operaciones de carga de datos.  
- **Memoria RAM disponible:** 13 GB.  

El uso de GPU fue fundamental, ya que permitió reducir los tiempos de entrenamiento de varias horas a minutos por epoch, haciendo viable la experimentación con diferentes configuraciones de hiperparámetros.  

## Entrenamiento

### División del Conjunto de Datos

Para garantizar una evaluación justa del modelo, el dataset se dividió en tres subconjuntos:  

- **Entrenamiento (80%)**: utilizado para ajustar los parámetros de la red neuronal.  
- **Validación (20%)**: empleado para ajustar hyperparameters y monitorear el desempeño del modelo durante el entrenamiento.  
- **Pruebas (20% aprox)**: Como un archivo de pruebas externo.  

La partición se realizó de forma **estratificada** cuando fue posible, para asegurar una proporción similar de clases (fondo, pulmón y lesiones) en cada subconjunto.  

---

### Proceso de Entrenamiento

1. **Función de pérdida (criterion):**  
   Se empleó una combinación ponderada de:  
   - **Cross-Entropy Loss**: maneja la clasificación multiclase por píxel.  
   - **Dice Loss**: mejora la segmentación en clases minoritarias (lesiones pequeñas).  
   - **Focal Loss**: refuerza la atención en ejemplos difíciles, mitigando el desbalance de clases.   

2. **Optimizador y tasa de aprendizaje:**  
   - Optimizador: **Adam** (Adaptive Moment Estimation).  
   - Learning rate inicial: `1e-4`.  
   - Estrategia de ajuste: **ReduceLROnPlateau**, disminuyendo la tasa cuando la métrica de validación dejaba de mejorar.  

3. **Hiperparámetros de entrenamiento:**  
   - **Número de epochs:** 50 (con early stopping si no había mejora en validación después de 10 epochs).  
   - **Tamaño del minibatch (batch size):** 6 imágenes.  
   - **Weight decay:** `1e-5` como regularización para evitar overfitting.  

4. **Técnicas de regularización empleadas:**  
   - **Early stopping:** previno sobreajuste al detener el entrenamiento cuando la métrica de validación se estancaba.  
   - **Data augmentation:** actuó como regularizador al aumentar la diversidad del dataset.  
   - **WeightedRandomSampler:** aumentó la frecuencia de muestras con lesiones en cada batch, mejorando la sensibilidad en clases minoritarias.  

---

### Conclusiones sobre los Hiperparámetros

- El **batch size pequeño (6)** resultó adecuado dadas las limitaciones de memoria GPU y la resolución de las imágenes.  
- El **learning rate** tuvo un gran impacto: valores mayores (`1e-3`) causaban divergencia, mientras que valores más bajos (`1e-5`) ralentizaban demasiado el aprendizaje.  
- La combinación de pérdidas **CE + Dice + Focal** fue clave para lograr un equilibrio entre precisión global y sensibilidad a lesiones pequeñas.  
- El **ReduceLROnPlateau** permitió alcanzar una convergencia más estable, evitando caer en mínimos locales.  

---

### Consideraciones de Hardware

El entrenamiento se llevó a cabo en la infraestructura de **Kaggle Notebooks**, utilizando:  

- **GPU NVIDIA Tesla P100 (16 GB VRAM)**.  
- **CPU Intel Xeon** de soporte para operaciones de carga de datos.  
- **Memoria RAM disponible:** 13 GB.  

El uso de GPU fue fundamental, ya que permitió reducir los tiempos de entrenamiento de varias horas a minutos por epoch, haciendo viable la experimentación con diferentes configuraciones de hiperparámetros.  

