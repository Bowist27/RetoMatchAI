# Multiclass Lung CT Segmentation – UNet++

## Introducción

Este proyecto implementa un modelo de segmentación semántica multiclase en imágenes de tomografía computarizada (TAC) de pulmón, con el objetivo de identificar estructuras y lesiones asociadas a **COVID-19**.  

La tarea se desarrolla en el marco de la competencia Kaggle COVID-19 CT Segmentation, y busca construir un modelo capaz de distinguir entre:  

- Fondo  
- Tejido Pulmonar  
- Lesión 1 (opacidades)  
- Lesión 2 (consolidaciones u otras anomalías)  

El modelo seleccionado es U-Net++, entrenado con una combinación de pérdidas Cross-Entropy + Dice + Focal y técnicas de balanceo para mejorar la detección de clases minoritarias.

---

## Dataset

- Fuente: [Kaggle COVID-19 CT segmentation dataset](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans)  
- Muestras: ~100 imágenes TAC con sus máscaras de segmentación.  
- Formato: imágenes en escala de grises y máscaras multiclase.  
- Procesamiento previo:
  - Redimensionamiento a 384x384 píxeles.  
  - Normalización clipping de HU entre -1500, 500, estandarización por media y desviación.  
  - Aumento de datos con Albumentations (rotaciones, flips, crops, variación de contraste/brillo).  
  - Balanceo mediante WeightedRandomSampler para priorizar imágenes con lesiones.  

---

## Modelo

### Arquitectura
- Base: U-Net++ (Nested U-Net).  
- Encoder: backbone preentrenado en ImageNet (e.g. ResNet34).  
- Decoder: bloques de upsampling + convoluciones + BatchNorm + ReLU.  
- Conexiones densas: entre encoder y decoder para reducir la brecha semántica.  
- Capa de salida: convolución 1x1 → 4 canales (clases).  
- Activación final: Softmax por píxel.  

---

## Entrenamiento

- División de datos: 80% entrenamiento y 20% validación. 
- Función de pérdida: combinación ponderada de CE + Dice + Focal.  
- Optimizador: Adam lr=1e-4, weight_decay=1e-5.  
- Batch size: 6.  
- Epochs: hasta 60  
- Regularización: data augmentation y weight decay

### Hardware
- GPU: NVIDIA Tesla P100 (16 GB VRAM).  
- Entorno: Kaggle Notebooks (CPU Intel Xeon + 13 GB RAM).  

---

## Ejemplo Resultados

### Métricas de evaluación
- Dice Coefficient (por clase y promedio)    
- Accuracy global por píxel  

### Resultados cuantitativos
- Dice promedio inicial (epoch 1): 0.36  
- Mejor Dice promedio validación (epoch 41): 0.84  

Ejemplo de métricas:  

| Epoch | Val Loss | Lesión 1 | Lesión 2 | Dice Lesión Pulmón | Dice Fondo    | Dice Mean |
|-------|----------|------------|-------------|---------------|---------------|-----------|
| 1     | 0.789    | 0.0271     | 0.0338      | 0.3843        | 0.6807        | 0.3663    |
| 10    | 0.385    | 0.1864     | 0.3347      | 0.8844        | 0.9861        | 0.7351    |
| 41    | 0.188    | 0.5857     | 0.6134      | 0.9147        | 0.9932        | 0.8404    |

### Resultados visuales
- El modelo segmenta correctamente estructuras pulmonares y lesiones extensas.  
- Tiende a subsegmentar lesiones muy pequeñas (falsos negativos).  
- Puede sobrepredecir bordes pulmonares en presencia de ruido.

Ejemplo:

<img width="985" height="314" alt="image" src="https://github.com/user-attachments/assets/c6286d1c-9862-4c8c-a822-44c34928b91e" />
  

---
## Implicaciones Éticas

El desarrollo de modelos de segmentación médica en el contexto de COVID-19 no solo debe enfocarse en el rendimiento técnico, sino también en el cumplimiento de normas legales, regulatorias y principios éticos que regulan el manejo de datos médicos y su aplicación clínica.

### Cumplimiento de Normas y Principios Éticos

- **Confidencialidad y privacidad de datos médicos:**  
  El dataset utilizado proviene de la competencia de **Kaggle COVID-19 CT Segmentation**, el cual fue previamente **anonimizado** antes de su publicación. Esto asegura que no existan identificadores personales que comprometan la identidad de los pacientes.  
  - Se cumple con los principios de la normativa **HIPAA (Health Insurance Portability and Accountability Act, EE.UU.)**, que protege la información médica de los pacientes.  
  - También se alinea con el **GDPR (General Data Protection Regulation, Unión Europea)**, que regula el uso de datos personales en investigaciones y garantiza derechos como el consentimiento informado, la minimización de datos y el principio de anonimización.  

  El modelo está diseñado como herramienta de apoyo para los profesionales de la salud, no como sustituto del juicio clínico. Esto coincide con los principios éticos promovidos por la OMS (Organización Mundial de la Salud) para la aplicación de IA en salud: seguridad, supervisión humana y transparencia.  

---

### Normatividad Correspondiente

En el repositorio del proyecto se documenta la normatividad aplicable al reto y al socio formador:  

- **Política de Kaggle Datasets:** todos los datos provienen de repositorios públicos con licencias de uso académico e investigativo.  
- **Declaración de Helsinki:** principios éticos de investigación médica, aplicables incluso a datos anonimizados.  
- **HIPAA (EE.UU.):** cumplimiento en cuanto al manejo de datos médicos anonimizados.  
- **GDPR (UE):** cumplimiento en relación con el uso de datos sensibles y los principios de anonimización y transparencia.  

---

### Consideraciones Finales

- El modelo debe considerarse **un apoyo diagnóstico**, nunca un reemplazo de la decisión médica.  
- La documentación del repositorio deja explícito el cumplimiento de **GDPR** y **HIPAA**, garantizando transparencia y responsabilidad ética.  
