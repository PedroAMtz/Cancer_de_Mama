## Detección de cáncer de mama mediante IA

El proposito de este proyecto es explorar diferentes herramientas de preprocesamiento digital de imágenes y técnicas de machine learning y deep learning para el desarrollo de un modelo capaz de clasificar imágenes de rayos X de mama, buscando principalmente la clasificación entre hallazgos benignos y malignos.

### 1. Base de datos utilizada

Los datos utilizados en este proyectos pertenecen a una competicion de kaggle: **RSNA Screening Mammography Breast Cancer Detection** (<https://www.kaggle.com/c/rsna-breast-cancer-detection>)

!['Imagen de la competencia'](/comptkaggle.png)

Esta base de datos consiste en más de 54,000 imágenes de rayos X tipo DICOM de alta resolución en conjunto con metadatos que proporcionan cierta información adicional respecto con características específicas de los pacientes.


### 2. Herramientas propuestas

Para el desarrollo del proyecto se propone utilizar las siguientes herramientas con la intención de gestionar el proyecto y cada una de las etapas de procesamiento, modelos, entrenamientos y resultados obtenidos.

* **DagsHub:** Plataforma abierta dedicada a la logística y control de versiones de los modelos, experimentos y datos que se utilicen.
* **GoogleClaboratory:** Entrono de desarrollo de Python. (acceso en línea)
* **JupyterNotebook:** Entorno de ejecución de Python para su ejecución local. (acceso local)
* **MLFlow:** Plataforma abierta para almacenamiento de métricas y artefactos correspondientes a los experimentos que se fueran a realizar.

### 3. Pipeline propuesta

!['Imagen de pipeline'](/Proposed_Pipeline.png.png)

### 4. Estrategia de experimentación

#### **DATOS**
* Utilizar imágenes a diferentes resoluciones (256x256, 512x512, 1024x1024)
* Implementar diferentes filtros que permitan mejorar el contraste de las imágenes 
* Realizar aumento de datos y lidiar con el imbalance de las clases

#### **MODELO**
* Utilizar arquitecturas de CNN´s
* Establecer métricas de desempeño
* Considerar funciones de pérdida personalizadas
