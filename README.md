# yolov8-aerial-analysis
# Detección de Autos en Tomas Aéreas con YOLO

Este repositorio contiene el código y análisis para la preparación, análisis y entrenamiento de un modelo de detección de objetos utilizando el formato YOLO.

## 📋 Descripción del Proyecto
El objetivo es procesar un dataset de imágenes aéreas de vehículos para entrenar un modelo de detección (YOLOv8/YOLOv3). El proyecto abarca:
1. Análisis exploratorio de los datos (EDA).
2. Interpretación y corrección de etiquetas (Bounding Boxes).
3. División del dataset (Train/Test).
4. Configuración para entrenamiento.

## 📂 Dataset
El dataset utilizado proviene de:
- [Aerial Cars Dataset (GitHub)](https://github.com/jekhor/aerial-cars-dataset)

## ⚙️ Requisitos
- Python 3.8+
- OpenCV
- Matplotlib
- Ultralytics (YOLO)

## 🚀 Estructura del Proyecto
├── data/                  # Dataset crudo (ignorado por git)
├── src/                   # Scripts de procesamiento
│   ├── analisis_datos.py  # Script de conteo y visualización
│   └── split_dataset.py   # Script para dividir train/val
├── notebooks/             # Jupyter Notebooks de entrenamiento
├── README.md              # Documentación
└── requirements.txt       # Dependencias
