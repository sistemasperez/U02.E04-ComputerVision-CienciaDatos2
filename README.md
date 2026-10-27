# INF-8239 · Unidad 02 · Proyecto de Visión por Computadora (CV)

> **Autor académico del diseño base:** Edwin Ramón José Nolasco  
> **Adaptación y experimentación:** Clasificación de Moda en Fashion-MNIST (CNN vs. Baseline Denso)  
> **Estudiante:** Aníbal Pérez  

---

## 1. Descripción del Proyecto y Dataset

Este proyecto implementa un flujo reproducible de Deep Learning orientado a la clasificación de imágenes de prendas y calzado en miniatura ($28 \times 28$ píxeles, escala de grises) sobre el conjunto de datos de referencia **Fashion-MNIST** (Zalando Research).

* **Dataset:** 70,000 imágenes divididas rigurosamente en:
  * **Entrenamiento:** 54,000 imágenes (77.1%).
  * **Validación:** 6,000 imágenes (8.6%) para monitoreo con *Early Stopping*.
  * **Prueba (Ciega):** 10,000 imágenes (14.3%), exactamente 1,000 por clase.
* **Ficha Técnica Completa:** Consulte [MODEL_CARD.md](MODEL_CARD.md) para el desglose exhaustivo de métricas por clase, análisis de hardware/costo, limitaciones y consideraciones de monitoreo en producción.

---

## 2. Configuración del Entorno y Ejecución Rápida

El proyecto utiliza [`uv`](https://docs.astral.sh/uv/) con Python 3.12 para garantizar un entorno reproducible y ligero.

### A. Ejecución en CPU (Entorno Estándar)
```bash
# 1. Sincronizar dependencias sin paquetes pesados de desarrollo
uv sync --no-dev --extra cpu

# 2. Verificar entorno y detección de aceleración por hardware
uv run python scripts/check_runtime.py

# 3. Entrenar ambos modelos y generar artefactos en reports/
uv run python scripts/train_cv.py --epochs 8
```

### B. Ejecución en GPU (Opcional · WSL2 / Linux con NVIDIA CUDA)
```bash
uv sync --extra gpu
uv run python scripts/train_cv.py --epochs 8
```

---

## 3. Resultados Experimentales y Comparativa de Modelos

Los modelos fueron evaluados sobre el conjunto de prueba (10,000 imágenes no vistas durante el entrenamiento):

| Modelo | F1-Macro | Accuracy Global | Parámetros Entrenables | Tiempo Train (CPU Intel i5) | Inferencia por Imagen | Rol Experimental |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`dense_baseline`** | **0.864** | 0.868 | 50,890 | **63.8 s** (~1 min) | **0.448 ms** | **Línea Base (MLP Plano):** Conecta directamente píxeles a neuronas densas con `Flatten()`. Rápido de entrenar pero ciego a la topología espacial. |
| **`small_cnn`** | **0.789** | 0.793 | **19,466** | 1,367.2 s (~22.8 min) | **0.565 ms** | **Modelo Convolucional:** Arquitectura ligera con filtros 2D, `MaxPooling2D`, `GlobalAveragePooling2D` y `Dropout(0.25)`. Ahorra un **61.8% de parámetros**. |

> **Nota metodológica:** En un presupuesto de 8 épocas con CPU, la CNN se encontraba en plena curva ascendente de aprendizaje (`val_accuracy` subiendo de 0.67 a 0.80) debido a la fuerte regularización de `Dropout`, mientras que la red densa convergía rápidamente.

---

## 4. Artefactos y Evidencia de Evaluación

La ejecución de `scripts/train_cv.py` genera automáticamente en la carpeta `reports/`:

1. **[reports/cv_metrics.json](reports/cv_metrics.json):** Registro numérico estructurado de F1-macro, parámetros, tiempo de cómputo y latencia.
2. **[reports/confusion_cnn.png](reports/confusion_cnn.png):** Matriz de confusión de las 10 clases, evidenciando excelente desempeño en calzado/pantalones ($F_1 > 90\%$) y fricción semántica en prendas superiores (camisas vs. camisetas).
3. **[reports/cnn_errors.png](reports/cnn_errors.png):** Cuadrícula visual con 16 imágenes reales de error mostrando la etiqueta real frente a la predicción del modelo.
4. **[models/best_cnn.keras](models/best_cnn.keras):** Pesos del mejor modelo convolucional serializados para inferencia.

---

## 5. Batería de Pruebas Unitarias

Para validar el contrato de datos y la arquitectura matemática del modelo:

```bash
uv run pytest -q
```
*(Valida la normalización $[0, 1]$, rechazo de dimensiones incorrectas y suma probabilística de Softmax = 1.0).*

---

## 6. Guía de Reproducibilidad Paso a Paso

Para reproducir íntegramente este laboratorio en cualquier máquina con Python 3.12:

```bash
# 1. Clonar el repositorio
git clone https://github.com/sistemasperez/U02.E04-ComputerVision-CienciaDatos2.git
cd U02.E04-ComputerVision-CienciaDatos2

# 2. Instalar Python 3.12 y sincronizar dependencias de CPU con uv
uv python install 3.12
uv sync --no-dev --extra cpu

# 3. Comprobar detección de entorno y aceleración de hardware
uv run python scripts/check_runtime.py

# 4. Ejecutar suite de pruebas unitarias (validación de contratos)
uv run pytest -q

# 5. Ejecutar pipeline completo de entrenamiento, evaluación y generación de artefactos
uv run python scripts/train_cv.py --epochs 8
```
Al finalizar la ejecución, los resultados numéricos y gráficos se actualizarán automáticamente en la carpeta `reports/` y el modelo serializado en `models/best_cnn.keras`.


