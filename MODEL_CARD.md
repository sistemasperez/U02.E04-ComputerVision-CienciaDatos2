# Model Card · Fashion-MNIST CNN

## 1. Modelo y versión
* **Nombre del modelo:** `small_cnn` (Red Neuronal Convolucional Compacta con GlobalAveragePooling y Dropout).
* **Modelo de referencia (Baseline):** `dense_baseline` (Perceptrón Multicapa / Red Densa de 64 neuronas con Flatten).
* **Versión:** 0.1.0 · Octubre 2026.
* **Autor / Estudiante:** Aníbal Pérez.
* **Diseño base:** Edwin Ramón José Nolasco (INF-8239 · Maestría en Inteligencia Artificial, UASD).
* **Framework:** TensorFlow 2.21.0 / Keras 3.15.1 (Python 3.12).
* **Artefacto serializado:** `models/best_cnn.keras` (guardado automáticamente mediante `ModelCheckpoint` por mejor `val_loss`).

---

## 2. Uso previsto
* **Propósito principal:** Clasificación automática de imágenes monocromáticas en miniatura ($28 \times 28$ píxeles) correspondientes a 10 categorías de indumentaria y calzado.
* **Dominio de aplicación:** Triaje y catalogación preliminar de artículos en plataformas de comercio electrónico minorista (*fast-fashion*) y benchmarking educativo para la evaluación de arquitecturas convolucionales frente a modelos densos.
* **Entorno operativo:** Dispositivos edge o servidores de bajo consumo con CPU x86_64, gracias a su reducida huella de parámetros (<20k parámetros).

---

## 3. Usos fuera de alcance
* **Imágenes fotográficas reales de alta resolución:** El modelo no procesa imágenes con fondos complejos, iluminación no homogénea o perspectivas tridimensionales variables.
* **Imágenes en color (RGB):** No está diseñado para procesar información cromática sin conversión previa a escala de grises.
* **Prendas compuestas o múltiples:** No realiza detección de objetos ni segmentación; asume estrictamente una única prenda aislada y centrada por imagen.
* **Decisiones críticas o industriales:** Prohibido su uso para certificación de uniformes médicos o equipos de protección individual (EPI/seguridad laboral).

---

## 4. Dataset y particiones
* **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research).
* **Volumen:** 70,000 imágenes en escala de grises ($28 \times 28$ píxeles), 10 clases perfectamente balanceadas (7,000 imágenes por clase).
* **Licencia:** MIT License (datos abiertos de Zalando SE).
* **Estrategia de partición (Prevención de fuga / Data Leakage):**
  * **Entrenamiento (`x_train`):** 54,000 imágenes (77.14%) para el aprendizaje de pesos convolucionales.
  * **Validación (`x_valid`):** 6,000 imágenes (8.57%) utilizadas exclusivamente por `EarlyStopping` (paciencia = 2) y `ModelCheckpoint`.
  * **Prueba ciega (`x_test`):** 10,000 imágenes (14.29%), 1,000 por clase, evaluadas estrictamente al final sin reutilización de información.

---

## 5. Preprocesamiento
Implementado en `src/inf8239_u02_cv/data.py`:
1. **Normalización por escalado lineal:** Conversión de enteros de 8 bits `uint8` $[0, 255]$ a flotantes `float32` en el intervalo $[0.0, 1.0]$ dividiendo por 255.0. Esto evita la saturación temprana de activaciones y estabiliza el descenso de gradiente (optimizador Adam).
2. **Expansión de dimensiones:** Inclusión del canal monocromático mediante `values[..., None]`, transformando la forma de $(N, 28, 28)$ a $(N, 28, 28, 1)$ para satisfacer el contrato de entrada de capas convolucionales 2D.
3. **Contrato de datos (`validate_images`):** Comprobación en tiempo de ejecución de dimensionalidad, rango $[0, 1]$, finitud numérica (`np.isfinite`) y concordancia dimensional entre tensores de imágenes y etiquetas.

---

## 6. Métricas globales y por clase

### A. Comparativa Global (Evaluación en Test Set · 10,000 imágenes)

| Modelo | F1-Macro | Accuracy Global | Parámetros Entrenables | Tiempo Train (CPU) | Inferencia por Imagen |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`dense_baseline`** (Red Densa) | **0.864** | 0.868 | 50,890 | **63.8 s** (~1 min) | **0.448 ms** |
| **`small_cnn`** (Red Convolucional) | **0.789** | 0.793 | **19,466** (-61.8%) | 1,367.2 s (~22.8 min) | **0.565 ms** |

### B. Rendimiento Detallado por Clase (`small_cnn`)

| Clase (#) | Nombre de la Prenda | Precisión | Recall | F1-Score | Soporte (Test) | Diagnóstico Cualitativo |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Trouser (Pantalón)** | **0.989** | **0.935** | **0.961** | 1,000 | **Excelente:** Silueta vertical alargada muy característica, casi nula confusión. |
| **8** | **Bag (Bolso)** | 0.924 | 0.934 | **0.929** | 1,000 | **Excelente:** Estructura cerrada compacta con asas detectables. |
| **5** | **Sandal (Sandalia)** | 0.950 | 0.885 | **0.916** | 1,000 | **Excelente:** Tiras delgadas y espacios vacíos distintivos. |
| **9** | **Ankle boot (Botín)** | 0.928 | 0.899 | **0.913** | 1,000 | **Excelente:** Perfil angular con caña alta bien discriminada de zapatillas. |
| **7** | **Sneaker (Zapatilla)** | 0.857 | 0.948 | **0.900** | 1,000 | **Sólido:** Confusión menor únicamente con botines o sandalias deportivas. |
| **3** | **Dress (Vestido)** | 0.751 | 0.851 | **0.798** | 1,000 | **Bueno:** Ocasionalmente confundido con abrigos largos o camisetas largas. |
| **0** | **T-shirt/top (Camiseta)** | 0.655 | 0.804 | **0.722** | 1,000 | **Moderado:** Confusión severa con camisas (Clase 6). |
| **2** | **Pullover (Pulóver)** | 0.694 | 0.664 | **0.679** | 1,000 | **Bajo:** Fricción continua con abrigos (Clase 4) y camisas (Clase 6). |
| **4** | **Coat (Abrigo)** | 0.654 | 0.653 | **0.653** | 1,000 | **Bajo:** La textura gruesa y solapas son difíciles de distinguir de pulóveres. |
| **6** | **Shirt (Camisa)** | **0.494** | **0.356** | **0.414** | 1,000 | **Crítico (Talón de Aquiles):** Menor a 50% de precisión. La resolución de $28 \times 28$ no permite resolver botones ni cuellos frente a camisetas. |

* **Evidencia Gráfica:** Consulte [reports/confusion_cnn.png](reports/confusion_cnn.png) para la matriz completa y [reports/cnn_errors.png](reports/cnn_errors.png) para la inspección de 16 imágenes reales de error.

---

## 7. Comparación de costo y eficiencia computacional
* **Hardware utilizado:** Intel Core i5 (arquitectura x86_64, ejecución en CPU mono/multi-hilo con aceleración Intel oneDNN).
* **Parámetros:** La CNN utiliza únicamente **19,466 parámetros** frente a los 50,890 del baseline denso (reducción del 61.8% en memoria y huella de almacenamiento).
* **Tiempo de entrenamiento:**
  * Red Densa: **63.8 segundos** (las multiplicaciones matriciales densas están altamente optimizadas en CPU).
  * Red Convolucional: **1,367.2 segundos (~22.8 minutos)**. La convolución requiere operaciones de ventana deslizante en tensores 4D que en CPU demandan un alto costo de cómputo en ausencia de aceleración masiva en paralelo (CUDA/GPU).
* **Tiempo de inferencia:** Ambas arquitecturas presentan latencias sub-milisegundo (**0.45 ms** para Densa, **0.57 ms** para CNN), lo que permite procesar hasta ~1,770 imágenes por segundo en producción sobre CPU estándar.

---

## 8. Limitaciones y riesgos
1. **Subconvergencia de la CNN en presupuesto de 8 épocas:**  
   La CNN incluye regularización estricta (`GlobalAveragePooling2D` y `Dropout(0.25)`). En 8 épocas, la CNN se encontraba en plena fase de aprendizaje ascendente (pasó de `val_accuracy` 0.67 en época 1 a 0.80 en época 8 sin sobreajuste), mientras que la red densa convergía rápidamente en pocas épocas. Con un régimen de 20-30 épocas o entrenamiento con GPU, la CNN tiende a sobrepasar a la red densa.
2. **Ambigüedad intra-categoría en prendas superiores (Topwear):**  
   Existe una frontera difusa entre *T-shirt/top*, *Pullover*, *Coat* y *Shirt*. En $28 \times 28$ píxeles sin color ni textura microscópica, las diferencias morfológicas son mínimas.
3. **Ausencia de contexto tridimensional:**  
   El modelo asume fondos planos; cualquier prenda sobrepuesta, arrugada o llevada puesta por una persona en ángulo oblicuo generará degradación en las predicciones.

---

## 9. Supervisión y monitoreo
1. **Umbral de decisión y abstención (*Rejection Option*):**  
   Implementar un umbral de corte de probabilidad en Softmax ($p \ge 0.65$). Si la probabilidad de la clase ganadora es inferior, la imagen debe derivarse a revisión humana o clasificarse bajo la etiqueta genérica *"Prenda Superior por Verificar"*, mitigando el riesgo de la Clase 6 (Camisas).
2. **Monitoreo de deriva de datos (*Data Drift*):**  
   Monitorear la distribución de intensidades lumínicas y la proporción de predicciones de la Clase 6 en producción para detectar cambios en el estilo de fotografía o compresión de imágenes entrantes.
3. **Métrica de latencia en servicio:**  
   Establecer una alerta si la latencia percentil 99 ($P_{99}$) supera los 2.0 ms por petición en la API de inferencia.

