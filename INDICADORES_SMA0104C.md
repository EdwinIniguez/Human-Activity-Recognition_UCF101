# Mapeo de Indicadores SMA0104C — Análisis de información

Ámbito: Proyecto `Human-Activity-Recognition_UCF101` (LSTM sobre esqueletos 2D, 10 clases).  
Fuentes: `Notebooks/dataExperiments_10cls.ipynb`, `scripts/train_10cls.py`, `scripts/infer_10cls.py`, `docs/REPORT.md`, `artifacts/*.csv` y `artifacts/*.png`.

Solo se listan los indicadores con evidencia en estos archivos.

---

## Indicador: Procesa grandes volúmenes de datos de manera eficiente
**Evidencia:**
- `scripts/train_10cls.py`: carga pickle de secuencias y usa `DataLoader` con `pin_memory=True`, `num_workers` configurables y `collate_fn` para batches de longitud variable (eficiencia en I/O y CPU-GPU pipeline).
- `Notebooks/dataExperiments_10cls.ipynb` (celdas iniciales): preconversión a NumPy y right-padding para acelerar `__getitem__` (~10x vs. on-the-fly).
- `docs/REPORT.md` (sección de pipeline): describe batching y aceleración con AMP (Automatic Mixed Precision).

**Cómo verificar:**
- Ejecutar `python scripts/train_10cls.py --pickle src/har/data/Dataset/ucf101_2d_10cls.pkl --epochs 1 --batch_size 32 --num_workers 4` y observar tiempos por iteración.
- Revisar en el notebook la celda de tiempos de acceso tras la preconversión (más rápida que lectura directa).

---

## Indicador: Genera tableros útiles y correctos que apoyan la toma de decisiones
**Evidencia:**
- `artifacts/per_class_metrics_CE_10cls.csv`: métricas por clase (precision/recall/F1) para priorizar mejoras.
- `artifacts/confusion_matrix_10cls.png`: matriz de confusión para decidir en qué clases ajustar datos o pesos.
- `Notebooks/dataExperiments_10cls.ipynb` (celdas de visualización): gráficos de curvas de pérdida y accuracy.
- `docs/REPORT.md` (sección de resultados): tablas resumidas de métricas y hallazgos clave.

**Cómo verificar:**
- Abrir `artifacts/confusion_matrix_10cls.png` y `per_class_metrics_CE_10cls.csv`.
- En el notebook, ejecutar las celdas de visualización de métricas (curvas Train/Val, matriz de confusión).

---

## Indicador: Mide correctamente el desempeño del modelo y sus métricas son correctas
**Evidencia:**
- `scripts/train_10cls.py`: cálculo de `loss`, `accuracy`, métricas macro y per-class; guarda `training_history_10cls.csv`.
- `scripts/infer_10cls.py`: genera `predictions.csv`, `per_class_metrics.csv`, `confusion_matrix.png` con el checkpoint final.
- `docs/REPORT.md` (sección de evaluación): reporte de accuracy global 62.9%, macro F1 ~0.61, y detalle por clase.

**Cómo verificar:**
- Ejecutar `python scripts/infer_10cls.py --checkpoint artifacts/best_10cls.pt --pickle src/har/data/Dataset/ucf101_2d_10cls.pkl --out_dir outputs` y revisar las métricas generadas.
- Abrir `artifacts/training_history_10cls.csv` y verificar columnas `train_loss`, `val_loss`, `val_acc`.

---

## Indicador: Interpreta los resultados de las predicciones de los modelos y los interpreta en el contexto del problema de manera correcta
**Evidencia:**
- `docs/REPORT.md` (secciones de análisis):
  - Identifica clases débiles (ej. clase 5 con recall bajo) y propone ajustar pesos o recolectar más datos.
  - Analiza trade-offs precisión/recall y su impacto en la tarea de reconocimiento de actividades.
- `Notebooks/dataExperiments_10cls.ipynb` (celdas finales): conclusiones sobre qué clases confunden más y por qué.
- `artifacts/per_class_metrics_CE_10cls.csv`: soporte numérico para la interpretación.

**Cómo verificar:**
- Leer las conclusiones en `docs/REPORT.md` y contrastar con la matriz de confusión.
- Revisar en el notebook las celdas de interpretación y las visualizaciones por clase.
