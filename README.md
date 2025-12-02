# Reconocimiento de actividades humanas con Deep Learning (UCF101 - esqueletos 2D)

Resumen
-------
Este repositorio contiene código y notebooks para entrenar modelos de reconocimiento de actividades humanas usando esqueletos 2D extraídos del dataset UCF101. El objetivo es experimentar con pipelines de preprocesado (normalización, padding), un `torch.utils.data.Dataset` cómodo para los esqueletos y un baseline con LSTM, además de mantener artefactos reproducibles (checkpoints, historial de entrenamiento y métricas por clase).

Estructura del repositorio
--------------------------
- `Dataset/` : metadatos y esqueletos 2D (por ejemplo `2d-skels/` y `data.md`).
- `dataExperiments.ipynb` : notebook principal donde se carga `ucf101_2d.pkl`, se construye `df_ann` (resumen de anotaciones), se define `UCFSkeletonDataset`, transforms, collate functions y se ejecutan experimentos de entrenamiento.
- `Notebooks/` : notebooks auxiliares (p. ej. `Human_Activity_Recognition.ipynb`).
- `Models/` : modelos (por ejemplo `Models/lstm_model.py` con la clase `SkeletonLSTM`).
- Salidas generadas tras entrenamiento (commonly saved at repo root):
	- `annotations_summary.csv` : resumen de anotaciones generado desde el pickle.
	- `lstm_minimal_checkpoint.pt`, `lstm_trained.pt` : checkpoints de modelo.
	- `training_history.csv` : historial de pérdidas y métricas por época.

Qué hay ya implementado
-----------------------
- Lectura del pickle `ucf101_2d.pkl` y creación de un `pd.DataFrame` resumen (`df_ann`) con columnas útiles como `frame_dir`, `total_frames`, `img_shape`, `label`, `keypoint_shape` y `has_keypoint_score`.
- `UCFSkeletonDataset` que devuelve keypoints como tensor `(T, V, C)`, etiqueta y meta (frame_dir, total_frames, score si aplica). Acepta un `transform` (p.ej. `NormalizeKeypoints`) para normalizar coordenadas.
- Transform composable (`Compose`, `NormalizeKeypoints`) para separación de preocupaciones entre datos y normalización.
- Funciones de collate/padding (`pad_sequence_kp`, `ucf_collate_fn`, variante para right-padding) que producen batches `(B, T, V, C)` y máscaras.
- Baseline `SkeletonLSTM` en `Models/lstm_model.py` y celdas de entrenamiento en `dataExperiments.ipynb` (ejecuciones mínimas y extendidas ya guardadas como checkpoints y `training_history.csv`).

Dependencias recomendadas
-------------------------
Estas son las librerías mínimas que el notebook usa. Ajusta versiones según tu entorno y GPU:

Python 3.8+

Ejemplo de `requirements.txt` (sugerido):

```
torch
torchvision
pandas
numpy
scikit-learn
matplotlib
jupyter
tqdm
```

Cómo preparar el entorno (PowerShell)
-----------------------------------
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Luego arrancar Jupyter
jupyter notebook
```

Uso básico
----------
- Abrir y ejecutar `dataExperiments.ipynb` (celdas ordenadas para: cargar datos → construir `df_ann` → crear `Dataset` y `DataLoader` → entrenar/evaluar modelo).
- `Notebooks/Human_Activity_Recognition.ipynb` contiene experimentos adicionales y visualizaciones.

Artefactos y registros
----------------------
- Tras ejecutar las celdas de entrenamiento se generan:
	- `*.pt` : checkpoints de PyTorch.
	- `training_history.csv` : pérdidas/precisión por época.
	- `annotations_summary.csv` : versión tabular del pickle para inspección rápida.

Recomendaciones prácticas
-------------------------
- Si vas a ejecutar múltiples experimentos, preprocesa y guarda muestras normalizadas en disco para acelerar iteraciones.
- Ajusta `batch_size` según tu GPU (p. ej. GTX 1650 → `batch_size` 8–16). Habilita AMP (mixed precision) para ahorrar memoria si el training loop lo soporta.
- Usa `num_workers=2..4` en `DataLoader` en Windows para no saturar I/O.

Próximos pasos sugeridos (ya planeados)
-------------------------------------
- Filtrar y remapear el conjunto a 10 clases seleccionadas y volver a entrenar (reduce tiempo y facilita tuning).
- Ejecutar comparativas con regularización (dropout, weight decay) y augmentaciones temporales (random crop/jitter/flip) en el subconjunto de 10 clases.
- Guardar métricas por clase (`per_class_metrics.csv`) y matriz de confusión para seleccionar clases donde la regularización aporte mayor mejora.

Contacto y seguimiento
----------------------
Si quieres, puedo:
- filtrar el dataset a las 10 clases que prefieras y remapear etiquetas (0..9) en `dataExperiments.ipynb`.
- lanzar un experimento controlado con `SkeletonLSTM` + dropout + weight decay y guardar resultados.

Licencia y notas
-----------------
Este repositorio es una base educativa para experimentación con UCF101 y esqueletos 2D; respeta las licencias del dataset original y de las librerías usadas.

---
Actualizado automáticamente para reflejar el pipeline y próximos pasos de experimentación.