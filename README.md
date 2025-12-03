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

Resumen
----
Este repositorio contiene código y notebooks para trabajar con esqueletos 2D extraídos del dataset UCF101. Está orientado a experimentación: procesamiento de anotaciones, construcción de datasets y collates, un baseline LSTM y scripts reproducibles para entrenar e inferir.

Estructura (resumen)
----
- `src/` — código fuente (package). Busca `src/har/...` si seguiste la reorganización.
- `scripts/` — scripts CLI: `train_10cls.py`, `infer_10cls.py`.
- `data/raw/` — datos originales (no versionar archivos pesados aquí).
- `data/processed/` — pickles y datos preparados (` ucf101_2d_10cls.pkl`).
- `notebooks/` y `Notebooks/` — notebooks de experimentación.
- `artifacts/` — checkpoints, historiales y métricas generadas.
- `requirements.txt` — dependencias Python.

Quick start (PowerShell)
----
1. Crear y activar entorno:
```powershell
conda create -n har python=3.10 -y
conda activate har
```

2. Instalar PyTorch (elige la variante para CPU o GPU en https://pytorch.org/get-started/locally/). Ejemplo CPU:
```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

3. Instalar el resto de dependencias:
```powershell
pip install -r requirements.txt
```

4. (Opcional) Instalar el paquete en editable para imports tipo `from har...`:
```powershell
pip install -e .
```

Entrenar (ejemplo)
----
Entrena el modelo en la subset de 10 clases:
```powershell
& C:/Users/edosa/anaconda3/envs/nlp/python.exe scripts\train_10cls.py --pickle data\processed\ucf101_2d_10cls.pkl --epochs 20 --batch_size 8 --num_workers 2 --save_dir artifacts
```

Inferencia (ejemplo)
----
Genera predicciones y métricas sobre la partición de validación:
```powershell
& C:/Users/edosa/anaconda3/envs/nlp/python.exe scripts\infer_10cls.py --checkpoint artifacts\best_10cls.pt --pickle data\processed\ucf101_2d_10cls.pkl --out_dir inference_outputs --batch_size 32 --in_channels 4
```

Salida relevante
----
- `artifacts/` → checkpoints y `training_history_10cls.csv`.
- `inference_outputs/` → `predictions.csv`, `per_class_metrics_inference.csv`, `confusion_matrix.png`.


---
Actualizado el README con instrucciones de uso y ejemplos de comandos.