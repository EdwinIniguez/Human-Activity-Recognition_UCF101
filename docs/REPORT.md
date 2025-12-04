 REPORT — Human Activity Recognition (UCF101, 2D skeletons)
========================================

Última actualización: 2025-12-03
Repositorio: Human-Activity-Recognition_UCF101

Resumen ejecutivo
-----------------
Este informe resume el trabajo realizado para preparar y experimentar con un pipeline basado en esqueletos 2D extraídos de UCF101. El objetivo fue crear un flujo reproducible que vaya desde las anotaciones (pickles) hasta entrenamiento e inferencia de un baseline LSTM y proporcionar artefactos reproducibles (checkpoints, historiales, métricas).

Contenido del informe
- Contexto y archivos clave
- Preprocesamiento y transformaciones
- Dataset, collate y DataLoader
- Modelo (SkeletonLSTM)
- Configuración de entrenamiento usada (ejemplos)
- Resultados de una corrida de prueba (inferencia + 1 época de entrenamiento)
- Artefactos generados
- Reproducibilidad: comandos y recomendaciones
- Problemas conocidos y siguientes pasos

1. Contexto y archivos clave
---------------------------
Rutas importantes en el repositorio (relativas al root):
- `src/har/Models/lstm_model.py`  — implementación de `SkeletonLSTM`.
- `src/har/data/Dataset/ucf101_2d_10cls.pkl` — pickle con el subconjunto de 10 clases (procesado).
- `scripts/train_10cls.py` — CLI reproducible para entrenamiento.
- `scripts/infer_10cls.py` — CLI reproducible para inferencia sobre split de validación.
- `requirements.txt` — dependencias sugeridas.
- `artifacts/` — checkpoints y historiales (ej. `best_10cls.pt`).
- `docs/REPORT.md` — este archivo.

2. Preprocesamiento y transformaciones
--------------------------------------
- Lectura del pickle: las anotaciones se cargan desde `ucf101_2d_10cls.pkl` (estructura: lista de dict por muestra, con `keypoint`, `keypoint_score`, `label`, `frame_dir`, `img_shape`, etc.).
- Optimización: convertimos listas a `numpy.ndarray(dtype=float32)` al cargar (pre-conversión) para que el `Dataset.__getitem__` sea rápido.
- Selección de persona: si una muestra contiene >1 persona, se toma la persona con mayor score medio si `keypoint_score` está disponible.
- Normalización: opcional, `NormalizeKeypoints` normaliza coordenadas `x`/`y` por dimensiones `img_shape`.
- Velocidad (opcional): el dataset puede concatenar `vel` (primera derivada temporal) a la posición para producir `in_channels=4` (x,y + vel_x,vel_y). Esto es controlado por `--in_channels` en los scripts y por el dataset `include_velocity` flag.

3. Dataset, collate y DataLoader
--------------------------------
- `UCFSkeletonDataset` (definido en `scripts/*.py` y en `src/har/data`): devuelve muestras con `keypoint` como tensor `(T, V, C)` donde:
  - T: frames temporal
  - V: número de joints (17)
  - C: canales (2 o 4)
- Collate: `ucf_collate_fn_right` (right-pad) produce batches `(B, T_max, V, C)` y una máscara booleana `(B, T_max, V)` que puede usarse si se implementa.
- Recomendación: ajustar `num_workers` y `pin_memory` según el hardware. En Windows, `num_workers=0` es lo más estable; en Linux se puede aumentar a 2-4.

4. Modelo — SkeletonLSTM
------------------------
- Arquitectura principal (resumen):
  - Entrada: por frame `x` de dimensión `V * C` (joints * canales).
  - `fc_in`: MLP frame-wise que proyecta `V*C -> 256`.
  - `LSTM` bidireccional (hidden_dim=256 por defecto, 2 capas por defecto).
  - `fc_out`: MLP classifier que proyecta el output final del LSTM a `num_classes`.
- Clase: `SkeletonLSTM(num_joints=17, in_channels=2|4, hidden_dim=256, lstm_layers=2, num_classes=<n>)`.

5. Configuración de entrenamiento (ejemplos)
--------------------------------------------
- Guía de comandos (PowerShell):
  - Entrenamiento reproducible:
    ```powershell
    & C:/Users/edosa/anaconda3/envs/nlp/python.exe scripts\train_10cls.py --pickle src\har\data\Dataset\ucf101_2d_10cls.pkl --epochs 20 --batch_size 8 --val_batch 16 --num_workers 0 --save_dir artifacts
    ```
  - Inferencia:
    ```powershell
    & C:/Users/edosa/anaconda3/envs/nlp/python.exe scripts\infer_10cls.py --checkpoint artifacts\best_10cls.pt --pickle src\har\data\Dataset\ucf101_2d_10cls.pkl --out_dir inference_outputs --batch_size 32 --in_channels 4
    ```
- Hiperparámetros por defecto usados en pruebas:
  - Optimizer: Adam
  - LR: 1e-3
  - Weight decay: 1e-4
  - Dropout (fc_in/fc_out): 0.3
  - Loss: CrossEntropyLoss (con class weights computed por frecuencia)
  - AMP: opcional (`--use_amp`) — requiere CUDA funcional.

6. Resultados (corridas de prueba)
----------------------------------
Se realizaron pruebas rápidas para validar el pipeline:
- Inferencia con `artifacts/best_10cls.pt` (modelo entrenado con `in_channels=4`): generó `inference_outputs_test3/*` con métricas por clase. Algunas clases muestran recall/precision bajas — resultado esperado dado que es un baseline y los datos son desafiantes.
- Entrenamiento corto: 1 época sobre la misma subset usando CPU:
  - Comando: ver sección 5.
  - Resultado: guardó `artifacts_test_train/best_10cls.pt` y `training_history_10cls.csv`. Val acc ~0.205 en CPU durante test (1 época).

Notas sobre resultados
- Las métricas actuales son informativas para debugging; un entrenamiento más largo y/o con GPU y AMP mejorará tiempos y posiblemente desempeño.
- Para comparar modelos, recomendamos guardar checkpoints con metadata (in_channels, label_map, args).

7. Artefactos generados
-----------------------
- `artifacts/best_10cls.pt` — checkpoint guardado durante los entrenamientos principales.
- `artifacts_test_train/training_history_10cls.csv` — historial de la corrida de prueba.
- `inference_outputs_test3/predictions.csv`, `per_class_metrics_inference.csv`, `confusion_matrix.png` — salidas de la inferencia rápida.
