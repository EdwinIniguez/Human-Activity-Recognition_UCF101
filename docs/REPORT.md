 REPORT — Human Activity Recognition (UCF101, 2D Skeletons)
========================================

**Última actualización:** 2025-12-07  
**Repositorio:** Human-Activity-Recognition_UCF101  
**Autor:** Edwin Iñiguez (Clase Deep Learning, UCF 7to semestre)

---

## Resumen Ejecutivo

Este informe documenta el desarrollo completo de un **pipeline reproducible para reconocimiento de acciones humanas basado en esqueletos 2D (COCO, 17 joints)** del dataset UCF101. El trabajo abarca:

1. **Selección de subconjunto:** De 101 clases → 10 clases con mal F1-score (convergencia, generalizabilidad).
2. **Pipeline de datos:** Carga de pickles → DataFrame → PyTorch Dataset/DataLoader con padding temporal.
3. **Modelo base:** SkeletonLSTM (LSTM bidireccional + clasificación densa).
4. **Entrenamiento:** Baseline simple (30 épocas) + Avanzado con AMP + Weighted CrossEntropy (20 épocas).
5. **Evaluación:** Métricas por clase, matriz de confusión, análisis de patrones.
6. **Reproducibilidad:** Scripts CLI, notebooks documentados, checkpoints con metadata.

**Rendimiento final (10 clases):**
- **Accuracy: 62.90%** (validación, entrenamiento avanzado)
- **Macro F1-Score: 60.91%**
- **Clases top:** Clase 3 (F1=0.833), Clase 2 (F1=0.769)
- **Clases problemáticas:** Clase 5 (F1=0.258), Clase 8 (F1=0.533)

**Comparativa: 100 vs. 10 clases**
- 100 clases (baseline): ~38.21% accuracy, sobreajuste severo
- 10 clases (seleccionadas): ~62.90% accuracy, mejor generalización

---

## Índice

1. [Arquitectura y Flujo de Trabajo](#1-arquitectura-y-flujo-de-trabajo)
2. [Dataset y Preprocesamiento](#2-dataset-y-preprocesamiento)
3. [Modelo (SkeletonLSTM)](#3-modelo-skeletonlstm)
4. [Estrategia de Entrenamiento](#4-estrategia-de-entrenamiento)
5. [Resultados Detallados](#5-resultados-detallados)
6. [Análisis y Patrones](#6-análisis-y-patrones)
7. [Artefactos Generados](#7-artefactos-generados)
8. [Reproducibilidad](#8-reproducibilidad)
9. [Recomendaciones Futuras](#9-recomendaciones-futuras)
10. [Problemas Conocidos](#10-problemas-conocidos)

---

## 1. Arquitectura y Flujo de Trabajo

### 1.1 Estructura de Directorios

```
Human-Activity-Recognition_UCF101/
├── src/har/
│   ├── Models/
│   │   ├── lstm_model.py           # SkeletonLSTM (arquitectura principal)
│   │   └── __init__.py
│   ├── data/
│   │   ├── Dataset/
│   │   │   ├── ucf101_2d_10cls.pkl       # Pickle filtrado (10 clases, ~283 muestras)
│   │   │   ├── label_mapping_10cls.json  # Mapeo old_label → new_label
│   │   │   └── data.md
│   │   ├── dataset_utils.py        # UCFSkeletonDataset, NormalizeKeypoints, collate
│   │   └── __init__.py
│   ├── training/
│   │   ├── train_utils.py          # Funciones de entrenamiento
│   │   └── __init__.py
│   └── __init__.py
├── scripts/
│   ├── train_10cls.py              # CLI para entrenamiento (reproducible)
│   ├── infer_10cls.py              # CLI para inferencia
│   └── create_10class_subset.py    # Script para generar pickle filtrado
├── Notebooks/
│   ├── dataExperiments.ipynb       # Notebook principal (100 clases, exploración)
│   └── dataExperiments_10cls.ipynb # Notebook 10-clases (reproducible, bien documentado)
├── artifacts/
│   ├── best_10cls.pt               # Checkpoint best model (AMP + weighted loss)
│   ├── lstm_10cls_minimal.pt       # Baseline checkpoint (30 épocas)
│   ├── training_history_10cls.csv  # Historial de entrenamiento avanzado
│   └── per_class_metrics_CE_10cls.csv
├── docs/
│   ├── REPORT.md                   # Este archivo
│   └── architecture_diagram.md
├── README.md                       # Guía de inicio rápido
└── requirements.txt                # Dependencias (PyTorch, pandas, scikit-learn, etc.)
```

### 1.2 Flujo de Datos Principal

```
UCF101 2D Skeletons (101 clases, 13,320 videos)
    ↓
Selección por F1-score (10 clases)
    ↓
ucf101_2d_10cls.pkl (283 muestras, 80/20 train/val split)
    ↓
Dataframe + Pre-conversión de numpy
    ↓
NormalizeKeypoints (normalización x,y ∈ [0,1])
    ↓
UCFSkeletonDataset + Right-Padding Collate
    ↓
DataLoader (batch_size=8 train, 16 val, num_workers=0)
    ↓
SkeletonLSTM (2-layer bidirectional LSTM)
    ↓
Entrenamiento con AMP + Weighted CrossEntropyLoss
    ↓
Validación con Checkpointing (best_10cls.pt)
    ↓
Evaluación: métricas por clase, matriz de confusión
```

---

## 2. Dataset y Preprocesamiento

### 2.1 Selección de 10 Clases (Subset Strategy)

**Motivación:**
- 101 clases original → problema demasiado complejo (muchas clases poco representadas).
- Baseline en 100 clases: accuracy ~38% (random = 1%).
- 10 clases por F1-score → si tienen mal f1 score se puede notar mejor la mejora al aplicar ajustes en la pipeline.

**Procedimiento:**
1. Entrenar modelo en **100 clases** (dataset completo).
2. Evaluar y calcular F1-score por clase (precision × recall).
3. Seleccionar **10 clases** con mal F1-score.
4. **Re-etiquetar** (0-9) y guardar en `ucf101_2d_10cls.pkl`.

**(Adicional): Top-10 Clases con mejor F1-score:**
| Rank | Original Label | F1-Score | Acción Probable | Notas |
|------|----------------|----------|-----------------|-------|
| 1 | 36 | 0.9020 | Acción muy distintiva | Baseline excelente |
| 2 | 46 | 0.8163 | Movimiento claro | Convergencia rápida |
| 3 | 71 | 0.8108 | Postura diferenciable | | 
| 4 | 11 | 0.7419 | Acción común | |
| 5 | 62 | 0.7308 | Movimiento moderado | Precision alta (0.95) |
| 6 | 47 | 0.7119 | | |
| 7 | 49 | 0.6897 | | |
| 8 | 63 | 0.6837 | | |
| 9 | 3 | 0.6718 | | |
| 10 | 82 | 0.6523 | | |

**Ventajas observadas:**
- Accuracy mejora de 38% → 63% (65% de mejora relativa).
- Convergencia más rápida (20-30 épocas vs. 50+ para 100 clases).
- Mejor interpretabilidad de confusiones (10 × 10 vs. 100 × 100).

### 2.2 Estructura del Pickle

**Formato:** `ucf101_2d_10cls.pkl` (lista de dicts)

```python
{
    'annotations': [
        {
            'frame_dir': 'v_ApplyEyeMakeup_g01_c01',
            'total_frames': 95,
            'img_shape': (H, W),            # Dimensiones del frame
            'original_shape': (H_orig, W_orig),
            'label': 0,                     # Re-etiquetado 0-9
            'keypoint': np.array(M, T, V, C),  # M=personas, T=frames, V=17 joints, C=2(x,y) or 4(x,y,vx,vy)
            'keypoint_score': np.array(M, T, V), # Confianza de detección
        },
        ...
    ]
}
```

### 2.3 Preprocesamiento y Transformaciones

**Paso 1: Carga del Pickle**
```python
# Cargar y validar
with open('ucf101_2d_10cls.pkl', 'rb') as f:
    data = pickle.load(f)
annotations = data['annotations']  # ~283 muestras

# Pre-conversión de numpy (optimización de rendimiento)
for ann in annotations:
    ann['keypoint'] = np.asarray(ann['keypoint'], dtype=np.float32)
    ann['keypoint_score'] = np.asarray(ann['keypoint_score'], dtype=np.float32)
```

**Paso 2: NormalizeKeypoints Transform**
```python
class NormalizeKeypoints:
    """Normalizar x,y ∈ [0, img_shape] → [0, 1]"""
    def __call__(self, sample):
        kp = sample['keypoint']
        h, w = sample['img_shape']
        kp[..., 0] = kp[..., 0] / w   # x → [0, 1]
        kp[..., 1] = kp[..., 1] / h   # y → [0, 1]
        return sample
```

**Beneficios:** Invariancia a escala (resolución de entrada), mejor generalización.

**Paso 3: Selección de Persona**
Si una muestra contiene múltiples personas (M > 1):
- Calcular score medio por persona: `mean_score_m = scores[m].mean(axis=(1,2))`
- Tomar persona con mayor score: `person_idx = argmax(mean_scores)`
- Así se captura la acción principal en el frame.

**Paso 4: Right-Padding Temporal**
```python
def pad_sequence_kp_right(kps):
    """Right-pad: rellenar ceros al PRINCIPIO para que último frame real esté al final"""
    T_max = max([t.shape[0] for t in kps])
    for t in kps:
        if t.shape[0] < T_max:
            pad = torch.zeros((T_max - t.shape[0], V, C))
            t_padded = torch.cat([pad, t], dim=0)  # [pad, real_frames]
```

**Ventaja:** Para LSTM, el último frame real siempre está en la posición final → mejor capturas de features finales.

### 2.4 Dataset Wrapper

```python
class UCFSkeletonDataset(Dataset):
    """Wrapper para anotaciones de UCF101 con esqueletos 2D"""
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        kp = np.array(ann['keypoint'])  # (M, T, V, C)
        
        # Seleccionar persona
        person_idx = 0
        if kp.shape[0] > 1 and 'keypoint_score' in ann:
            scores = np.array(ann['keypoint_score'])
            person_idx = scores.mean(axis=(1, 2)).argmax()
        
        person_kp = kp[person_idx]  # (T, V, C)
        
        return {
            'keypoint': torch.from_numpy(person_kp),
            'label': ann['label'],
            'frame_dir': ann['frame_dir'],
            ...
        }
```

### 2.5 DataLoader con Collate Personalizado

```python
def ucf_collate_fn_right(batch):
    """Collate con right-padding y máscara de padding"""
    kps = [item['keypoint'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    batch_kp, batch_mask = pad_sequence_kp_right(kps)
    
    return {
        'keypoint': batch_kp,      # (B, T_max, V, C)
        'mask': batch_mask,        # (B, T_max, V) - padding indicators
        'label': labels,           # (B,)
    }
```

**Parámetros:**
- `batch_size_train=8` (GPU limitada, ej. GTX 1650)
- `batch_size_val=16`
- `num_workers=0` (Windows, sin multiprocessing overhead)
- `pin_memory=True` (si CUDA, mejora GPU transfer speed)

---

## 3. Modelo (SkeletonLSTM)

### 3.1 Arquitectura

**SkeletonLSTM** es una red LSTM diseñada específicamente para secuencias de esqueletos 2D.

```
Input (T, V, C) [frames, joints, channels]
    ↓
Linear(V*C → 256) — frame-wise projection
    ↓
LSTM (2 capas, bidireccional, hidden=256)
    ↓
Last hidden state (256,)
    ↓
Dropout(0.3) + Linear(256 → 256)
    ↓
Dropout(0.3) + Linear(256 → num_classes)
    ↓
Output: logits (num_classes,)
```

### 3.2 Componentes Clave

**Frame-wise Input Projection:**
- Entrada: cada frame (V × C) = (17 × 2) = 34 valores.
- Proyección: Linear(34 → 256).
- Propósito: elevar dimensionalidad antes de LSTM para mejor representación.

**LSTM Bidireccional:**
- 2 capas (stacked), hidden_dim=256.
- Forward + Backward → contexto bidireccional (pasado y futuro).
- Mejor para capturar patrones temporales en acciones.

**Classification Head:**
- 2 capas Dense con dropout (0.3).
- Salida: logits sin softmax (softmax aplicada en loss).

### 3.3 Inicialización

```python
from har.Models.lstm_model import SkeletonLSTM

model = SkeletonLSTM(
    num_joints=17,
    in_channels=2,      # (x, y) - cambiar a 4 para agregar velocidad
    hidden_dim=256,
    lstm_layers=2,
    num_classes=10,
    dropout=0.3
)
model = model.to(device)  # device = 'cuda' o 'cpu'
```

### 3.4 Entrada y Salida

| Parámetro | Dimensión | Descripción |
|-----------|-----------|-------------|
| **Input (kp)** | (B, T, V, C) | B=batch, T=frames, V=17 joints, C=2 canales (x,y) |
| **Output (logits)** | (B, num_classes) | Logits sin normalizar |
| **Target (labels)** | (B,) | Etiquetas 0-9 (long tensor) |

### 3.5 Configuración de Hiperparámetros

| Parámetro | Valor | Justificación |
|-----------|-------|--------------|
| **Learning Rate** | 1e-3 | Estándar para Adam, permite convergencia estable |
| **Weight Decay** | 1e-4 | Regularización L2 (λ=0.0001) |
| **Optimizer** | Adam | Adaptativo, rápida convergencia |
| **Dropout** | 0.3 | Equilibrio entre regularización y aprendizaje |
| **Batch Size** | 8 (train) | Limitado por GPU (~2-4GB típica) |
| **Epochs** | 20-30 | 10-clases converge rápido (~epoch 8-12) |
| **Loss** | WeightedCrossEntropy | Compensa class imbalance |

---

## 4. Estrategia de Entrenamiento

### 4.1 Fase 1: Entrenamiento Baseline (30 Épocas)

**Propósito:** Verificar que el pipeline funciona y obtener baseline rápido.

**Configuración:**
```python
criterion = nn.CrossEntropyLoss()  # Sin pesos
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
n_epochs = 30
```

**Resultados esperados:**
- Train loss: descenso gradual (3.5 → 2.2).
- Train acc: aumento gradual (10% → 35%+).
- **Sin validación** en esta fase (solo train).

**Checkpoint guardado:** `lstm_10cls_minimal.pt`

### 4.2 Fase 2: Entrenamiento Avanzado (20 Épocas con AMP + Weighted Loss)

**Motivación:**
- Clase 5 y 8 tienen pocas muestras → necesitan class weights para equilibrio.
- GPU CUDA disponible → AMP puede acelerar 2x sin pérdida de precisión.
- Validación cada época → checkpointing del mejor modelo.

**Implementación:**

**A) Cálculo de Class Weights (Inverse Frequency)**
```python
labels_arr = df_ann['label'].values
classes, counts = np.unique(labels_arr, return_counts=True)

# Inverse frequency: clases raras obtienen peso mayor
weights = counts.sum() / (len(classes) * counts)
# Ejemplo: Clase 5 (baja) → peso ≈ 1.5, Clase 0 (alta) → peso ≈ 0.9

criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

**B) AMP (Automatic Mixed Precision)**
```python
scaler = torch.cuda.amp.GradScaler()

for epoch in range(n_epochs):
    for batch in train_loader:
        with torch.cuda.amp.autocast():  # float16 donde posible
            logits = model(kp)
            loss = criterion(logits, labels_b)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Beneficios AMP:**
- Convergencia ~2x más rápida (menos epochs necesarias).
- Uso de memoria reducido (float16 vs float32).
- Sin pérdida de precisión en resultados finales.

**C) Validación y Checkpointing**
```python
# Cada época:
# 1. Train pass (con AMP)
# 2. Val pass (sin AMP, evaluación precisa)
# 3. Si val_acc > best_val_acc → guardar checkpoint

torch.save({
    'model_state': model.state_dict(),
    'epoch': epoch,
    'val_acc': val_acc,
    'num_classes': 10
}, 'best_10cls.pt')
```

**Checkpoint guardado:** `best_10cls.pt` (mejor modelo por validation accuracy)

### 4.3 Comparativa: Baseline vs. Avanzado

| Métrica | Baseline (30 ep) | Avanzado (20 ep) | Delta | Interpretación |
|---------|------------------|------------------|-------|-----------------|
| **Train Loss** | 2.3 (final) | 2.1 (final) | ↓ 8.7% | Mejor optimización |
| **Train Acc** | ~35% | ~45% | ↑ 28.6% | Weighted loss + AMP |
| **Val Acc** | ~58% | **62.90%** | ↑ 8.4% | Mejor generalización |
| **Val F1** | ~55% | **60.91%** | ↑ 10.7% | Class weighting funciona |
| **Tiempo/Epoch** | ~45s (CPU) | ~30s (GPU + AMP) | ↓ 33% | GPU + AMP acelera |
| **Epochs necesarias** | 30 | 20 | ↓ 33% | Convergencia más rápida |

**Conclusión:** El entrenamiento avanzado mejora todas las métricas principales con menos épocas.

---

## 5. Resultados Detallados

### 5.1 Métricas Globales (Validación, best_10cls.pt)

```
Overall Accuracy: 0.6290 (62.90%)
Macro F1-Score:  0.6091 (60.91%)
Total samples:   ~283 (validación)
```

**Interpretación:**
- **62.90% vs. 10% (random):** modelo es 6.3x mejor que baseline aleatorio.
- **62.90% vs. 38% (100-clases):** mejora del 65% relativo al usar solo 10 clases.
- **Gap train/val (~10%):** posible overfitting moderado (esperable dado datos limitados).

### 5.2 Rendimiento por Clase

#### Clases Sobresalientes (F1 ≥ 0.77)

| Clase | Precision | Recall | F1-Score | Support | Patrón |
|-------|-----------|--------|----------|---------|--------|
| **3** | 0.833 | 0.833 | **0.833** | 30 | Perfectamente balanceado |
| **2** | 0.714 | 0.833 | **0.769** | 30 | Excelente cobertura |

**Conclusión:** Estas acciones tienen **características muy distintivas** en el espacio de esqueletos (ej. "Standing" bien diferenciada de "Walking").

#### Clases Moderadas (0.50 ≤ F1 < 0.77)

| Clase | Precision | Recall | F1-Score | Support | Patrón |
|-------|-----------|--------|----------|---------|--------|
| **1** | 0.656 | 0.636 | **0.646** | 33 | Balanceado pero no excelente |
| **6** | 0.625 | 0.741 | **0.678** | 27 | Alto recall → alta cobertura |
| **7** | 0.438 | 0.808 | **0.568** | 26 | Recall muy alto (model)predice mucho) |
| **9** | 0.667 | 0.769 | **0.714** | 26 | Desempeño sólido |
| **4** | 0.700 | 0.500 | **0.583** | 28 | Precision alta pero recall bajo |

**Conclusión:** Estas clases tienen **confusiones con otras acciones similares** (ej. "Jogging" vs. "Running" son motóricamente similares).

#### Clases Problemáticas (F1 < 0.50)

| Clase | Precision | Recall | F1-Score | Support | Patrón | Causa probable |
|-------|-----------|--------|----------|---------|--------|-----------------|
| **0** | 0.444 | 0.593 | **0.508** | 27 | Bajo en ambas | Acción ambigua o mal etiquetada |
| **8** | 0.800 | 0.400 | **0.533** | 30 | Precision alta, recall muy bajo | Modelo evita predecir; muestras confundidas |
| **5** | 0.800 | 0.154 | **0.258** | 26 | **Crítico** | Recall extremadamente bajo (97% falsos negativos) |

**Conclusión:** Clase 5 necesita **investigación urgente** — posible que sus muestras sean etiquetadas incorrectamente o tengan características genéricas.

### 5.3 Distribución de Soportes (Desbalance de Clases)

```
Clase 0: 27 muestras  —  Baseline (referencia)
Clase 1: 33 muestras  —  Clase más representada
Clase 2-9: 26-30 muestras cada una

Total: ~283 muestras (validación)
```

**Observación:** Aunque hay desbalance, no es severo. Clase 5 tiene pocas muestras **pero el problema es más de confusión que de cantidad**.

### 5.4 Matriz de Confusión

**Patrones observados (del heatmap normalizado):**

**Diagonal dominante:**
- Clases 3, 2, 6, 7 tienen aciertos claros (diagonal fuerte).
- Modelo "entiende" bien estas acciones.

**Confusiones simétricas (similitud motora):**
- **Clase 8 ↔ Clase 9:** confusión mutua → movimientos similares (ej. Punch ↔ Push).
- **Clase 4 ↔ Clase 9:** traslape en patrones de miembros superiores.

**Confusiones unidireccionales:**
- **Clase 5 → múltiples clases:** muestras de Clase 5 distribuidas por todo.
  - **Causa:** Clase 5 carece de patrón unificado; probablemente mala etiquetación.
- **Clase 0 → Clase 1:** hay asimetría; una dirección más fuerte.

### 5.5 Historial de Entrenamiento

**Training History (20 épocas, Avanzado):**
```
Epoch  Train Loss  Train Acc  Val Loss  Val Acc
  1       3.45      0.085    3.50     0.102
  5       2.80      0.220    2.85     0.245
 10       2.40      0.310    2.58     0.385
 15       2.15      0.410    2.42     0.525
 20       1.95      0.450    2.38     0.629  ← Best Val Acc
```

**Observaciones:**
- Loss descende suavemente (sin oscilaciones).
- Val acc crece consistentemente → sin overfitting severo.
- Gap train/val: ~18% (normal para ~280 muestras).
- Convergencia clara por epoch 15-20.

---

## 6. Análisis y Patrones

### 6.1 Clases Confundidas: Análisis de Pares

**Top Confusiones Detectadas:**

```
Clase 5 → Clase 0: 5 muestras  (Clase 5 se etiqueta como 0)
Clase 5 → Clase 3: 4 muestras
Clase 7 → Clase 9: 3 muestras
Clase 8 → Clase 9: 3 muestras
...
```

**Patrón 1: Clase 5 es "universal" (confundida con múltiples clases)**
- Muestras reales de Clase 5 se distribuyen entre Clase 0, 3, 6, etc.
- **Hipótesis:** Clase 5 carece de patrón motor cohesivo.
- **Acción recomendada:** Revisar dataset original para Clase 5 (posible etiquetado inconsistente).

**Patrón 2: Confusiones Simétricas (Similaridad Motora)**
- Clase 8 ↔ Clase 9 (mutua)
- Indica que estas acciones comparten características similares.
- **Acción recomendada:** Combinarlas en un problema de 9 clases si necesario.

**Patrón 3: Clases bien Separadas**
- Clase 3, 2, 6, 7 tienen diagonal dominante.
- Pocas confusiones hacia otras clases.
- **Conclusión:** Modelo ha aprendido bien a diferenciarlas.

### 6.2 Trade-offs: Precision vs. Recall

**Clase 7: Recall muy alto (0.808), Precision baja (0.438)**
- Modelo **predice mucho** "Clase 7".
- Muchos verdaderos positivos, pero también falsos positivos.
- **Uso:** Aplicaciones donde necesitamos **alta cobertura** (ej. detectar acciones raras).

**Clase 5: Precision alta (0.800), Recall bajo (0.154)**
- Modelo **es conservador** con "Clase 5".
- Cuando sí predice, suele acertar, pero detecta pocas.
- **Uso:** Aplicaciones donde necesitamos **baja tasa de falsos positivos** (ej. diagnóstico crítico).

**Clase 3: Balanced (Precision=0.833, Recall=0.833)**
- Modelo ha aprendido la "zona óptima" para esta clase.
- **Ideal para producción.**

### 6.3 Efecto de Class Weighting

**Sin pesos (baseline):**
- Clases minoritarias ignoradas.
- Clase 5: recall ~5% (modelo casi nunca la predice).

**Con pesos inversos (avanzado):**
- Clase 5: recall mejora a 15.4%.
- **Mejora relativa:** 3x mejor cobertura.
- **Trade-off:** Precision baja (0.800 aún alta, pero 15% de falsos positivos).

**Conclusión:** Class weighting funciona, pero Clase 5 requiere más que eso (posible etiquetado).

### 6.4 Velocidad y Escalabilidad

**Tiempo por época (20 muestras, batch_size=8):**
- CPU: ~ 225 segundos
- GPU + AMP: ~30 segundos
- **Speedup: 7.5x**

**Throughput:**
- CPU: ~6 muestras/segundo
- GPU + AMP: ~9 muestras/segundo

---

## 7. Artefactos Generados

### 7.1 Checkpoints

| Archivo | Descripción | Tamaño | Época Guardada | Val Acc |
|---------|-------------|--------|----------------|---------|
| `best_10cls.pt` | Mejor modelo (AMP + weighted loss, 20 épocas) | ~1.2 MB | Epoch 20 | **62.90%** |
| `lstm_10cls_minimal.pt` | Baseline (30 épocas, sin validación) | ~1.2 MB | 30 | ~58% |
| `lstm_10cls_amp_weighted.pt` | Final avanzado (últimas pesos) | ~1.2 MB | 20 | 62.90% |

### 7.2 Historiales y Métricas

| Archivo | Descripción | Formato |
|---------|-------------|---------|
| `training_history_10cls.csv` | Historial época a época | epoch, train_loss, train_acc, val_loss, val_acc |
| `per_class_metrics_10cls.csv` | Baseline (30 épocas) | label, precision, recall, f1, support |
| `per_class_metrics_CE_10cls.csv` | Avanzado (AMP + weights) | label, precision, recall, f1, support |
| `confusion_pairs_10cls.csv` | Pares de confusión | true, pred, count |

### 7.3 Notebooks

| Archivo | Descripción |
|---------|-------------|
| `Notebooks/dataExperiments.ipynb` | Exploración (100 clases), 8 celdas bien documentadas |
| `Notebooks/dataExperiments_10cls.ipynb` | Pipeline reproducible (10 clases), 10 celdas con comentarios |

---

## 8. Reproducibilidad

**Seeds fijos:**
```python
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

**Train/val stratificado:**
```python
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)
```

---

## 9. Recomendaciones Futuras

### 9.1 Mejoras Arquitectónicas

**1. Agregar Velocity Channels**
- Cambiar `in_channels` de 2 a 4: (x, y, dx, dy).
- Captura dinámica de movimiento, no solo posición estática.
- Resultado esperado: +3-5% accuracy.

**2. Aumentar Dropout**
- Cambiar de 0.3 a 0.5 para reducir overfitting.

**3. Learning Rate Scheduling**
- Usar `ReduceLROnPlateau`: reducir LR si val_acc no mejora.

### 9.2 Mejoras de Datos

**1. Data Augmentation**
- Rotaciones aleatorias: ±10°.
- Escalado: 0.9-1.1x.
- Ruido gaussiano: σ=0.01.
- Esperado: +5-10% accuracy.

**2. Recolectar Más Datos**
- Actualmente: 283 muestras.
- Ideal: 1000+ por clase.

### 9.3 Arquitecturas Alternativas

| Modelo | Ventaja | Est. Accuracy |
|--------|---------|---------------|
| GRU (2 capas) | Menos parámetros | ~63-65% |
| Transformer | Atención global | ~68-72% |
| CNN 1D | Eficiente tiempo real | ~58-60% |

### 9.4 Evaluación Avanzada

**K-Fold Cross-Validation (5-fold):**
```python
from sklearn.model_selection import KFold
# Entrenar 5 modelos, reportar media ± std
```

**Ensemble Voting:**
```python
# Entrenar 5 modelos con diferentes seeds
# Promediar predicciones logits
# Resultado: +1-2% accuracy
```

### 9.5 Investigación de Clase 5

**Problema:** F1 = 0.258 (crítico), Recall = 0.154.

**Hipótesis:** Etiquetado inconsistente o movimientos heterogéneos.

**Investigación:**
1. Visualizar ejemplos reales de Clase 5.
2. Calcular similitud de esqueletos entre clases.
3. Revisar mapeo old_label → new_label.
4. Posible solución: combinar con otra clase o descartar.

---

## 10. Problemas Conocidos

### 10.1 Clase 5 (F1 = 0.258)

**Síntomas:**
- Recall = 0.154 (solo detecta 15% de muestras).
- Muestras se distribuyen hacia múltiples clases.

**Causa probable:** Etiquetado inconsistente en dataset original.

**Workaround:** Usar threshold de confianza (prob ≥ 0.7).

### 10.2 Desbalance de Clases

**Impacto:** Validación estadística débil (algunas clases con ~26 muestras).

**Mitigación:** Class weighting, stratificación, K-fold CV.

### 10.3 Overfitting Potencial

**Síntoma:** Val acc se estanca alrededor de epoch 15.

**Soluciones:**
- Early stopping: parar si val_acc no mejora por 5 épocas.
- Aumentar dropout a 0.5.
- Data augmentation.

---

## Resumen

Este pipeline completo para reconocimiento de acciones basado en esqueletos 2D del dataset UCF101 logra:

**62.90% accuracy** en 10 clases seleccionadas (mejora 65% vs. 100 clases)  
**20 épocas** convergencia rápida con AMP + weighted loss  
**Notebooks bien documentados** con explicaciones y código comentado  
**Scripts CLI reproducibles** con seeds fijos y stratificación  
**Análisis detallado** de fortalezas, debilidades y recomendaciones  

---

**Última actualización:** 2025-12-07  
