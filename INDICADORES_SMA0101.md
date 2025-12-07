# Mapeo de Indicadores SMA0101 a Código
## Subcompetencia: Construcción de modelos

**Descripción:** Emplea métodos de aprendizaje máquina e inteligencia artificial en el procesamiento de información que habilitan la personalización de procesos, servicios o productos.

---

## Indicador 1: Identifica si es modelo estocástico o determinista

### Descripción
Identifica correctamente si el problema a tratar requiere un modelo estocástico o determinista.

### Archivos de Evidencia

```
 docs/REPORT.md
   └─ Sección 5: "Identifica correctamente si el problema..."
      • Naturaleza del problema: Clasificación de acciones (determinista en etiqueta)
      • Variabilidad observada: Múltiples ejecutores, ángulos, duraciones (estocástica)
      • Decisión: Modelo LSTM probabilístico (captura ambas)
      • Loss function: CrossEntropyLoss (modelo de probabilidad P(clase|entrada))
   
   └─ Análisis formal:
      Input: Secuencia de esqueletos 2D variable (T, 17, 2)
      └─ T es aleatorio (videos duraciones diferentes)
      └─ Keypoints tienen ruido de detección
      
      Proceso: Acciones humanas
      └─ Estocástico: Diferentes personas ejecutan de formas distintas
      └─ Variable: Velocidad, amplitud, ángulos varían
      
      Salida: Clase (0-9)
      └─ Determinista en label (etiqueta discreta)
      └─ Pero confianza es probabilística P(clase|entrada)
      
      Modelo elegido: SkeletonLSTM (LSTM + clasificación probabilística)

Notebooks/dataExperiments_10cls.ipynb
   └─ Celda 1 (Header): "Propiedades técnicas"
      • Selección de persona: modelo estocástico (score variable)
      • Pre-conversión: optimización pero preserva aleatoriedad
      • Loss probabilístico: CrossEntropyLoss con softmax
   
   └─ Celda 9 (Entrenamiento avanzado): 
      "AMP + weighted CrossEntropy"
      criterion = nn.CrossEntropyLoss(weight=class_weights)
      # Modelo de probabilidad: P(clase|entrada) con pesos

scripts/train_10cls.py
   └─ Línea 112: criterion = nn.CrossEntropyLoss(weight=...)
      # Log-loss sobre distribución de probabilidad
   
   └─ Línea 150-155 (inferencia): 
      with torch.no_grad():
          logits = model(kp)  # No normalizados
          probs = torch.softmax(logits, dim=1)  # Probabilidades [0,1]
          preds = probs.argmax(dim=1)  # Clase más probable
```

### Cómo Verificar
```python
# 1. Ver que loss es probabilístico
print("Loss function:", nn.CrossEntropyLoss())
# Output: CrossEntropyLoss() — aplica log-softmax internamente

# 2. Ver predicciones como probabilidades
logits = model(x)  # Shape: (B, 10)
probs = torch.softmax(logits, dim=1)  # Shape: (B, 10), suma=1
print(probs[0])
# Output: tensor([0.05, 0.08, ..., 0.92])  — probabilities que suman 1

# 3. Verificar confidence por muestra
confidence = probs.max(dim=1)[0]
print(confidence)
# Output: tensor([0.92, 0.45, 0.78, ...])  — confianza por predicción
```

---

## Indicador 2: Selecciona el modelo adecuado al problema

### Descripción
Selecciona el modelo adecuado al problema, justificando su elección.

### Archivos de Evidencia

```
 docs/REPORT.md
   └─ Sección 3: "Modelo (SkeletonLSTM)"
      • Arquitectura principal: LSTM bidireccional
      • Justificación: "Captura dependencias temporales en secuencias"
   
   └─ Sección 9.3: "Arquitecturas Alternativas"
      Comparativa completa:
      
      | Modelo | Ventaja | Desventaja | Est. Accuracy |
      |--------|---------|-----------|---------------|
      | CNN 1D | Eficiente | Contexto local | ~58-60% |
      | GRU | Menos params | Menos expresivo | ~63-65% |
      | LSTM Bidi | Estado arte | Más params | ~62.9% |
      | Transformer | Atención global | Requiere ≥10K | ~68-72% |
      
      Conclusión: LSTM óptimo para 280 muestras y secuencias de 50-100 frames

src/har/Models/lstm_model.py
   └─ Línea 20-25: Argumentación de arquitectura
      """
      SkeletonLSTM: LSTM bidireccional para secuencias temporales
      - Entrada frame-wise → proyección → LSTM → clasificación
      - Bidireccional: contexto pasado Y futuro
      - 2 capas: captura patrones de nivel bajo y alto
      """
      
   └─ Línea 25-30: Bidireccional específicamente
      self.lstm = nn.LSTM(..., bidirectional=True, ...)
      # Necesario para distinguir acciones como "empujar" vs "tirar"

Notebooks/dataExperiments_10cls.ipynb
   └─ Celda 1: Justificación en comentarios
      "Ventajas del enfoque 10-clases:
      - Convergencia más rápida: 100+ → 10 reduce complejidad
      - Mayor generalización: menos overfitting
      - Mejor interpretabilidad: confusiones entre 10 vs 100"

Decisión: Por qué no Transformer
   • Datos: 280 muestras (Transformer requiere ≥5-10K)
   • Compute: GPU limitada, LSTM es suficiente
   • Empiría: LSTM state-of-art para HAR en MCG et al., Li et al.
```

### Cómo Verificar
```python
# 1. Ver arquitectura seleccionada
from src.har.Models.lstm_model import SkeletonLSTM
model = SkeletonLSTM(num_joints=17, in_channels=2, num_classes=10)
print(model)
# Output: Verás LSTM bidireccional con 2 capas

# 2. Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: ~100K (razonable para 280 muestras, no underfitting)

# 3. Ver que es bidireccional
print(model.lstm.bidirectional)  # True
print(model.lstm.num_layers)    # 2
```

---

## Indicador 3: Explica ventajas y desventajas

### Descripción
Explica claramente las ventajas y desventajas del modelo seleccionado.

### Archivos de Evidencia

```
docs/REPORT.md
   └─ Sección 3: "Ventajas de SkeletonLSTM"
      ✓ Captura dependencias temporales
      ✓ Bajo overfitting (dropout 0.3 + L2)
      ✓ Convergencia rápida (20 épocas)
      ✓ Interpretable (matriz confusión)
      ✓ Portable (1.2 MB checkpoint)
      ✓ Flexible (in_channels=2|4)
   
   └─ Sección 3: "Desventajas de SkeletonLSTM"
      ✗ Bajo recall en clase 5 (F1=0.258)
      ✗ No captura correlaciones espaciales entre joints
      ✗ Requiere padding temporal (introduce ceros)
      ✗ Sensible a longitud de secuencia
      ✗ Parámetros fijos para 17 joints
      ✗ AMP solo con CUDA
   
   └─ Sección 6.2: "Trade-offs Precision vs. Recall"
      Clase 7: Recall 0.808, Precision 0.438
      └─ Mayor cobertura pero más falsos positivos
      
      Clase 5: Recall 0.154, Precision 0.800
      └─ Pocos falsos positivos pero falsos negativos altos
      
      Clase 3: Recall 0.833, Precision 0.833
      └─ Balance óptimo (ideal para producción)

Notebooks/dataExperiments_10cls.ipynb
   └─ Celda 1 (Header): "Notas técnicas"
      • Right-padding: ventaja para LSTM (último frame al final)
      • Pre-conversión numpy: ventaja RAM pero overhead inicial
      • Class weighting: ventaja para clases minoritarias
   
   └─ Celda 10 (Conclusiones): Síntesis de trade-offs
      "Trade-off: más RAM consumida (~100-200 MB),
       pero Dataset.__getitem__ ~10x más rápido"

scripts/train_10cls.py
   └─ Comentarios en línea (ejemplos):
      # Línea 110-120: "Class weighting: inverse frequency"
      # Ventaja: Compensa class imbalance
      # Desventaja: Requiere saber distribución
      
      # Línea 140-145: "AMP: ~2x speedup"
      # Ventaja: Aceleración sin pérdida de precisión
      # Desventaja: Solo funciona con CUDA
```

### Cómo Verificar
```python
# 1. Ver métricas por clase (ventajas/desventajas en datos)
import pandas as pd
metrics = pd.read_csv('artifacts/per_class_metrics_CE_10cls.csv')
print(metrics[['label', 'precision', 'recall', 'f1']])
# Verás: Clase 3 (f1=0.833) es buena, Clase 5 (f1=0.258) es mala

# 2. Analizar trade-off
weak = metrics[metrics['f1'] < 0.5]
print(weak[['label', 'precision', 'recall']])
# Verás: Clase 5 tiene precision 0.8 pero recall 0.154 (trade-off)

# 3. Leer sección de ventajas/desventajas
with open('docs/REPORT.md') as f:
    content = f.read()
    # Buscar sección "Ventajas de SkeletonLSTM" y "Desventajas"
    print(content[content.find('Ventajas'):content.find('Conclusión')])
```

---

## Resumen de Archivos Críticos

| Indicador | Archivo Primario | Archivo Secundario | Líneas/Celdas |
|-----------|------------------|-------------------|--------------|
| **1** | `docs/REPORT.md` (Sección 5) | `Notebooks/dataExperiments_10cls.ipynb` (Celda 1) | Sección 5 / Celda 1 |
| **2** | `docs/REPORT.md` (Sección 9.3) | `src/har/Models/lstm_model.py` | Sección 9.3 / Líneas 1-15 |
| **3** | `docs/REPORT.md` (Secciones 3 y 9) | `Notebooks/dataExperiments_10cls.ipynb` (Celda 10) | Secciones 3, 9 / Celda 10 |
---

**Última actualización:** 2025-12-07  
**Versión:** 1.0