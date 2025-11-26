import torch
import torch.nn as nn
import torch.nn.functional as F

# Por que RNN?
# Las RNN son adecuadas para datos secuenciales. En este caso donde usamos datos
# de esqueletos 2D, las posiciones de los joints cambian a lo largo del tiempo,
# formando una secuencia que puede ser modelada eficazmente por una RNN.
# si trabajaramos con datos de video crudo, una CNN 3D o un modelo basado en Transformers.

# Por que LSTM?
# Para manejar el problema del desvanecimiento del gradiente, permitiendo
# capturar dependencias a largo plazo en secuencias. 
# (Esto lo vimos en clase esta en la presentacion).
# En el caso de las acciones con el dataset ucf101, hay que entender cómo 
# las posiciones de los joints evolucionan a lo largo de los frames.

class SkeletonLSTM(nn.Module):
    """
    Modelo baseline para reconocimiento de acciones basado únicamente
    en esqueletos 2D (keypoints) con una arquitectura MLP + Bi-LSTM + Classifier.

    Entrada esperada:
        keypoints: Tensor shape (batch, T, V, C)
            - T: número de frames (longitud temporal)
            - V: número de joints (17 en UCF101-2D)
            - C: coordenadas (2: x,y)

    Features opcionales:
        scores: (batch, T, V) → si se desean añadir como 3er canal.

    Output:
        logits: Tensor (batch, num_classes)
    """

    def __init__(
        self,
        num_joints=17,
        in_channels=2,             # 2: x,y ; si agregas score usa 3
        hidden_dim=256,
        lstm_layers=2,
        num_classes=101,
        dropout=0.3,
        bidirectional=True
    ):
        super().__init__()

        self.num_joints = num_joints
        self.in_channels = in_channels
        self.input_dim = num_joints * in_channels
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm_hidden_out = hidden_dim * (2 if bidirectional else 1)

        # --- MLP previo (por frame) ---
        self.fc_in = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- LSTM temporal ---
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # --- Clasificador ---
        self.fc_out = nn.Sequential(
            nn.Linear(self.lstm_hidden_out, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, keypoints, scores=None):
        """
        keypoints: (B, T, V, C)
        scores (opcional): (B, T, V)
        """

        B, T, V, C = keypoints.shape

        # Opcional: concatenar scores como tercer canal
        if scores is not None:
            # scores: (B, T, V) -> (B, T, V, 1)
            scores = scores.unsqueeze(-1)
            keypoints = torch.cat([keypoints, scores], dim=-1)  # ahora C = 3

        # Aplanar joints: (B, T, V*C)
        x = keypoints.reshape(B, T, -1)

        # --- MLP por frame ---
        x = self.fc_in(x)  # (B, T, 256)

        # --- LSTM ---
        x, _ = self.lstm(x)  # (B, T, hidden_dim*(1 o 2))

        # Usamos la salida del último frame (aunque se puede usar avg pooling)
        x_last = x[:, -1, :]  # (B, lstm_hidden_out)

        # --- Clasificador ---
        logits = self.fc_out(x_last)

        return logits


# Para pruebas rápidas
if __name__ == "__main__":
    model = SkeletonLSTM(num_classes=5)
    dummy_keypoints = torch.randn(2, 64, 17, 2)  # (batch=2, T=64, joints=17, 2 coords)
    out = model(dummy_keypoints)
    print("Output shape:", out.shape)  # -> (2, 5)
