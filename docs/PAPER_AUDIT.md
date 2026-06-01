# Alineación con EEG-ATCNet (oficial)

Este proyecto sigue el repositorio de los autores: https://github.com/Altaheri/EEG-ATCNet

## Implementado como el repo oficial

| Elemento | Fuente oficial |
|----------|----------------|
| Ventana EEG | `t1=1.5*fs`, `t2=6*fs` → 1125 muestras (`preprocess.load_BCI2a_data`) |
| `Conv_block_` | L2=0.009, max_norm=0.6, F1=16, D=2, Kc=64, pool 8/7 |
| `TCN_block_` | depth=2, k=4, filters=32, dilations 1 y 2, `he_uniform` |
| `mha_block` | LN ε=1e-6, heads=2, key_dim=8, dropout=0.5 |
| Clasificador | `fuse='average'`: 5×Dense(4) → Average → softmax |
| Datos | `StandardScaler` por canal (fit train) |
| Entrenamiento | `validation_data=test`, EarlyStopping `val_accuracy`, ReduceLROnPlateau |
| Repeticiones | `N_TRAIN_RUNS=10`, mejor seed por sujeto |
| LR | 0.001 (`main_TrainTest.py`) |
| LOSO | Ambas sesiones T+E; train 8 sujetos, test 1 |
| Subject-dependent | T train, E test |

Parámetros totales: ~115K (paper: 115.2K).

## Diferencia respecto al repo

- **Datos**: GDF + MNE (oficial usa `.mat` BNCI). Misma ventana temporal.
- **Formato entrada**: `(B,1,22,1125)` con permute interno equivalente a `Permute(3,2,1)` del oficial.

Configuración central: `src/config.py`.
