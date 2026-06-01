# Ejecutar en Paperspace

Repositorio: **https://github.com/manuelquistial/modelo_atcnet**

## 1. Crear máquina

En [Paperspace Gradient](https://gradient.paperspace.com/):

- Template: **PyTorch / TensorFlow** (GPU, p. ej. A4000 o superior)
- Disco persistente recomendado para `data/` y resultados

## 2. Clonar el proyecto

```bash
git clone git@github.com:manuelquistial/modelo_atcnet.git
cd modelo_atcnet
```

Si usas HTTPS:

```bash
git clone https://github.com/manuelquistial/modelo_atcnet.git
cd modelo_atcnet
```

## 3. Instalar dependencias

```bash
chmod +x scripts/setup_paperspace.sh
./scripts/setup_paperspace.sh
source .venv/bin/activate
```

## 4. Descargar el dataset (.mat, como EEG-ATCNet)

```bash
source .venv/bin/activate
python scripts/download_bci2a.py --yes
```

Archivos en `data/raw/BCICIV_2a_mat/` (`A01T.mat` … `A09E.mat`).

Alternativa: copiar los `.mat` desde tu máquina:

```bash
scp data/raw/BCICIV_2a_mat/*.mat paperspace@<IP>:~/modelo_atcnet/data/raw/BCICIV_2a_mat/
```

## 5. Entrenar

```bash
source .venv/bin/activate
cd ~/modelo_atcnet   # o la ruta donde clonaste

python scripts/verify_shapes.py
python scripts/run_subject_dependent.py
python scripts/run_loso.py
# Opcional (largo):
# python scripts/run_ablation.py
# python scripts/run_baselines.py
```

Resultados en `data/results/`.

## 6. Ruta de datos alternativa (opcional)

Si guardas los MAT fuera del repo:

```bash
export MODEL_ATCNET_DATA_DIR=/storage/bci2a_mat
```

Esa carpeta debe contener directamente `A01T.mat`, etc.

## 7. Tiempos orientativos (GPU)

| Script | Nota |
|--------|------|
| `run_subject_dependent.py` | 9 sujetos × 10 runs, early stopping |
| `run_loso.py` | 9 folds × 10 runs |
| `run_ablation.py` | 45 entrenamientos |

Usa `tmux` o `nohup` para sesiones largas:

```bash
nohup python scripts/run_subject_dependent.py > logs_sd.txt 2>&1 &
tail -f logs_sd.txt
```

## Referencia

Implementación alineada con [Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet). Ver `docs/PAPER_AUDIT.md`.
