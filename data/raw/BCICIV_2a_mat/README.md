# BCI IV-2a (MAT format — EEG-ATCNet / MOABB)

Archivos esperados (planos en esta carpeta):

- `A01T.mat`, `A01E.mat`, …, `A09T.mat`, `A09E.mat`

También se acepta el layout del repo oficial: `s1/A01T.mat`, …, `s9/A09E.mat`.

## Descarga automática (recomendado)

```bash
pip install moabb
python scripts/download_bci2a.py --yes
```

Usa [MOABB `BNCI2014_001`](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html) (mismo dataset que BCI Competition IV-2a).

## Manual

Desde [BNCI Horizon](http://bnci-horizon-2020.eu/database/data-sets) (carpeta `001-2014`) o el paquete MAT de la competición.
