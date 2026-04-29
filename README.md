# gsn-fed-learning

Projekt zaliczeniowy: **Federated Learning dla detekcji retinopatii cukrzycowej**  
Dataset: [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)  
Framework: [Flower](https://flower.ai) >= 1.17

## Notebooki

| Notebook | Opis | Status |
|----------|------|--------|
| `01_eda.ipynb` | Eksploracja datasetu | feature/eda |
| `02_baseline.ipynb` | Centralny model (upper bound) | TODO |
| `03_federated.ipynb` | Federated Learning z Flower | TODO |

## Strategia branchy

```
main
├── feature/eda          eksploracja datasetu
├── feature/model        EfficientNet-B0 + preprocessing
├── feature/fl-setup     konfiguracja Flower
└── feature/experiments  eksperymenty i wyniki
```

## Dataset
- 35 126 obrazów fundus (lewe + prawe oko)
- 5 klas: 0 = brak DR, 4 = proliferacyjna DR
- Imbalance: ~73% klasa 0
- Metryka: **Quadratic Weighted Kappa (QWK)**
