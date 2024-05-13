# Folder to store the training scripts and evaluations
## Metrics
## 1º Iteration (Split Test Train)
| Model   |         MAE          |         MSE          |        MAPE          |
|---------|----------------------|----------------------|----------------------|
|  KNN    |        |                      |                      |
|  DT     |   |                      |                      |
|  RF     | 3833538.6766666668   | 20399042021756.254   | 1125.4061468998893   |

## 2º Iteración (KFold - Cross Validator)
| Model   |         MAE          |         MSE          |        MAPE          |
|---------|----------------------|----------------------|----------------------|
|  KNN    | 8109623.410509619    | 108645008175168.02   | 211.9595032399446    |
|  DT     | 9866412.024105048    | 143204623332782.06   | 365.09101801074763   |
|  RF     | 9596764.33166747     | 130841362527146.53   | 336.70053511541505   |

## 3º Iteración (Split Time Series - Cross Validator)
| Model   |         MAE          |         MSE          |        MAPE          |
|---------|----------------------|----------------------|----------------------|
|  KNN    | 8109623.410509619    | 108645008175168.02   | 211.9595032399446    |
|  DT     | 9866412.024105048    | 143204623332782.06   | 365.09101801074763   |
|  RF     | 9596764.33166747     | 130841362527146.53   | 336.70053511541505   |