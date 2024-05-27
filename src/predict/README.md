# Folder to store the training scripts and evaluations
## Metrics
## 1º Iteration (Split Test Train)
| Model   |         MAE          |         MSE          |        MAPE          |
|---------|----------------------|----------------------|----------------------|
|  KNN    | 10349185.888162345   | 165309373223268.34   | 243.65553499924394   |
|  DT     | 10233695.634005116   | 157329651266024.47   | 252.54049687652977   |
|  RF     | 10020948.75568875    | 143449752326045.62   | 240.65894100895693   |

## 2º Iteración (KFold - Cross Validator)
| Model   |         MAE          |         MSE          |        MAPE          |
|---------|----------------------|----------------------|----------------------|
|  KNN    | 8109623.410509619    | 108645008175168.02   | 211.9595032399446    |
|  DT     | 9866412.024105048    | 143204623332782.06   | 365.09101801074763   |
|  RF     | 9596764.33166747     | 130841362527146.53   | 336.70053511541505   |

## 3º Iteración (TsFresh - Cross Validator)
| Model   |         MAE          |         MSE          |        MAPE          |
|---------|----------------------|----------------------|----------------------|
|  KNN    | 9941704.64116674     | 147716184669670.53   | 354.56927066629555   |
|  DT     | 10176752.514536511   | 148505264418794.25   | 320.8084057136354    |
|  RF     | 10105131.536167238   | 142607994420256.03   | 339.58984158702856   |


## Optimized by Bayesian OPT using deafult dataframe:
DT -> Error absoluto medio (MAE): 2712615.5024054036
DT -> Error cuadrático medio (MSE): 52186491537514.98
DT -> Error porcentaje absoluto medio (MAPE): 56.92034277026964

KNN -> Error absoluto medio (MAE): 8109623.410509619
KNN -> Error cuadrático medio (MSE): 108645008175168.02
KNN -> Error porcentaje absoluto medio (MAPE): 211.9595032399446

RF -> Error absoluto medio (MAE): 7377770.541208173
RF -> Error cuadrático medio (MSE): 80740432852899.17
RF -> Error porcentaje absoluto medio (MAPE): 243.90736844303015

ADABOOST -> Error absoluto medio (MAE): 9421054.550152041
ADABOOST -> Error cuadrático medio (MSE): 112531138390984.9
ADABOOST -> Error porcentaje absoluto medio (MAPE): 295.904117980348