# Predicción de Crímenes Violentos por Población

## Descripción del Proyecto

Este proyecto tiene como objetivo predecir la tasa de crímenes violentos por población (`ViolentCrimesPerPop`) utilizando el dataset **Communities and Crime** del repositorio UCI Machine Learning Repository.

Durante el desarrollo se aplicaron técnicas de:

* Preprocesamiento de datos
* Manejo de valores faltantes
* Selección de características
* Reducción de dimensionalidad
* Modelos de regresión y optimización de hiperparámetros

El propósito fue identificar el modelo con mejor capacidad predictiva para el problema planteado.

---

# 1. Configuración del Entorno

## Instalación de dependencias

```bash
pip install ucimlrepo umap-learn scikit-learn numpy pandas matplotlib seaborn
```

## Librerías utilizadas

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `ucimlrepo`
* `umap-learn`

---

# 2. Carga y Unificación de Datos

```python
from ucimlrepo import fetch_ucirepo

communities = fetch_ucirepo(id=183)

X = communities.data.features
y = communities.data.targets

df = pd.concat([X, y], axis=1)
```

### Dimensiones iniciales

| Dataset                   | Dimensión   |
| ------------------------- | ----------- |
| Variables predictoras `X` | (1994, 127) |
| Variable objetivo `y`     | (1994, 1)   |
| Dataset completo `df`     | (1994, 128) |

---

# 3. Preprocesamiento de Datos

## 3.1 Eliminación de Variables Identificadoras

Se eliminaron las columnas:

* `state`
* `county`
* `community`
* `communityname`
* `fold`

Estas variables no aportaban valor predictivo.

Dimensión resultante:

```text
df_model → (1994, 123)
```

---

## 3.2 Conversión a Datos Numéricos

Todas las variables fueron convertidas a tipo `float64`, reemplazando valores inválidos por `NaN`.

---

## 3.3 Manejo de Valores Faltantes

### Eliminación de columnas con demasiados NaN

Se eliminaron 22 variables con más del 40% de datos faltantes.

Ejemplos:

* `PolicAveOTWorked`
* `LemasTotalReq`
* `PolicBudgPerPop`

Dimensión resultante:

```text
df_model → (1994, 101)
```

### Imputación simple

La variable `OtherPerCap` fue imputada usando la media.

### Imputación avanzada

Se utilizó `IterativeImputer` con un estimador:

```python
SVR(kernel='rbf')
```

El dataset imputado se almacenó en:

```text
df_model_imputed
```

---

## 3.4 Eliminación de Variables con Baja Variabilidad

No se encontraron variables con menos de 2 valores únicos.

---

## 3.5 Eliminación de Variables Altamente Correlacionadas

Se eliminaron variables con correlación absoluta mayor a `0.90`.

### Resultados

| Dataset            | Variables eliminadas | Dimensión final |
| ------------------ | -------------------- | --------------- |
| `df_model`         | 32                   | (1994, 69)      |
| `df_model_imputed` | 38                   | (1994, 85)      |

---

## 3.6 División Entrenamiento y Prueba

Se realizó una división:

* 80% entrenamiento
* 20% prueba

```python
random_state = 42
```

---

# 4. Selección de Características

El análisis se realizó sobre:

```text
X_final_imputed
```

---

## 4.1 Análisis de Relevancia

Se eliminaron variables con correlación absoluta menor a `0.05` respecto a la variable objetivo.

Variables eliminadas:

* `PctSameState85`
* `PctVacMore6Mos`
* `PctWorkMomYoungKids`
* `LemasSwornFT`
* `householdsize`
* `racePctAsian`
* `PctEmplManu`

Dimensión resultante:

```text
X_final_imputed_reduced → (1994, 77)
```

---

## 4.2 Análisis de Redundancia

No se encontraron variables altamente correlacionadas después de la reducción.

---

# 5. Reducción de Dimensionalidad

## 5.1 PCA

Se evaluó PCA conservando distintos porcentajes de varianza.

| Varianza explicada | Componentes |
| ------------------ | ----------- |
| 85%                | 18          |
| 90%                | 24          |
| 95%                | 35          |
| 99%                | 51          |

### Resultados destacados

* Los modelos `MLP` obtuvieron mejor desempeño usando PCA.
* El mejor resultado fue:

```text
Test_R2 = 0.6430
```

con:

```text
35 componentes (95% de varianza)
```

---

## 5.2 UMAP

Se aplicó UMAP reduciendo los datos a 2 dimensiones:

```python
n_neighbors = 15
```

### Resultados

| Modelo   | Test_R2 |
| -------- | ------- |
| SVR UMAP | 0.4222  |
| MLP UMAP | 0.4813  |

La reducción extrema a 2 dimensiones produjo una pérdida importante de información predictiva.

---

# 6. Modelos Evaluados

Se implementó una función de evaluación utilizando:

* `KFold` con 8 particiones
* `shuffle=True`
* `random_state=42`

## Métricas utilizadas

* MAE
* RMSE
* R²

---

## Modelos entrenados

### Regresión Lineal

* Con imputación
* Sin imputación

### Random Forest Regressor

Optimización mediante:

```python
RandomizedSearchCV
```

### Support Vector Regressor (SVR)

Configuraciones evaluadas:

* `linear`
* `poly`
* `rbf`

### KNN Regressor

Distintos valores de:

```python
n_neighbors
```

### MLP Regressor

Se probaron diferentes arquitecturas y funciones de activación:

* `relu`
* `tanh`
* `logistic`

---

# 7. Mejores Resultados

| Modelo                                       | Test_MAE | Test_RMSE | Test_R2  |
| -------------------------------------------- | -------- | --------- | -------- |
| SVR Imputed (Kernel: Poly, C=0.01, Deg=3)    | 0.087626 | 0.124734  | 0.675157 |
| SVR Imputed (Kernel: RBF, Gamma=0.01, C=100) | 0.087070 | 0.124791  | 0.674858 |
| MLP Imputed (32,16) relu                     | 0.086205 | 0.124981  | 0.673870 |

---

# 8. Conclusiones

* Los modelos entrenados con datos imputados obtuvieron mejores resultados generales.

* Los mejores modelos fueron:

  * `SVR Imputed (Kernel: Poly, C=0.01, Deg=3)`
  * `MLP Imputed (32,16) relu`

* Ambos alcanzaron valores de `R²` cercanos a `0.67`, mostrando una buena capacidad predictiva.

* PCA conservó mejor la información relevante que UMAP para este problema.

* La selección de características y el manejo de valores faltantes influyeron significativamente en el desempeño final.

---

# 9. Ejecución del Proyecto

1. Abrir el notebook en Google Colab o Jupyter.
2. Instalar las dependencias.
3. Ejecutar todas las celdas en orden secuencial.

El proyecto está estructurado para ejecutar automáticamente:

* Preprocesamiento
* Modelado
* Evaluación
* Comparación de resultados


Link del video
https://youtu.be/UTq2AwSToD4
