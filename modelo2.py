# Librerias
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
file_path = "./data/input/"

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

jsonObj = pd.read_json(path_or_buf=file_path + "MLA_100k_checked_v3.jsonlines", lines=True)

df_shipping = jsonObj['shipping'].apply(pd.Series)
jsonObj = pd.concat([jsonObj,df_shipping], axis=1)

data_final = jsonObj[["condition","price","local_pick_up","free_shipping","listing_type_id",
                     "initial_quantity","sold_quantity","available_quantity"]]

# listing type id
condiciones_type = [data_final['listing_type_id'].str.contains('bronze',case=False, na=False),
                    data_final['listing_type_id'].str.contains('free',case=False, na=False),
                    data_final['listing_type_id'].str.contains('silver',case=False, na=False),
                    data_final['listing_type_id'].str.contains('gold',case=False, na=False)]

opciones_type = ["bronze","free","silver","gold"]

data_final["listing_type_id_final"] = np.select(condiciones_type,opciones_type,default="Otro")

# Data modelo

data_modelo = data_final[["condition","price","listing_type_id_final",
                          "initial_quantity","sold_quantity","available_quantity"]]

# Limpieza outliers

# Borrar caso donde la cantidad inicial es mayor a 9990 y el articulo es usado
data_modelo = data_modelo[~((data_modelo['initial_quantity']>9990) & (data_modelo['condition']==0))]

data_modelo = data_modelo[data_modelo['price'] < 4000000]

# grafica = plt.boxplot(data_modelo['price'])

# Target binaria
data_modelo['condition'] = data_modelo['condition'].map({'new': 1, 'used': 0})

data_modelo['sold_ratio'] = data_modelo['sold_quantity'] / (data_modelo['initial_quantity'] + 1e-5)
data_modelo['available_ratio'] = data_modelo['available_quantity'] / (data_modelo['initial_quantity'] + 1e-5)
data_modelo['price_per_unit'] = data_modelo['price'] / (data_modelo['initial_quantity'] + 1e-5)

# Transformaciones logarítmicas
data_modelo['price_log'] = np.log1p(data_modelo['price'])
data_modelo['sold_log'] = np.log1p(data_modelo['sold_quantity'])

data_modelo_final = data_modelo[["condition","price","listing_type_id_final",
                                "initial_quantity","sold_quantity","available_quantity","sold_ratio",
                                 "available_ratio","price_log","sold_log"]]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Definir variables
FEATURES = ["price", "listing_type_id_final",
            "initial_quantity", "sold_quantity", "available_quantity",
            "sold_ratio", "available_ratio", "price_log", "sold_log"]
TARGET = "condition"

# Separar en features y target
X = data_modelo_final[FEATURES]
y = data_modelo_final[TARGET]

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definir transformaciones para columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('sold_log')  # Ya tenemos price_log
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Por si hay NaN
    ('scaler', StandardScaler())
])

categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Definir modelos a probar
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Evaluar cada modelo con validación cruzada
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Cross-validation con 5 folds
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    results[name] = cv_scores
    print(f"{name}: AUC Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")

# Entrenar el mejor modelo en todo el conjunto de entrenamiento
best_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42))  # Asumiendo que fue el mejor
])

best_model.fit(X_train, y_train)

# Evaluar en test
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nEvaluación en Test:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

# Obtener nombres de características después del preprocesamiento
# Para variables one-hot encoded
ohe_columns = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
    'onehot'].get_feature_names_out(categorical_features)
all_features = numeric_features + list(ohe_columns)

# Para modelos tree-based
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    importances = best_model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(11, 8))
    plt.title("Importancia de Variables")
    plt.barh(range(11), importances[indices][:11][::-1], align='center')
    plt.yticks(range(11), [all_features[i] for i in indices[:11]][::-1])
    plt.xlabel("Importancia Relativa")
    plt.tight_layout()
    plt.show()

from sklearn.model_selection import GridSearchCV

# Definir parámetros para LightGBM (ejemplo)
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7],
    'classifier__num_leaves': [31, 63, 127]
}

grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor AUC: {grid_search.best_score_:.4f}")


# Best model 2

# Obtener el mejor modelo encontrado por GridSearchCV
best_model2 = grid_search.best_estimator_

# Hacer predicciones en el conjunto de test
y_pred = best_model2.predict(X_test)  # Predicciones de clase (0 o 1)
y_proba = best_model2.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Mostrar resultados
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Matriz de confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte completo de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))