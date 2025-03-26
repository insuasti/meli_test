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
file_path = "./data/input/"

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

jsonObj = pd.read_json(path_or_buf=file_path + "MLA_100k_checked_v3.jsonlines", lines=True)

df_shipping = jsonObj['shipping'].apply(pd.Series)
jsonObj = pd.concat([jsonObj,df_shipping], axis=1)

data_final = jsonObj[["condition","price","local_pick_up","free_shipping","mode","listing_type_id",
                      "automatic_relist","status","initial_quantity","sold_quantity","available_quantity"]]

# listing type id
condiciones_type = [data_final['listing_type_id'].str.contains('bronze',case=False, na=False),
                    data_final['listing_type_id'].str.contains('free',case=False, na=False),
                    data_final['listing_type_id'].str.contains('silver',case=False, na=False),
                    data_final['listing_type_id'].str.contains('gold',case=False, na=False)]
opciones_type = ["bronze","free","silver","gold"]

data_final["listing_type_id_final"] = np.select(condiciones_type,opciones_type,default="Otro")

# status
data_final['status'].value_counts()
data_final['status_final'] = np.where((data_final['status'] == "active"),1,0)

# Data modelo

data_modelo = data_final[["condition","price","local_pick_up","free_shipping","mode","listing_type_id_final",
                      "automatic_relist","status_final","initial_quantity","sold_quantity","available_quantity"]]

# Target binaria
data_modelo['condition'] = data_modelo['condition'].map({'new': 1, 'used': 0})

# Limpieza outliers

# Borrar caso donde la cantidad inicial es mayor a 9990 y el articulo es usado

data_modelo = data_modelo[~((data_modelo['initial_quantity']>9990) & (data_modelo['condition']==0))]

data_modelo = data_modelo[data_modelo['price'] < 4000000]

grafica = plt.boxplot(data_modelo['price'])

descriptivas = data_modelo.describe()

# MODELO ----------------------------------------------------------------------------------
# Features y target
X = data_modelo.drop(columns='condition')
y = data_modelo['condition']

X['sold_ratio'] = X['sold_quantity'] / (X['initial_quantity'] + 1e-5)
X['available_ratio'] = X['available_quantity'] / (X['initial_quantity'] + 1e-5)
X['price_per_unit'] = X['price'] / (X['initial_quantity'] + 1e-5)

# Transformaciones logarítmicas
X['price_log'] = np.log1p(X['price'])
X['sold_log'] = np.log1p(X['sold_quantity'])

X = X.drop(columns=['price', 'sold_quantity', 'available_quantity'])

# Identificar columnas numericas y categoricas
numeric_features = ['price_log', 'sold_log', 'price_per_unit', 'sold_ratio', 'available_ratio', 'initial_quantity']
categorical_features = ['mode', 'listing_type_id_final']
boolean_features = ['local_pick_up', 'free_shipping', 'automatic_relist']
other_features = ['status_final']

# Preprocesador para variables numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocesador para variables categóricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combinar en un ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='passthrough')  # para incluir booleanas y otras numéricas sin procesar

# Definir modelos Base

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Validación cruzada y comparación

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f'{name}: Accuracy promedio = {scores.mean():.4f} ± {scores.std():.4f}')

# El mejor modelo fue XGBoost

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Definir el pipeline con preprocesador y XGBoost
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False,
                                 eval_metric='logloss',
                                 random_state=42,
                                 reg_alpha=1))
])

# Hiperparámetros para grid search
param_grid_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6, 8],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__gamma': [0, 1]
}

# Cross-validation estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=cv, scoring='accuracy',
                               verbose=2, n_jobs=-1)

grid_search_xgb.fit(X, y)

print(f"Mejor accuracy validación cruzada: {grid_search_xgb.best_score_:.4f}")
print("Mejores hiperparámetros:")
for param, val in grid_search_xgb.best_params_.items():
    print(f"{param}: {val}")

# Train-test split para evaluación final
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Entrenar mejor modelo
best_xgb_model = grid_search_xgb.best_estimator_
best_xgb_model.fit(X_train, y_train)

# Predicción y evaluación
y_pred = best_xgb_model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Importancia variables
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Extraer el modelo XGBoost desde el pipeline
xgb_model = best_xgb_model.named_steps['classifier']

# Nombres de las variables originales
numeric_features = ['price_log', 'sold_log', 'price_per_unit', 'sold_ratio', 'available_ratio', 'initial_quantity']
remainder_features = ['local_pick_up', 'free_shipping', 'automatic_relist', 'status_final']

# Extraer nombres de las columnas codificadas con OneHotEncoder
ohe = best_xgb_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_features = ohe.get_feature_names_out(['mode', 'listing_type_id_final'])

# Combinar todos los nombres en orden
all_features = np.concatenate([numeric_features, cat_features, remainder_features])

# Obtener importancia de variables del modelo
importances = xgb_model.feature_importances_

# Crear dataframe de importancia
importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Graficar
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.gca().invert_yaxis()
plt.title('Importancia de las Variables - XGBoost')
plt.xlabel('Importancia')
plt.tight_layout()
plt.show()


# otro modelo

from sklearn.ensemble import ExtraTreesClassifier

modelExtraTrees = ExtraTreesClassifier(random_state=123)
modelExtraTrees.fit(X_train, y_train)
y_train_hat = modelExtraTrees.predict(X_train)
y_test_hat = modelExtraTrees.predict(X_test)

print(modelExtraTrees)