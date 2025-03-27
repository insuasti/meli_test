"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""
# Librerias
import pandas as pd
import numpy as np
import json

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = transform_dataset(clean_dataset(read_data(file_path="./data/input/")))
    N = -10000
    X_train = data[:N].drop(columns=['condition'])
    X_test = data[N:].drop(columns=['condition'])
    y_train = data[:N]["condition"]
    y_test = data[N:]["condition"]
    return X_train, y_train, X_test, y_test

def read_data(file_path):

    jsonObj = pd.read_json(path_or_buf=file_path + "MLA_100k_checked_v3.jsonlines", lines=True)

    return jsonObj

def clean_dataset(data_set):

    df_shipping = data_set['shipping'].apply(pd.Series)
    data_set = pd.concat([data_set, df_shipping], axis=1)

    data_final = data_set[["condition", "price", "local_pick_up", "free_shipping", "listing_type_id",
                          "initial_quantity", "sold_quantity", "available_quantity"]]

    # listing type id
    condiciones_type = [data_final['listing_type_id'].str.contains('bronze', case=False, na=False),
                        data_final['listing_type_id'].str.contains('free', case=False, na=False),
                        data_final['listing_type_id'].str.contains('silver', case=False, na=False),
                        data_final['listing_type_id'].str.contains('gold', case=False, na=False)]

    opciones_type = ["bronze", "free", "silver", "gold"]

    data_final["listing_type_id_final"] = np.select(condiciones_type, opciones_type, default="Otro")

    # Data modelo

    data_modelo = data_final[["condition", "price", "listing_type_id_final",
                              "initial_quantity", "sold_quantity", "available_quantity"]]

    # Limpieza outliers

    # Borrar caso donde la cantidad inicial es mayor a 9990 y el articulo es usado
    data_modelo = data_modelo[~((data_modelo['initial_quantity'] > 9990) & (data_modelo['condition'] == 0))]

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

    data_modelo_final = data_modelo[["condition", "price", "listing_type_id_final",
                                     "initial_quantity", "sold_quantity", "available_quantity", "sold_ratio",
                                     "available_ratio", "price_log", "sold_log"]]

    return data_modelo_final

def transform_dataset(data_set):
    # Definir columnas
    columnas_numericas = ["price", "initial_quantity", "sold_quantity",
                          "available_quantity", "sold_ratio", "available_ratio",
                          "price_log", "sold_log"]

    columnas_categoricas = ["listing_type_id_final"]
    # Separar los subconjuntos
    X_numericas = data_set[columnas_numericas]
    X_categoricas = data_set[columnas_categoricas]

    # Escalar variables numéricas
    scaler = StandardScaler()
    X_numericas_scaled = scaler.fit_transform(X_numericas)

    # Codificar variables categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_categoricas_encoded = encoder.fit_transform(X_categoricas)

    # Obtener los nombres de las columnas codificadas
    columnas_categoricas_encoded = encoder.get_feature_names_out(columnas_categoricas)

    # Crear DataFrames con los resultados
    df_numericas_scaled = pd.DataFrame(X_numericas_scaled, columns=columnas_numericas, index=data_set.index)
    df_categoricas_encoded = pd.DataFrame(X_categoricas_encoded, columns=columnas_categoricas_encoded,
                                          index=data_set.index)

    # Unir todos los datos transformados en un nuevo DataFrame
    X_transformado = pd.concat([df_numericas_scaled, df_categoricas_encoded], axis=1)

    data_set_transformado = pd.concat([X_transformado,data_set["condition"]],axis=1)

    return data_set_transformado






if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    # Diccionario con los modelos
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=950),
        "Random Forest": RandomForestClassifier(random_state=950),
        "Gradient Boosting": GradientBoostingClassifier(random_state=950),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=950),
        "LightGBM": LGBMClassifier(random_state=950)
    }

    # Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{nombre}: Accuracy = {acc:.4f}")

    # Definir el espacio de búsqueda de hiperparámetros
    param_distributions = {
        'num_leaves': randint(20, 150),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.01, 0.2),
        'n_estimators': randint(100, 1000),
        'subsample': uniform(0.6, 0.4),  # entre 0.6 y 1.0
        'colsample_bytree': uniform(0.6, 0.4),  # entre 0.6 y 1.0
        'reg_alpha': uniform(0, 1),  # L1 regularization
        'reg_lambda': uniform(0, 1)  # L2 regularization
    }

    # Crear el modelo base
    modelo_lgbm = LGBMClassifier(random_state=950, n_jobs=-1)

    # Configurar la búsqueda aleatoria
    busqueda = RandomizedSearchCV(
        estimator=modelo_lgbm,
        param_distributions=param_distributions,
        n_iter=30,  # número de combinaciones a probar (ajustable)
        scoring='accuracy',  # puedes usar f1, roc_auc si lo prefieres
        cv=5,  # 5-fold cross-validation
        verbose=1,
        random_state=950,
        n_jobs=-1  # usa todos los núcleos disponibles
    )

    # Ejecutar la búsqueda
    busqueda.fit(X_train, y_train)

    # Resultados
    print("Mejores hiperparámetros encontrados:")
    print(busqueda.best_params_)

    print(f"Mejor accuracy promedio en CV: {busqueda.best_score_:.4f}")




