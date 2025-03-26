# Librerias
import numpy as np
import pandas as pd
file_path = "./data/input/"

jsonObj = pd.read_json(path_or_buf=file_path + "MLA_100k_checked_v3.jsonlines", lines=True)

prueba = jsonObj.seller_address[0]

jsonObj.warranty.value_counts()

warranty_var = jsonObj.warranty.value_counts()

# Expandir columnas que contienen diccionarios
df_seller_address = jsonObj['seller_address'].apply(pd.Series)
country = df_seller_address['country'].apply(pd.Series)
state = df_seller_address['state'].apply(pd.Series)
city = df_seller_address['city'].apply(pd.Series)


pais = country.name.value_counts()
estado = state.name.value_counts()
ciudad = city.name.value_counts()

# sub_status
jsonObj.sub_status.value_counts()
jsonObj.deal_ids.value_counts()
jsonObj.base_price.describe()

df_shipping = jsonObj['shipping'].apply(pd.Series)
df_shipping.local_pick_up.value_counts()
df_shipping.methods.value_counts()
df_shipping.tags.value_counts()
df_shipping.free_shipping.value_counts()
df_shipping['mode'].value_counts()
df_shipping.dimensions.value_counts().sum()
df_shipping.free_methods.value_counts().sum()
jsonObj = pd.concat([jsonObj,df_shipping], axis=1)

non_mercado_pago_payment_methods = jsonObj['non_mercado_pago_payment_methods'].apply(pd.Series)

# Expandir 'non_pago_payment_methods' en columnas
df_non_pago = jsonObj['non_mercado_pago_payment_methods'].apply(pd.Series).add_prefix('payment_')
methods_non_pago = df_non_pago['payment_0'].apply(pd.Series)
jsonObj = pd.concat([jsonObj, df_non_pago], axis=1)

methods_non_pago.description.value_counts()
jsonObj.seller_id.value_counts()

df_non_pago.payment_0[0]

variations = jsonObj['variations'].apply(pd.Series).add_prefix("variacion_")
variations['variacion_0'].isnull().value_counts()

jsonObj.site_id.value_counts()
jsonObj.listing_type_id.value_counts()
jsonObj.buying_mode.value_counts()
jsonObj.listing_source.value_counts()
jsonObj.international_delivery_mode.value_counts()
jsonObj.official_store_id.isnull().value_counts()
jsonObj.original_price.isnull().value_counts()
jsonObj.automatic_relist.value_counts()
jsonObj.site_id.value_counts()
jsonObj.listing_type_id.value_counts()

atributos = jsonObj['attributes'].apply(pd.Series).add_prefix("atributo_")
atributos['atributo_0'].isnull().value_counts()

jsonObj.buying_mode.value_counts()
jsonObj.tags.value_counts()
jsonObj.coverage_areas.value_counts()
jsonObj.status.value_counts()

# Data final
# data_final = jsonObj[["condition","base_price","price","local_pick_up","free_shipping","mode","listing_type_id",
#                       "automatic_relist","date_created","stop_time","status","initial_quantity","sold_quantity",
#                       "available_quantity"]]

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

data_modelo.info()

jsonObj.info()

validacion = jsonObj.groupby(['condition'])[['initial_quantity','sold_quantity','available_quantity']].median()
maximos = jsonObj.groupby(['condition'])[['initial_quantity','sold_quantity','available_quantity']].max()

validacion2 = jsonObj[(jsonObj['initial_quantity']>= 9990) & (jsonObj['condition'] == "used")]
validacion2 = jsonObj[(jsonObj['initial_quantity']>= 9990) & (jsonObj['condition'] == "new")]
validacion_precio = jsonObj[jsonObj['price'] >= 2000000000]
validacion_precio2 = jsonObj[jsonObj['price'] >= 11111000]
validacion_sold_quantity = jsonObj[jsonObj['sold_quantity'] >= 8600]
validacion3 = jsonObj.query("initia_quantity >= 9990 and condition == 'used'")
validacion_mayor_sold = jsonObj[jsonObj['sold_quantity'] > jsonObj['initial_quantity']]
validacion_mayor_sold.value_counts()

descriptivas = jsonObj.describe()
jsonObj.condition.value_counts()
