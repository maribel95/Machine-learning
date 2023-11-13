import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
# Podemos generar un reporte que nos informa del accuracy, precision, recall y f1-score
# gracias a la librería classification_report de sci-kit learn.
from sklearn.metrics import classification_report

# Cargamos el dataset.
df = pd.read_csv("data/data.csv", delimiter=r"\s+")

# Borramos la columna que contiene el id de las filas.
df = df.drop('0', axis=1)

# Vamos a separar catalán e inglés y asignar el target; luego los combinaremos.
df2 = df

# Este tendrá sólo catalán con target=0
df = df.drop('angles', axis=1)
df = df.assign(target=pd.Series([0] * len(df)))
df = df.rename(columns={'catala': 'Palabras'})
# Este tendrá sólo inglés con target=1
df2 = df2.drop('catala', axis=1)
df2 = df2.assign(target=pd.Series([1] * len(df2)))
df2 = df2.rename(columns={'angles': 'Palabras'})
df = pd.concat([df, df2])

def quitar_tildes(palabra):
    cambios = {'a': ['à', 'á'],
               'e': ['è', 'é'],
               'i': ['ì', 'í', 'ï'],
               'o': ['ò', 'ó'],
               'u': ['ù', 'ú', 'ü']
                }
    for clave, valores in cambios.items():
        for valor in valores:
            palabra = palabra.replace(valor, clave)

    return palabra


def contar_todas_las_vocales(palabra):
    palabra = quitar_tildes(palabra)

    vocales = "aeiou"
    frecuencia_vocales = [0, 0, 0, 0, 0]  # inicializamos el vector con ceros

    # recorremos cada caracter de la palabra
    for caracter in palabra:
        # si el caracter es una vocal, aumentamos el contador correspondiente
        if caracter in vocales:
            indice = vocales.index(caracter)
            frecuencia_vocales[indice] += 1

    return frecuencia_vocales


def lleva_tilde(palabra):

    tildes = "áéíóúÁÉÍÓÚàèìòùÀÈÌÒÙüÜ"
    for letra in palabra:
        if letra in tildes:
            return 1
    return 0

def contiene_combinacion(combinacion,palabra):
    if combinacion in palabra:
        return 1
    else:
        return 0



def contar_vocales(palabra):
    vocales = ['a', 'e', 'i', 'o', 'u']
    contador = 0
    for letra in palabra:
        if letra.lower() in vocales:
            contador += 1
    return contador




df['num_letras'] = df['Palabras'].apply(len)
df['num_vocales'] = df['Palabras'].apply(contar_vocales)
df['contiene_qu'] = df['Palabras'].apply(lambda x: contiene_combinacion(combinacion="qu", palabra=x))
df['contiene_th'] = df['Palabras'].apply(lambda x: contiene_combinacion(combinacion="th", palabra=x))
df['contiene_ll'] = df['Palabras'].apply(lambda x: contiene_combinacion(combinacion="ll", palabra=x))
df['tilde'] = df['Palabras'].apply(lleva_tilde)

# añadimos las columnas del vector al DataFrame
df[["a", "e", "i", "o", "u"]] = pd.DataFrame(df["Palabras"].apply(contar_todas_las_vocales).tolist(), index=df.index)

# La numeración de filas ahora se repite, por lo que vamos a resetear los índices.
df = df.reset_index(level=0, drop=False)


X = df.loc[:, ['num_letras', 'num_vocales','contiene_qu', 'contiene_th', 'contiene_ll', 'tilde', 'a', 'e', 'i', 'o', 'u']]
y = df['target']

# Ahora crearemos los conjuntos de entrenamiento y test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)


param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [1, 10, 100],
    'gamma': [0.1, 1, 10],
}

# Creamos el modelo de SVM y lo entrenamos. Vamos a probar primero con kernel lineal y luego el radial.
clf = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=None, verbose=0, scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.best_params_)
print("****************************************")
print(clf.best_score_)
# Nuestro mejor clasificador
biel_moya = clf.best_estimator_
# Hacer una predicción con el modelo entrenado.
y_pred = biel_moya.predict(X_test)
# Métricas de evaluación
report = classification_report(y_test, y_pred)
print(report)
