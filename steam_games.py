import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
__errors__ = []

# Almacenar resultados de hiperparametros
r_squared_results = []
errors_results = []
alpha_values = [0.0001, 0.001, 0.01] 
epoch_limits = [500, 2000, 4000] 

def one_hot_encoding(df):
    '''Funcion para separar en diferentes atributos una variable categórica'''
    dummies = pd.get_dummies(df['primary_genre'], dtype=float)
    df = pd.concat([df, dummies], axis=1)
    df.drop('primary_genre', axis=1, inplace=True)
    return df

def scaling(X):
    """Estandarización (escalado) de columnas a partir del índice 8 hacia adelante"""
    means = X.mean()
    stds = X.std()
    new_X = (X - means) / stds
    return new_X

def split(df):
    """Función para separar el dataset en features y label y en test y train"""
    # Se mueve el df de forma aleatoria para eliminar ordenamientos y brindar aleatoriedad
    np.random.seed(42)
    df = df.sample(frac=1)

    # Se definen los atributos y el label
    X = df.drop('rating', axis=1)
    y = df['rating']

    
    # Se separa en train y test, 80% y 20% respectivamente
    split_index = int(0.8 * len(df))
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

def graph(df):
    """Funcion para graficar los atributos contra rating"""
    num_columns = 9
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i, column in enumerate(df.select_dtypes(include=['number']).columns):
        if column != 'rating':
            axes[i].scatter(df[column], df['rating'])
            axes[i].set_title(f'{column} vs rating')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('rating')


    plt.tight_layout()
    plt.show()
    
def clean_data(df):
    """Funcion para evaluar, eliminar y transformar datos iniciales del df"""
    
    # Eliminar columnas 
    df.drop(['Unnamed: 0', "game", "link", "release", "store_asset_mod_time", "detected_technologies", "all_time_peak_date", "store_genres", "publisher", "developer"], axis=1, inplace=True)
    
    df['players_right_now'] = df['players_right_now'].str.replace(',', '').astype(float)
    df['24_hour_peak'] = df['24_hour_peak'].str.replace(',', '').astype(float)

    #df = df.fillna(df["review_percentage"].mean())
    df.drop(df[df['review_percentage'].isna()].index, inplace=True)

    df.drop(df[df['players_right_now'].isna()].index, inplace=True)
    df.drop(df[df['24_hour_peak'].isna()].index, inplace=True)
    #print(df.isna().sum())

    #graph(df)
    
    df.drop(df[(df['peak_players'] <= 0) | (df['peak_players'] > 1000000)].index, inplace=True)
    df.drop(df[df['negative_reviews'] > 200000].index, inplace=True)
    df.drop(df[df['players_right_now'] > 100000].index, inplace=True)
    df.drop(df[df['all_time_peak'] > 500000].index, inplace=True)
    df.drop(df[df['positive_reviews'] > 1000000].index, inplace=True)
    df.drop(df[df['total_reviews'] > 1000000].index, inplace=True)
    df.drop(df[df['24_hour_peak'] > 200000].index, inplace=True)
    df = one_hot_encoding(df)
    return df

def plot_results():
    """Función para graficar la variación de los resultados cambiando los hiperparámetros"""
    alphas = np.array([x[0] for x in r_squared_results])
    epochs = np.array([x[1] for x in r_squared_results])
    r_squared = np.array([x[2] for x in r_squared_results])
    errors = np.array([x[2] for x in errors_results])

    # Gráfica del R^2
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for epoch in np.unique(epochs):
        idx = epochs == epoch
        plt.plot(alphas[idx], r_squared[idx], label=f'Epochs: {epoch}')
    plt.xlabel('Alpha')
    plt.ylabel('R^2')
    plt.title('R^2 vs Alpha and Epochs')
    plt.legend()

    # Gráfica del Error
    plt.subplot(1, 2, 2)
    for epoch in np.unique(epochs):
        idx = epochs == epoch
        plt.plot(alphas[idx], errors[idx], label=f'Epochs: {epoch}')
    plt.xlabel('Alpha')
    plt.ylabel('Error')
    plt.title('Error vs Alpha and Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_hyper(X_train, y_train, X_test, y_test):
    """Funcion para probar diferentes valores de alfa y epochs"""
    for alfa in alpha_values:
        for epochs_lim in epoch_limits:
            global __errors__
            __errors__ = []  
            
            # Inicializar parámetros
            params = np.zeros(X_train.shape[1])
            bias = 0
            epochs = 0

            # Entrenamiento
            while True:
                oldparams = np.copy(params)
                oldbias = bias
                params, bias = GD(params, bias, X_train, y_train, alfa)
                show_errors(params, bias, X_train, y_train)
                epochs += 1
                if np.allclose(oldparams, params) and np.allclose(oldbias, bias) or epochs == epochs_lim:
                    break

            # Calcular R^2 en conjunto de prueba
            r_squared_test = calc_rsquared(X_test, y_test, params, bias)
            r_squared_results.append((alfa, epochs_lim, r_squared_test))

            # Guardar los errores finales
            final_error = __errors__[-1]
            errors_results.append((alfa, epochs_lim, final_error))

    # Graficar los resultados
    plot_results()
    
#Modelo

def h(params, bias, sample):
    """Evalúa una función lineal h(x) con los parámetros actuales y el bias"""
    return np.dot(sample, params) + bias

def show_errors(params, bias, samples, y):
    """Agrega los errores generados por los valores estimados de h y el valor real y"""
    global __errors__
    hyp = h(params, bias, samples)
    error = hyp - y

    #MSE
    error_acum = np.sum(error**2) / len(y)
    __errors__.append(error_acum)

def GD(params, bias, samples, y, alfa):
    """Algoritmo de Descenso por Gradiente con cálculo del bias"""
    hyp = h(params, bias, samples)
    error = hyp - y
    gradient = np.dot(samples.T, error) / len(y)
    bias_gradient = np.mean(error)

    # Actualizar parámetros y bias
    params -= alfa * gradient
    bias -= alfa * bias_gradient

    return params, bias

def calc_rsquared(X, y, params, bias):
    """Calcula y retorna el R^2"""
    predictions = h(params, bias, X)
    
    ss_residual = np.sum((y - predictions) ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return r_squared

def main():
    
    # Se guardan los datos en un dataframe
    df = pd.read_csv('game_data_all.csv')
    #df.info()
    
    df = clean_data(df)
    
    X_train, X_test, y_train, y_test = split(df)
    
    X_train = scaling(X_train)
    X_test = scaling(X_test)
    
    # Modelo 
    
    X = X_train
    y = y_train
    X.to_numpy()
    y.to_numpy()
    
    X_test.to_numpy()
    X_test = X_test.fillna(0)
    y_test.to_numpy()

    params = np.zeros(X.shape[1])

    bias = 0
    alfa = 0.001
    epochs = 0
    epochs_lim = 4000

    #test_hyper(X_train, y_train, X_test, y_test)
    while True:
        oldparams = np.copy(params)
        oldbias = bias
        params, bias = GD(params, bias, X, y, alfa)
        show_errors(params, bias, X, y)
        epochs += 1
        if np.allclose(oldparams, params) and np.allclose(oldbias, bias) or epochs == epochs_lim:
            print("Final params:", params)
            print("Final bias:", bias)
            break

    r2_test = calc_rsquared(X_test, y_test, params, bias)
    r2_train = calc_rsquared(X, y, params, bias)
    print(f"Test R2: {r2_test}")
                
    print(f"Train R2: {r2_train}")

    # Graficar los errores
    plt.plot(__errors__)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error vs Epoch')
    plt.show()
    
    # Graficar el R^2 para el conjunto de train y test 
    plt.figure(figsize=(8, 6))
    plt.bar(['Train', 'Test'], [r2_train, r2_test], color=['skyblue', 'orange']) 
    plt.ylabel('R-squared')
    plt.title('R-squared para conjuntos de train y test')
    plt.show()

main()