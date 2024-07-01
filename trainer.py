import numpy as np
import matplotlib.pyplot as plt
import pickle

# Funzione per calcolare il prezzo stimato
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def main():
    # Carica i dati da un file (per esempio 'data.csv')
    data = np.loadtxt('data.csv', delimiter=',', skiprows=1)  # Salta la prima riga con i titoli
    mileage = data[:, 0]
    price = data[:, 1]

    # Normalizzazione dei dati
    mileage_mean = np.mean(mileage)
    mileage_std = np.std(mileage)
    price_mean = np.mean(price)
    price_std = np.std(price)

    mileage_normalized = (mileage - mileage_mean) / mileage_std
    price_normalized = (price - price_mean) / price_std

    # Parametri della discesa del gradiente
    learning_rate = 0.01
    num_iterations = 10000
    m = len(price)

    # Inizializza i parametri theta0 e theta1 a 0
    theta0 = 0
    theta1 = 0

    # Addestramento del modello
    for _ in range(num_iterations):
        tmp_theta0 = theta0 - (learning_rate / m) * np.sum(estimate_price(mileage_normalized, theta0, theta1) - price_normalized)
        tmp_theta1 = theta1 - (learning_rate / m) * np.sum((estimate_price(mileage_normalized, theta0, theta1) - price_normalized) * mileage_normalized)
        theta0 = tmp_theta0
        theta1 = tmp_theta1

    # Denormalizzazione dei parametri
    theta1 = theta1 * (price_std / mileage_std)
    theta0 = theta0 * price_std + price_mean - theta1 * mileage_mean

    # Salva i parametri del modello
    with open('model_parameters.pkl', 'wb') as file:
        pickle.dump((theta0, theta1), file)

    predicted_prices = estimate_price(mileage, theta0, theta1)

    # Plot dei dati e della linea di regressione
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, price, color='blue', label='Dati')
    plt.plot(mileage, predicted_prices, color='red', label='Regressione Lineare')
    plt.xlabel('Chilometraggio')
    plt.ylabel('Prezzo')
    plt.title('Regressione Lineare - Prezzo vs Chilometraggio')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
