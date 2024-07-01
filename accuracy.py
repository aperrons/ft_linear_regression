import numpy as np
import pickle
from sklearn.metrics import r2_score

def main():
    # Carica i dati da un file (per esempio 'data.csv')
    data = np.loadtxt('data.csv', delimiter=',', skiprows=1)  # Salta la prima riga con i titoli
    mileage = data[:, 0]
    price = data[:, 1]

    # Carica il modello addestrato
    with open('model_parameters.pkl', 'rb') as file:
        theta0, theta1 = pickle.load(file)

    # Predici i prezzi
    predicted_prices = theta0 + theta1 * mileage

    # Calcola e stampa l'R^2
    r2 = r2_score(price, predicted_prices)
    print(f"R^2: {r2}")

if __name__ == "__main__":
    main()
