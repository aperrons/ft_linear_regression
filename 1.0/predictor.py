import pickle

# Funzione per calcolare il prezzo stimato
def estimate_price(mileage, theta0, theta1):
    return max(0, theta0 + theta1 * mileage)  # Aggiunto max(0, ...)

def main():
    # Carica i parametri del modello da un file pickle
    with open('model_parameters.pkl', 'rb') as file:
        theta0, theta1 = pickle.load(file)

    # Richiede l'inserimento del chilometraggio
    mileage = float(input("Inserisci il chilometraggio dell'auto: "))

    # Calcola il prezzo stimato
    estimated_price = estimate_price(mileage, theta0, theta1)

    # Mostra il prezzo stimato
    print(f"Il prezzo stimato per un chilometraggio di {mileage} è: {estimated_price}")

if __name__ == "__main__":
    main()
