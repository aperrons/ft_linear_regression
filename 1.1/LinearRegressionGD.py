import numpy as np
import matplotlib.pyplot as plt
import pickle

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, max_iterations=10000, eps=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.eps = eps

def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def descend(x, y, theta0, theta1, learning_rate):
    dldtheta0 = 0.0
    dldtheta1 = 0.0
    N = x.shape[0]
    #loss = (y - yhat)**2 == (y - (wx +b))**2

    for xi, yi in zip(x, y):
        dldtheta0 += -2 * xi * (yi - (theta0 * xi + theta1))
        dldtheta1 += -2 * (yi - (theta0 * xi + theta1))
    
    theta0 = theta0 - learning_rate * (1/N)*dldtheta0
    theta1 = theta1 - learning_rate * (1/N)*dldtheta1

    return theta0, theta1

def estimate_price(x, theta0, theta1):
        return theta0 + theta1 * x

def main():
    x, y = load_data('../data.csv')

    theta0 = 0.0
    theta1 = 0.0
    learning_rate = 0.00000000001

    for epoch in range(1000):
        theta0, theta1 = descend(x, y, theta0, theta1, learning_rate)
        yhat = theta0 * x + theta1
        loss = np.divide(np.sum((y - yhat)**2, axis=0), x.shape[0])
        print(f'{epoch} loss is {loss}, parameters theta0: {theta0}, theta1: {theta1}')
    
    with open('model_parameters.pkl', 'wb') as file:
        pickle.dump((theta0, theta1), file)

    predicted_prices = estimate_price(x, theta0, theta1)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Dati')
    plt.plot(x, predicted_prices, color='red', label='Regressione Lineare')
    plt.xlabel('Chilometraggio')
    plt.ylabel('Prezzo')
    plt.title('Regressione Lineare - Prezzo vs Chilometraggio')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

