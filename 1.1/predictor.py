import numpy as np

def load_model(filename):
    with open(filename, 'r') as file:
        theta = np.array([float(x) for x in file.read().split()])
    return theta

def estimate_price(mileage, theta):
    return theta[0] + theta[1] * mileage

def main():
    filename = 'model.txt'
    theta = load_model(filename)
    
    mileage = float(input("Enter the mileage: "))
    price = estimate_price(mileage, theta)
    
    print(f"The estimated price for {mileage} mileage is: {price}")

if __name__ == "__main__":
    main()
