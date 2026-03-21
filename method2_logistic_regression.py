from dataloader import load_split
import numpy as np

class CustomL1LogisticRegression:
    """
    Method 2: Self-implemented L1 Logistic regression to predict CMV disease from gene expression.
    """
    def __init__(self, learning_rate=0.01, lambda_param=0.1, iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # TODO: Implement L1 regularization (Lasso) in the gradient descent loop
        pass

    def predict(self, X):
        # TODO: Implement prediction logic
        pass

def run_logreg_analysis():
    # Load data
    train_data = load_split("processed_data/train.h5ad")
    val_data = load_split("processed_data/val.h5ad")

    X_train = train_data["X"]
    y_cmv_train = train_data["y_cmv"]
    
    X_val = val_data["X"]
    y_cmv_val = val_data["y_cmv"]

    print(f"Training L1 Logistic Regression on {X_train.shape[0]} samples...")
    
    # model = CustomL1LogisticRegression()
    # model.fit(X_train, y_cmv_train)
    
    pass

if __name__ == "__main__":
    run_logreg_analysis()
