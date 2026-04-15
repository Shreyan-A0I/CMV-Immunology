import numpy as np
from dataloader import load_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt

# Turns score into a probability between 0 and 1
def sigmoid(z):
    # Clip z to prevent overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss + optional L1 penalty
def compute_log_loss(y_true, y_prob, lambda_param, weights):
    eps = 1e-12
    log_term = y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps)
    logistic_loss = -np.mean(log_term)

    if weights is not None:
        logistic_loss += lambda_param * np.sum(np.abs(weights))

    return logistic_loss

def fit_logistic_regression(X, y, learning_rate, lambda_param, iterations):
    # initialize parameters
    # Set everything to 0 at the start and let gradient descent find the optimal values
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0
    y = y.astype(np.float32)

    # store losses so we can graph them later
    loss_history = []

    # Gradient descent loop
    for step in range(iterations):
        # Linear predictor from lecture slides: theta_0 + sum_j theta_j * x_j
        scores = bias + X @ weights

        # Predicted probabilities P(Y=1|X), Calculate error term, and compute gradients
        probs = sigmoid(scores)
        errors = probs - y
        grad_w = (X.T @ errors) / n_samples
        grad_b = np.mean(errors)

        # Add L1 penalty to weight gradient and update values 
        grad_w += lambda_param * np.sign(weights)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        # Compute and store loss
        # Only store loss every 100 iterations to save time and to make graph more readable
        # Also added a print statement to track progress 
        if step % 100 == 0:
            loss = compute_log_loss(y, probs, lambda_param, weights)
            loss_history.append(loss)
            print(f"Iteration {step}")

    return weights, bias, loss_history

if __name__ == "__main__":
    # load data
    train_data = load_split("processed_data/train.h5ad")
    X_train = train_data["X"]
    y_train = np.asarray(train_data["y_cmv"]).astype(int)
    val_data = load_split("processed_data/val.h5ad")
    X_val = val_data["X"]
    y_val = np.asarray(val_data["y_cmv"]).astype(int)

    # Train model + learn parameters
    # The parameter here is: data, data, learning_rate, lambda_param, iterations
    weights, bias, loss_history = fit_logistic_regression(X_train, y_train, 0.01, 0.001, 1000)

    # Plot training loss over gradient descent steps
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # Infer
    scores = X_val @ weights + bias
    val_probs = sigmoid(scores)
    # Find the best threshold based on F1 score
    # We cant set the threshold to 0.5 because the data is imbalanced 
    # There are more postive samples than negative samples so if we set the threshold 
    # to 0.5, we might classify all samples as positive and get a high recall but low precision
    best_roc = 0
    best_threshold = 0.5 
    for threshold in np.arange(0.1, 0.95, 0.05):
        val_preds = (val_probs >= threshold).astype(int)

        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)
        # f1 = f1_score(y_val, val_preds, zero_division=0)
        roc_auc = roc_auc_score(y_val, val_probs)

        print(f"Threshold={threshold:.2f}, Precision={precision:.4f}, Recall={recall:.4f}, roc_auc={roc_auc:.4f}")

        if roc_auc > best_roc:
            best_roc = roc_auc
            best_threshold = threshold

    # Use the best threshold to convert probabilities into 0/1 labels
    val_preds = (val_probs >= best_threshold).astype(int)

    # Evaluation
    print("Precision:", precision_score(y_val, val_preds, zero_division=0))
    print("Recall:", recall_score(y_val, val_preds, zero_division=0))
    print("F1 Score:", f1_score(y_val, val_preds, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_val, val_probs))
    
