import numpy as np
from dataloader import load_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Turns score into a probability between 0 and 1
def sigmoid(z):
    # Clip z to prevent overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy loss + L1/L2 penalty
def compute_log_loss(y_true, y_prob, lambda_param, weights, penalty='l1'):
    eps = 1e-12
    log_term = y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps)
    logistic_loss = -np.mean(log_term)

    if weights is not None:
        if penalty == 'l1':
            logistic_loss += lambda_param * np.sum(np.abs(weights))
        else: # l2
            logistic_loss += (lambda_param / 2) * np.sum(weights**2)

    return logistic_loss

def fit_logistic_regression(X, y, learning_rate, lambda_param, iterations, penalty='l1', class_weight=None):
    # Parameters needed for training
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0
    y = y.astype(np.float32)

    # Class weights for Balancing
    if class_weight == 'balanced':
        num_pos = np.sum(y)
        num_neg = n_samples - num_pos
        w1 = n_samples / (2 * num_pos)
        w0 = n_samples / (2 * num_neg)
        sample_weights = np.where(y == 1, w1, w0)
        print(f"Using Balanced weights: Positive={w1:.2f}, Negative={w0:.2f}")
    else:
        sample_weights = np.ones(n_samples)

    # Store losses so we can graph them later
    loss_history = []
    tol_loss = 1e-6 # Early stopping tolerance

    # Gradient descent loop
    for step in range(iterations):
        # Linear predictor
        scores = bias + X @ weights

        # Predicted probabilities
        probs = sigmoid(scores)
        
        # Weighted error
        errors = (probs - y) * sample_weights
        
        # Calculate gradients
        grad_w = (X.T @ errors) / n_samples
        grad_b = np.mean(errors)

        # Add penalty to weight gradient
        if penalty == 'l1':
            grad_w += lambda_param * np.sign(weights)
        else: # l2
            grad_w += lambda_param * weights

        # Update values 
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        # Compute and store loss every 100 iterations
        if step % 100 == 0:
            loss = compute_log_loss(y, probs, lambda_param, weights, penalty)
            loss_history.append(loss)
            print(f"Iteration {step} | Loss: {loss:.4f}")
            
            # Early stopping check
            if len(loss_history) > 1 and abs(loss_history[-2] - loss_history[-1]) < tol_loss:
                print(f"Converged early at iteration {step}")
                break

    return weights, bias, loss_history

if __name__ == "__main__":
    train_data = load_split("processed_data/train.h5ad")
    X_train = train_data["X"]
    y_train = np.asarray(train_data["y_cmv"]).astype(int)
    val_data = load_split("processed_data/val.h5ad")
    X_val = val_data["X"]
    y_val = np.asarray(val_data["y_cmv"]).astype(int)

    # Use optimized hyperparameters 
    learning_rate = 0.1
    lambda_param = 0.001
    iterations = 1000
    penalty = 'l2'
    class_weight = 'balanced'

    print(f"Training optimized LR with LR={learning_rate}, Penalty={penalty.upper()}, Balanced={class_weight is not None}...")
    weights, bias, loss_history = fit_logistic_regression(
        X_train, y_train, learning_rate, lambda_param, iterations, 
        penalty=penalty, 
        class_weight=class_weight
    )

    # Plot training loss over gradient descent steps
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(loss_history)*100, 100), loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Optimized Logistic Regression Training Loss ({penalty.upper()}, Balanced)")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/method2_optimized_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Infer
    scores = X_val @ weights + bias
    val_probs = sigmoid(scores)
    
    # Find the best threshold based on F1 score
    best_f1 = 0
    best_threshold = 0.5 
    print("\nSearching for optimal confidence threshold:")
    for threshold in np.arange(0.1, 0.95, 0.025):
        val_preds = (val_probs >= threshold).astype(int)
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        roc_auc = roc_auc_score(y_val, val_probs)

        print(f"Threshold={threshold:.3f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Use the best threshold to convert probabilities into 0/1 labels
    print(f"\nFinal Best Threshold: {best_threshold:.3f}")
    val_preds = (val_probs >= best_threshold).astype(int)

    print("\nResults:")
    print(f"Precision: {precision_score(y_val, val_preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, val_preds, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_val, val_preds, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
    
    # Final Performance Plots
    plt.figure(figsize=(14, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, val_probs)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_val, val_probs)
    pr_auc = auc(rec, prec)
    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.axhline(y=np.mean(y_val), color='grey', linestyle='--', label='Random Chance')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/method2_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
