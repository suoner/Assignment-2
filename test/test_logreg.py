import sys
sys.path.insert(0, '..')
import numpy as np
import pytest
from regression.logreg import LogisticRegression

# Generating fake data for testing
np.random.seed(42)
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train = np.random.randint(0, 2, 100)  # 100 binary labels

# Initialize the model
model = LogisticRegression(num_feats=5, max_iter=10, tol=0.01, learning_rate=0.001, batch_size=10)

def test_updates():
    """
    Test gradient calculation and loss function.
    """
    # Add a column for the bias term in X_train
    X_train_padded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

    # Initial gradients and loss
    initial_gradient = model.calculate_gradient(X_train_padded, y_train)
    initial_loss = model.loss_function(X_train_padded, y_train)

    # Check that gradient shape is correct
    assert initial_gradient.shape == model.W.shape, "Gradient shape mismatch with weights"

    # Check that the gradient is not vanishing (all close to zero) or exploding (very large values)
    assert np.all(np.abs(initial_gradient) < 10), "Gradient exploding"
    assert np.all(np.abs(initial_gradient) > 1e-6), "Gradient vanishing"

    # Ensure the loss is within a reasonable range
    assert initial_loss > 0, "Initial loss should be positive"
    assert initial_loss < 10, "Initial loss is unusually high"

    # Train the model and check if loss decreases
    model.train_model(X_train, y_train, X_train, y_train)
    final_loss = model.loss_function(X_train_padded, y_train)
    assert final_loss < initial_loss, "Loss did not decrease after training"

def test_predict():
    """
    Test weight updates and predict functionality.
    """
    # Train the model on the training data
    model.train_model(X_train, y_train, X_train, y_train)

    # Check that weights have been updated (i.e., not equal to initial weights)
    assert not np.allclose(model.W, np.zeros_like(model.W)), "Weights did not update during training"

    # Add a column for the bias term in X_train
    X_train_padded = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

    # Test predict functionality on training data
    predictions = model.make_prediction(X_train_padded)

    # Check the output is binary (0 or 1) for classification
    assert np.all(np.isin(predictions, [0, 1])), "Predictions should be binary (0 or 1)"