import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def main():
    # Load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
                                                                  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # Scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # Initialize logistic regression model
    log_model = logreg.LogisticRegression(num_feats=X_train.shape[1], max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    
    # Plot loss history to visualize training progress
    log_model.plot_loss_history()
    
    # Evaluate on validation set
    y_pred = log_model.make_prediction(X_val)
    accuracy = np.mean(y_pred == y_val)
    print("Validation Accuracy:", accuracy)

if __name__ == "__main__":
    main()
