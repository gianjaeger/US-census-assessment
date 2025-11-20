import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def train_scaled_logistic(X_train, y_train):
    """
    Scales continuous variables and trains a Logistic Regression
    Returns: model, scaler, scaled_feature_names
    """
    # Continuous columns to scale
    continuous_cols = [
        'age', 
        'wage_per_hour', 
        'capital_gains', 
        'capital_losses', 
        'dividends_from_stocks', 
        'num_persons_worked_for_employer', 
        'weeks_worked'
    ]
    
    # Safety check
    cols_to_scale = [c for c in continuous_cols if c in X_train.columns]
    
    # Create a coüy to avoid messing up the RF data
    X_train_scaled = X_train.copy()
    
    # Initialize and fit scaler
    scaler = StandardScaler()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    
    # Train logistic regression
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, cols_to_scale

def plot_coefficients(model, feature_names, top_n=20):
    """
    Plots a diverging bar chart for the top coefficients
    """
    # Extract coefficients
    coeffs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0] 
    })
    
    # Sort by absolute impact (i.e., magnitude)
    coeffs['Abs_Coeff'] = coeffs['Coefficient'].abs()
    top_coeffs = coeffs.sort_values(by='Abs_Coeff', ascending=False).head(top_n)
    
    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top_coeffs['Coefficient']]
    
    ax = sns.barplot(
        data=top_coeffs, 
        x='Coefficient', 
        y='Feature', 
        palette=colors,
        edgecolor='none'
    )
    
    plt.title('Directionality of Income Drivers (Logistic Regression Coeffs)', fontsize=14)
    plt.xlabel('Impact on Likelihood of >$50k (Standardized)', fontsize=12)
    plt.ylabel('')
    
    plt.axvline(0, color='black', linewidth=1, linestyle='-')
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def evaluate_logistic(model, scaler, cols_to_scale, X_test, y_test):
    """
    Scales the test set and prints the evaluation metrics
    """
    # Scale test set
    X_test_scaled = X_test.copy()
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Classification Report
    print("--- Logistic Regression Evaluation ---")
    
    # Calculate balanced accuracy (to compare with RF)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {bal_acc:.2%}")
    print("-" * 30)
    
    print(classification_report(y_test, y_pred))
    
    # Plot normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=['< 50k', '> 50k'],
                yticklabels=['< 50k', '> 50k'])
    plt.title("Confusion Matrix (Logistic Regression)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()