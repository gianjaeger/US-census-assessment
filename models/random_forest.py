import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def tune_and_train(X_train, y_train):
    """
    Performs 3-Fold CV to optimize Depth, Size, Leaf Size, and Feature Selection.
    Returns the best trained model.
    """
    param_grid = {
        # 1. How deep?
        'max_depth': [10, 20, None],
        
        # 2. How many trees?
        'n_estimators': [100, 200],
        
        # 3. Leafe size?
        'min_samples_leaf': [1, 4], 
        
        # 4. Number of features checked at each split
        'max_features': ['sqrt', 'log2']
    }
    
    # Grid Search
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring='balanced_accuracy', # IMPORTANT
        verbose=1
    )
    
    print(f"Starting Grid Search on {len(X_train)} rows...")
    print("Optimizing: Depth, Estimators, Min_Leaf, and Max_Features")
    grid.fit(X_train, y_train)
    
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Score (Balanced Acc): {grid.best_score_:.2%}")
    
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Prints report and plots Confusion Matrix."""
    y_pred = model.predict(X_test)
    
    print("--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=['< 50k', '> 50k'],
                yticklabels=['< 50k', '> 50k'])
    plt.title("Confusion Matrix (Normalized)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(top_n)

    sns.set_style("ticks")
    
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.barplot(
        data=importances, 
        x='Importance', 
        y='Feature', 
        palette='viridis', 
        edgecolor='none',
        ax=ax
    )

    sns.despine(top=True, right=True)
    
    # Customizing Gridlines
    ax.minorticks_on() 
    
    # Grid lines formating
    ax.grid(visible=True, which='major', axis='x', color='#cccccc', linestyle='--', alpha=0.8, zorder=0)
    ax.grid(visible=True, which='minor', axis='x', color='#ebebeb', linestyle=':', alpha=1.0, zorder=0)

    # Labels
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xlabel("Relative Importance (Mean Decrease in Impurity)", fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.show()