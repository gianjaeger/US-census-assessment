import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def plot_decision_rules(X_train, y_train, feature_names):
    """
    Trains a shallow Decision Tree (Depth 3) to create a readable flowchart
    of the most critical rules for wealth.
    """
    print("Training shallow proxy tree for visualization...")
    
    # Train a shallow tree (max_depth=3 to makes it readable)
    dt_model = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Plot
    plt.figure(figsize=(20, 10))
    
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=['< 50k', '> 50k'],
        filled=True,      
        rounded=True,    
        fontsize=10,
        impurity=False,   
        proportion=True
    )
    
    plt.title("Decision logic flowchart (top 3 levels)", fontsize=16)
    plt.show()

    return dt_model