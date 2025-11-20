import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def plot_pdp(model, X_train, features_to_plot):
    """
    Plots the Partial Dependence of specific features.
    Shows the non-linear relationship between a feature and the probability of >50k.
    """
    print("Generating Partial Dependence Plots...")
    
    # Setting to 10000
    X_sample = X_train.sample(n=10000, random_state=42)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Display
    PartialDependenceDisplay.from_estimator(
        model,
        X_sample,
        features=features_to_plot,
        kind="average",
        n_cols=3,
        ax=ax,
        line_kw={"color": "blue", "linewidth": 2}
    )
    
    plt.suptitle("How Key Features Impact the Probability of High Income", fontsize=16, y=1.05)
    plt.show()