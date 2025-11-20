import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_categorical_probability(model, X_encoded, raw_df, category_col):
    """
    Plots the average predicted probability of >50k for each category 
    in a categorical column.
    """
    print(f"Analyzing probability distribution for: {category_col}...")

    # Get model probabialtiies
    probs = model.predict_proba(X_encoded)[:, 1]
    
    # Create temporary dataframe for plotting
    plot_df = pd.DataFrame({
        'Category': raw_df[category_col].values,
        'Probability': probs
    })
    
    # Calculate the mean probability per category and sort
    order = plot_df.groupby('Category')['Probability'].mean().sort_values(ascending=False).index
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.set_style("ticks")
    
    sns.barplot(
        data=plot_df, 
        x='Probability', 
        y='Category', 
        order=order,
        palette='viridis',
        edgecolor='none'
    )
    
    # Style
    sns.despine(top=True, right=True)
    plt.title(f"Model's Wealth Likelihood by {category_col}", fontsize=14)
    plt.xlabel("Average Predicted Probability of Earning >$50k", fontsize=12)
    plt.ylabel("")
    
    # Add a vertical line for the global average
    global_mean = probs.mean()
    plt.axvline(global_mean, color='red', linestyle='--', alpha=0.7)
    plt.text(global_mean + 0.01, 0, "Global Avg", color='red')
    
    plt.tight_layout()
    plt.show()