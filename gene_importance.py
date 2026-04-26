import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import load_split
from method2_logistic_regression import fit_logistic_regression

def get_gene_importance():
    print("Loading data...")
    train_data = load_split("processed_data/train.h5ad")
    genes = train_data["genes"]
    
    # Model parameters (optimized)
    lr, lam, iters = 0.1, 0.001, 1000
    
    print("Running Logistic Regression to extract weights...")
    weights, _, _ = fit_logistic_regression(
        train_data["X"], train_data["y_cmv"], 
        lr, lam, iters, 
        penalty='l2', 
        class_weight='balanced'
    )
    
    # Create Importance Table
    importance_df = pd.DataFrame({'Ensembl': genes, 'Weight': weights})
    
    # Optional: Mapping for top identified genes (for readability)
    mapping = {
        'ENSG00000198763': 'MT-ND2', 'ENSG00000248905': 'FMN1', 
        'ENSG00000067048': 'DDX3Y', 'ENSG00000134539': 'KLRD1 (CD94)',
        'ENSG00000183878': 'UTY', 'ENSG00000163421': 'S100A8',
        'ENSG00000156886': 'MT-ATP6', 'ENSG00000111796': 'S100A9',
        'ENSG00000134545': 'FAM133B', 'ENSG00000078596': 'ITM2A',
        'ENSG00000120738': 'EGR1', 'ENSG00000160856': 'FCRL3'
    }
    importance_df['Symbol'] = importance_df['Ensembl'].map(mapping).fillna(importance_df['Ensembl'])
    
    # Get top 10 positive and 10 negative
    top_pos = importance_df.sort_values('Weight', ascending=False).head(10)
    top_neg = importance_df.sort_values('Weight', ascending=True).head(10)
    plot_df = pd.concat([top_pos, top_neg]).sort_values('Weight')

    # Visualizing
    print("\nGenerating Gene Importance Plot...")
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    colors = ['#ff7675' if x > 0 else '#74b9ff' for x in plot_df['Weight']]
    plt.barh(plot_df['Symbol'], plot_df['Weight'], color=colors)
    
    plt.title('Top Gene Contributors (Method 2: Logistic Regression)', fontsize=15, pad=20)
    plt.xlabel('Coefficient Weight', fontsize=12)
    plt.axvline(0, color='black', lw=1)
    
    # Annotate sides
    plt.text(plot_df['Weight'].max()*0.5, 0, 'Predicts CMV+', color='#d63031', fontweight='bold', alpha=0.6)
    plt.text(plot_df['Weight'].min()*0.5, 0, 'Predicts CMV-', color='#0984e3', fontweight='bold', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("plots/gene_importance.png", dpi=300)
    print("Saved plot to plots/gene_importance.png")
    
    # Save CSV
    importance_df.to_csv("results/gene_weights.csv", index=False)
    print("Saved weights to results/gene_weights.csv")

if __name__ == "__main__":
    import os
    if not os.path.exists('results'): os.makedirs('results')
    get_gene_importance()
