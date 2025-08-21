import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def get_means(gene1: str, gene2: str, df):
    """
    get means for two arrays to input into predict function, returns concatenated array of means
    """
    col_lst = list(df.columns)

    gene1_lst = []
    gene2_lst = []
    for col in col_lst:
        split_col = col.split(sep = '+')
        if gene1 in split_col[0] and 'ctrl' in split_col[1]: gene1_lst.append(split_col)
        elif gene2 in split_col[0] and 'ctrl' in split_col[1]: gene2_lst.append(split_col)

    final_gene1_lst = ['+'.join(arr) for arr in gene1_lst]
    final_gene2_lst = ['+'.join(arr) for arr in gene2_lst]
    gene1_df = df[final_gene1_lst]
    gene2_df = df[final_gene2_lst]
    mean1_df = np.array(np.mean(gene1_df, 1)).reshape(1000,1)
    mean2_df = np.array(np.mean(gene2_df, 1)).reshape(1000,1)
    return np.concatenate([mean1_df, mean2_df], 1)

def arr_prediction(gene1, gene2, train_df, poisson = True): 
        mean_df = get_means(gene1, gene2, train_df)
        gene1_lst = []
        gene2_lst = []

        col_lst = list(train_df.columns)
        for col in col_lst:
                split_col = col.split(sep = '+')
                if gene1 in split_col[0] and 'ctrl' in split_col[1]: gene1_lst.append(split_col)
                elif gene2 in split_col[0] and 'ctrl' in split_col[1]: gene2_lst.append(split_col)

        gene1_lst = ['+'.join(arr) for arr in gene1_lst]
        gene2_lst = ['+'.join(arr) for arr in gene2_lst]
        gene1_df = train_df[gene1_lst]
        gene2_df = train_df[gene2_lst]
        added = np.array(gene1_df) + np.array(gene2_df) # what about multiplying by factor? grid search?

        #combined = train_df['+'.join([gene1, gene2])]
        gene1_df = np.reshape(gene1_df, (40000,1))
        gene2_df = np.reshape(gene2_df, (40000,1))
        #combined = np.reshape(combined, (1000,1))
        added = np.reshape(added, (40000,1))
        train_concat = np.concatenate([gene1_df, gene2_df, added], axis = 1)
        train_concat.shape

        # perform parameter tuning
        if poisson: model = PoissonRegressor(alpha = 7).fit(train_concat[:, :2], train_concat[:, 2])
        else: model = HistGradientBoostingRegressor(loss = 'poisson', learning_rate = 1, max_iter = 100).fit(train_concat[:, :2], train_concat[:, 2])
        return np.round(model.predict(mean_df), 14)
        
        """
        combined = combined.flatten()
        hist_data = [regmodel_predictions, combined]
        plt.hist(x = hist_data, stacked = True, color = ['red', 'blue'], label = ['regular poisson model', 'test_data'], bins = 30)
        plt.legend()
        plt.tight_layout()
        plt.xlabel('expression')
        plt.ylabel('count')
        plt.title(f'predictions versus test_data dist. ({gene1}, {gene2})')
        plt.show()
        rmse = np.sqrt(mean_squared_error(combined, regmodel_predictions))
        print(f'Regular Poisson Model: RMSE = {rmse:.3f}')
        print(f'shape of predictions: {regmodel_predictions.shape}')
        """
