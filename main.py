import pandas as pd
import numpy as np
from prediction_functions import arr_prediction
from processing import get_test_data

gene_perturb_lst = get_test_data(pd.read_csv('data//test_set.csv', header = None, dtype = str))
training_df = pd.read_csv('data//train_set.csv')
gene_labels = np.array([f'g{str(i+1).zfill(4)}' for i in range(1000)]).reshape(1000, 1)
perturb_arr = np.empty((len(gene_perturb_lst), 1000, 3), dtype = object)
for gene_lst_i, gene_pair in enumerate(gene_perturb_lst):
    gene1, gene2 = gene_pair[0], gene_pair[1]
    perturb_str = '+'.join([gene1, gene2])
    perturb_lst = np.array([perturb_str]*1000).reshape(1000, 1)
    predictions = arr_prediction(gene1, gene2, training_df).reshape(1000, 1).astype(str)
    # concat all predictions together, shape should be (1000, 3)
    perturb_arr[gene_lst_i] = np.concatenate([gene_labels, perturb_lst, predictions], 1)

perturb_arr = perturb_arr.reshape(perturb_arr.shape[0]*perturb_arr.shape[1], perturb_arr.shape[2])
perturb_df = pd.DataFrame(perturb_arr, columns = ['gene', 'perturbation', 'expression'])
perturb_df.to_csv('prediction/prediction.csv', index = False)