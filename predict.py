import re
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from models.vanilla_vae import VanillaVAE as VAE
from data_dict import *
import warnings

warnings.filterwarnings("ignore")



def predict(code, X_test, platform, pickle_file, model_type, data_type):
    data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                 'radical_data': radical_data, 'cube_data': cube_data}
    model_dict = {'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression, 'L1': Lasso,
                  'L2': Ridge, 'VAE': VAE}

    with open(platform, 'r') as f:
        gene_dict = json.load(f)
        f.close()

    count = 0
    num = len(gene_dict)
    gene_list = []
    print('Now start predict gene...')
    data_test = data_dict[data_type](X_test)
    with open(pickle_file, 'rb') as f:
        while True:
            try:
                count += 1
                temp = pickle.load(f)
                gene = temp[0]
                gene_data_test = []
                for residue in data_test.index:
                    if residue in gene_dict[gene]:
                        gene_data_test.append(data_test.loc[residue])
                gene_data_test = np.array(gene_data_test)
                gene_list.append(gene)
                # print('Now predicting ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(int(count * 100 / num)) + '% ...')
                model = temp[1]
                pred2 = model.predict(gene_data_test.T)
                if count == 1:
                    data_test_pred = pred2.T
                else:
                    data_test_pred = np.vstack([data_test_pred, pred2.T])
                print('finish!')

            except EOFError:
                break

    data_test_pred = pd.DataFrame(np.array(data_test_pred), index=gene_list)
    data_test_pred.to_csv(code + "_gene_level" + "(" + data_type + '_' + model_type + ").txt", sep='\t')
    print("Predicting finish!")


if __name__ == '__main__':
    # Parameter description：
    # code: dataSet ID such as GSE66695 ( string )
    # test_file: test file name( .txt )
    # platform: Gene correspond to methylation characteristics( json file )
    # pickle_file: Parameters of regression model( pickle file )
    # model_type: type of regression model ( string )
    # data_type: type of data ( string )

    # example
    code = "GSE66695"
    test_file = "data_test.txt"
    platform = "platform.json"
    pickle_file = "GSE66695_LinearRegression_origin_datatrain_model.pickle"
    model_type = "VAE"
    data_type = "origin_data"

    test_data = pd.read_table(test_file, index_col=0)

    # predict(code,test_data,platform,pickle_file,model_type, data_type)
