import re
import os
import json
import numpy as np
import pandas as pd
import torch
from scipy import stats
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from torch import nn
from keras import backend as K, losses
import AutoEncoder as AE
import warnings
from keras.models import load_model
from keras.models import Model  # 泛型模型
warnings.filterwarnings("ignore")


def origin_data(data):
    return data


def square_data(data):
    return data ** 2


def log_data(data):
    return np.log(data + 1e-5)


def radical_data(data):
    return data ** (1 / 2)


def cube_data(data):
    return data ** 3


def predict(path,date,code, X_test,Y_test, platform, pickle_file, model_type, data_type,model,predict_model_type,residue_name_list=[],
            datasetNameList=[]
            ):
    data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                 'radical_data': radical_data, 'cube_data': cube_data}
    model_dict = {'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression, 'L1': Lasso,
                  'L2': Ridge,'AE':AE.Autoencoder}

    with open(platform, 'r') as f:
        gene_dict = json.load(f)
        f.close()

    count = 0
    num = len(gene_dict)
    gene_list = []
    print('Now start predict gene...')
    data_test = data_dict[data_type](X_test)
    print("data_test")
    print(data_test)
    '''
    first_col=data_test.index
    #data_test_filtered=pd.DataFrame(data_test[])
    first_time_add_pd=True
    for i in first_col:
        if i in residue_name_list:
            if first_time_add_pd:
                first_time_add_pd=False
                data_test_filtered = pd.DataFrame(data_test.loc[i])
            else:
                data_test_filtered[i]=data_test.loc[i]
    #data_test=data_test[first_col in residue_name_list]
    print("data_test after selecting residue")
    print(data_test_filtered)
    data_test=data_test_filtered
    '''
    if False:
        data_test_pred=None

        gene_data_test=data_test
        # model=LinearRegression(gene_data_test)
        if (model_type == 'AE'):
            hidden_size = 15
            print("gene_data_test.shape")
            print(gene_data_test.shape)
            # model=AE.Autoencoder(in_dim=gene_data_test.shape[1], h_dim=hidden_size)
            model = torch.load('network.pth')  # load network from parameters saved in network.pth @ 22-2-18
            # images = AE.to_var(gene_data_test.T.view(gene_data_test.T.size(0), -1))
            # images = images.float()
            #gene_data_test = torch.from_numpy(gene_data_test)
            gene_data_test = AE.to_var(gene_data_test.view(gene_data_test.size(0), -1))
            gene_data_test = gene_data_test.float()
            out = model(gene_data_test)
            out = out.view(out.size(0), -1)
            pred2 = out.detach().numpy()

        else:
            pred2 = model.predict(gene_data_test.T)

        if count == 1:
            data_test_pred = pred2.T
        else:
            print("data_test_pred")
            print(data_test_pred)
            print("pred2.T")
            print(pred2.T)
            data_test_pred = np.vstack([data_test_pred, pred2.T])
        print('finish!')

    gene_data_test = []
    residue_is_added={}
    for i in residue_name_list:
        residue_is_added[i]=False
    #residue_name_list=[]
    if True:
        with open(path+pickle_file, 'rb') as f:
            while True:
                try:
                    count += 1
                    if count % 1000 == 0:
                        print("count=%d"% count)
                    temp = pickle.load(f)
                    gene = temp[0]
                    if(model_type!='AE'):
                        gene_data_test = []
                    print_flag=False
                    for iii,residue in enumerate(data_test.index):
                        percentage=int(float(iii)/len(data_test.index)*100)
                        if count % 1000==0 and percentage % 50 ==0 and print_flag==False:
                            print_flag=True
                            print("now in data test index %d ,%f percent"%(iii,percentage))

                        if residue in gene_dict[gene] and (residue in residue_name_list) and residue_is_added[residue]==False:
                            #residue_name_list.append(residue)
                            residue_is_added[residue]=True
                            gene_data_test.append(data_test.loc[residue])
                    if (model_type != 'AE'):
                        gene_data_test = np.array(gene_data_test)
                    gene_list.append(gene)
                    #print("gene_data_test")
                    #print(gene_data_test)
                    #print("gene_list")
                    #print(gene_list)
                    # print('Now predicting ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(int(count * 100 / num)) + '% ...')
                    model = temp[1] # deleted

                    #model=LinearRegression(gene_data_test)
                    if(model_type=='AE'):
                        pass
                    else:
                        pred2 = model.predict(gene_data_test.T)
                        if count == 1:
                            data_test_pred = pred2.T
                        else:
                            print("data_test_pred")
                            print(data_test_pred)
                            print("pred2.T")
                            print(pred2.T)
                            data_test_pred = np.vstack([data_test_pred, pred2.T])
                        print('finish!')


################################################################################
                except EOFError:
                    break
        if(model_type=='AE') :
            relu_thresh=0
            l_rate=K.variable(0.01)
            def relu_advanced(x):
                return K.relu(x, threshold=relu_thresh)
            # L1 regularizer with the scaling factor updateable through the l_rate variable (callback)
            def variable_l1(weight_matrix):
                return l_rate * K.sum(K.abs(weight_matrix))

            def myLoss(y_true, y_pred):
                return losses.binary_crossentropy(y_true, y_pred)

            #loaded_autoencoder = load_model(path+date + 'AE.h5',custom_objects={'variable_l1': variable_l1,'relu_advanced':relu_advanced})
            loaded_fcn = load_model(path+date + 'FCN.h5',custom_objects={'relu_advanced':relu_advanced,'myLoss':myLoss})
            gene_data_test = np.array(gene_data_test)
            #hidden_size = 15
            print("gene_data_test.shape")
            print(gene_data_test.shape)
            '''
            model_ae=torch.load(date+'_auto-encoder.pth')
            model_nn = torch.load(date+'_fully-connected-network.pth')  # load network from parameters saved in network.pth @ 22-2-18
            gene_data_test = torch.from_numpy(gene_data_test)
            gene_data_test = AE.to_var(gene_data_test.view(gene_data_test.size(0), -1))
            gene_data_test = gene_data_test.float()
            embedding = model_ae.code(gene_data_test.T)
            '''

            input_to_encoding_model = Model(inputs=loaded_fcn.input,
                                       outputs=loaded_fcn.get_layer('input_to_encoding').output)
            # embedding=ae.code(torch.tensor(gene_data_train.T).float())
            embedding = input_to_encoding_model.predict(gene_data_test.T)

            fcn_predict_model = Model(inputs=loaded_fcn.input,
                                            outputs=loaded_fcn.get_layer('prediction').output)

            print("predicting:after ae, embedding is ")
            print(embedding)
            print(embedding.shape)
            [ae_out,pred_out] = loaded_fcn.predict(gene_data_test.T)
            print("ae_out is")
            print(ae_out)
            print("prediction is")
            print(pred_out)

            normalized_pred_out = [[0]*len(datasetNameList) for i in range(len(pred_out))]
            num_wrong_pred = 0
            if len(datasetNameList)>1:

                for i,item in enumerate(pred_out):
                    for i_dataset, datasetName in enumerate(datasetNameList):
                        if item[i_dataset] >= 0.5:
                            normalized_pred_out[i_dataset][i]=1
                            num_wrong_pred += round(abs(Y_test.iloc[i]-1.0))
                        elif item[i_dataset] < 0.5:
                            normalized_pred_out[i_dataset][i]=0
                            num_wrong_pred += round(abs(Y_test.iloc[i] - 0.0))
            elif len(datasetNameList) == 1:
                for i,item in enumerate(pred_out):
                    if item >= 0.5:
                        normalized_pred_out.append(1)
                        num_wrong_pred += round(abs(Y_test.iloc[i]-1.0))
                    elif item < 0.5:
                        normalized_pred_out.append(0)
                        num_wrong_pred += round(abs(Y_test.iloc[i] - 0.0))

            print("normalized pred_out=")
            print(normalized_pred_out)
            print("test label is")
            print(Y_test)
            print("num_wrong_pred=%d, total test num=%d,accuracy=%f"%(num_wrong_pred,len(Y_test),1.0-num_wrong_pred/len(Y_test)))

            #out = out.view(out.size(0), -1)
            data_test_pred = pred_out#.numpy()
            normalized_data_test_pred = normalized_pred_out
            #print('Now predicting ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(int(count * 100 / num)) + '% ...')

            '''if count == 1:
                data_test_pred = pred2.T
            else:
                print("data_test_pred")
                print(data_test_pred)
                print("pred2.T")
                print(pred2.T)
                data_test_pred = np.vstack([data_test_pred, pred2.T])'''
            print('finish!')

    data_test_pred = pd.DataFrame(np.array(data_test_pred))
    data_test_pred.to_csv(path+date+"_"+code + "_gene_level" + "(" + data_type + '_' + model_type + ").txt", sep='\t')
    data_test_ae_out = pd.DataFrame(np.array(ae_out))
    data_test_ae_out.to_csv(path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "AE_output).txt",
                          sep='\t')

    normalized_data_test_pred = pd.DataFrame(np.array( normalized_data_test_pred))
    normalized_data_test_pred.to_csv(path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "normalized_pred).txt",
                          sep='\t')

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
    model_type = "LinearRegression"
    data_type = "origin_data"

    test_data = pd.read_table(test_file, index_col=0)

    predict(code,test_data,platform,pickle_file,model_type, data_type)
