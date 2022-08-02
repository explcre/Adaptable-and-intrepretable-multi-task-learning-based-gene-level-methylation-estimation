import re
import os
import json
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
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
from MeiNN.config import config
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


def predict(path,date,code, X_test,Y_test, platform, pickle_file, model_type, data_type,HIDDEN_DIMENSION, toTrainMeiNN,
            model,predict_model_type,residue_name_list=[],
            datasetNameList=[],separatelyTrainAE_NN=False,multiDatasetMode="multi-task",
            toAddGenePathway = False, toAddGeneSite = False,
            num_of_selected_residue = 1000, lossMode = 'reg_mean', selectNumPathwayMode = '=num_gene',
            num_of_selected_pathway = 500,
            AE_epoch_from_main = 1000, NN_epoch_from_main = 1000, gene_pathway_dir = "./dataset/GO term pathway/matrix.csv",
            pathway_name_dir = "./dataset/GO term pathway/gene_set.txt",
            gene_name_dir = "./dataset/GO term pathway/genes.txt",
            framework='keras'
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

            def maskedDatasetLoss(y_true, y_pred):
                ans = 0
                if not y_true == 0.5:
                    return (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

                return ans

            #loaded_autoencoder = load_model(path+date + 'AE.h5',custom_objects={'variable_l1': variable_l1,'relu_advanced':relu_advanced})
            gene_data_test = np.array(gene_data_test)
            ae_out=None
            print("separatelyTrainAE_NN=")
            print(separatelyTrainAE_NN)
            print("multiDatasetMode=")
            print(multiDatasetMode)

            if framework == 'keras':
                if separatelyTrainAE_NN:

                    autoencoder = load_model(path + date + 'AE.h5',
                                    custom_objects={'relu_advanced': relu_advanced,'explainableAELoss':myLoss})
                    embedding2pred_nn = load_model(path + date + 'embedding2pred_nn.h5',
                                     custom_objects={'relu_advanced': relu_advanced,'explainableAELoss':myLoss})
                    input_to_encoding_model = Model(inputs=autoencoder.input,
                                            outputs=autoencoder.get_layer('input_to_encoding').output)
                    ae_out = autoencoder.predict(gene_data_test.T)
                    embedding=input_to_encoding_model.predict(gene_data_test.T)
                    pred_out= embedding2pred_nn.predict(embedding)
                else:#train AE and NN together
                    if multiDatasetMode=="multi-task":
                        print("DEBUG INFO: in the multi-task")
                        loaded_fcn_multitask = load_model(path + date + 'multi-task-MeiNN.h5',
                                            custom_objects={'relu_advanced': relu_advanced, 'myLoss': myLoss})
                        print(loaded_fcn_multitask.summary())
                        print("datasetname list length: %d"%len(datasetNameList))
                        print(datasetNameList)
                        print("gene_data_test.shape")
                        print(gene_data_test.shape)

                        input_to_encoding_model = Model(inputs=loaded_fcn_multitask.input,
                                                    outputs=loaded_fcn_multitask.get_layer('input_to_encoding').output)

                        # embedding=ae.code(torch.tensor(gene_data_train.T).float())
                        embedding = input_to_encoding_model.predict(gene_data_test.T)

                        #fcn_predict_model = Model(inputs=loaded_fcn_multitask.input,
                        #                          outputs=loaded_fcn_multitask.get_layer('prediction').output)
                        data_test_pred=None
                        if len(datasetNameList) > 1:
                            [ae_out, pred_out1,pred_out2,pred_out3,pred_out4,pred_out5,pred_out6] = loaded_fcn_multitask.predict(gene_data_test.T)
                            print("ae_out is")
                            print(ae_out)
                            print("prediction%d is" % 1)

                            print(pred_out1)
                            print("prediction%d is" % 2)
                            print(pred_out2)
                            print("prediction%d is" % 3)
                            print(pred_out3)
                            print("prediction%d is" % 4)
                            print(pred_out4)
                            print("prediction%d is" % 5)
                            print(pred_out5)
                            print("prediction%d is" % 6)
                            print(pred_out6)

                            # data_test_pred = [pred_out1,pred_out2,pred_out3,pred_out4,pred_out5,pred_out6]
                            # data_test_pred = pred_out
                            data_test_pred = pd.DataFrame(np.array(pred_out1))
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred1).txt", sep='\t')
                            data_test_pred = pd.DataFrame(np.array(pred_out2))
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred2).txt", sep='\t')
                            data_test_pred = pd.DataFrame(np.array(pred_out3))
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred3).txt", sep='\t')
                            data_test_pred = pd.DataFrame(np.array(pred_out4))
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred4).txt", sep='\t')
                            data_test_pred = pd.DataFrame(np.array(pred_out5))
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred5).txt", sep='\t')
                            data_test_pred = pd.DataFrame(np.array(pred_out6))
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred6).txt", sep='\t')
                            data_test_ae_out = pd.DataFrame(np.array(ae_out))
                            data_test_ae_out.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "AE_output).txt",
                                sep='\t')
                    else:
                        loaded_fcn = load_model(path+date + 'MeiNN.h5',custom_objects={'relu_advanced':relu_advanced,'myLoss':myLoss,'maskedDatasetLoss':maskedDatasetLoss})

                        #hidden_size = 15
                        print("gene_data_test.shape")
                        print(gene_data_test.shape)

                        input_to_encoding_model = Model(inputs=loaded_fcn.input,
                                       outputs=loaded_fcn.get_layer('input_to_encoding').output)
                        # embedding=ae.code(torch.tensor(gene_data_train.T).float())
                        embedding = input_to_encoding_model.predict(gene_data_test.T)

                        fcn_predict_model = Model(inputs=loaded_fcn.input,
                                            outputs=loaded_fcn.get_layer('prediction').output)

                        [ae_out,pred_out] = loaded_fcn.predict(gene_data_test.T)
                        # evaluate the model
                        score = loaded_fcn.evaluate(gene_data_test.T, [gene_data_test.T,Y_test.T], verbose=0)
                        print("FCN score")
                        print(score)
                        print('FCN Test score:', score[0])
                        print('FCN Test accuracy:', score[1])

                        fcn_predict_model.compile(optimizer='Adam',loss='binary_crossentropy')
                        score_pred = fcn_predict_model.evaluate(gene_data_test.T, Y_test.T, verbose=0)
                        print("prediction score")
                        print(score_pred)
                        print("ae_out is")
                        print(ae_out)
                        print("prediction is")
                        print(pred_out)
                        data_test_pred = pred_out
                        data_test_pred = pd.DataFrame(np.array(data_test_pred))
                        data_test_pred.to_csv(
                            path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                            str(separatelyTrainAE_NN) + "pred).txt", sep='\t')
                        data_test_ae_out = pd.DataFrame(np.array(ae_out))
                        data_test_ae_out.to_csv(
                            path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "AE_output).txt",
                            sep='\t')
                        # print('prediction Test score:', score_pred[0])
                        # print('prediction Test accuracy:', score_pred[1])
                        normalized_pred_out = [[0] * len(datasetNameList) for i in range(len(pred_out))]
                        num_wrong_pred = 0
                        if len(datasetNameList) > 1:
                            for i, item in enumerate(pred_out):
                                for i_dataset, datasetName in enumerate(datasetNameList):
                                    if item[i_dataset] >= 0.5:
                                        normalized_pred_out[i_dataset][i] = 1
                                        num_wrong_pred += round(abs(Y_test.iloc[i] - 1.0))
                                    elif item[i_dataset] < 0.5:
                                        normalized_pred_out[i_dataset][i] = 0
                                        num_wrong_pred += round(abs(Y_test.iloc[i] - 0.0))
                        elif len(datasetNameList) == 1:
                            for i, item in enumerate(pred_out):
                                if item >= 0.5:
                                    normalized_pred_out.append(1)
                                    num_wrong_pred += round(abs(Y_test.iloc[i] - 1.0))
                                elif item < 0.5:
                                    normalized_pred_out.append(0)
                                    num_wrong_pred += round(abs(Y_test.iloc[i] - 0.0))

                    print("normalized pred_out=")
                    print(normalized_pred_out)
                    print("test label is")
                    print(Y_test)
                    print("num_wrong_pred=%d, total test num=%d,accuracy=%f" % (
                    num_wrong_pred, len(Y_test), 1.0 - num_wrong_pred / len(Y_test)))

                    # out = out.view(out.size(0), -1)
                    data_test_pred = pred_out  # .numpy()
                    normalized_data_test_pred = normalized_pred_out

            elif framework == 'pytorch':
                if separatelyTrainAE_NN:
                    pass
                else:
                    if multiDatasetMode=='multi-task':

                        gene_data_test = np.array(gene_data_test)
                        hidden_size = 15
                        print("gene_data_test.shape")
                        print(gene_data_test.shape)
                        '''
                        model_ae = AE.MeiNN(config, path, date, code, gene_data_test.T, Y_test.T, platform, model_type, data_type,
                                    HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=AE_epoch_from_main,
                                    NN_epoch_from_main=NN_epoch_from_main, separatelyTrainAE_NN=separatelyTrainAE_NN,model_dir='./results/models',
                                    gene_to_residue_or_pathway_info=my_gene_to_residue_info,toAddGeneSite=toAddGeneSite,
                                    toAddGenePathway=toAddGenePathway,
                                    multiDatasetMode=multiDatasetMode,datasetNameList=datasetNameList,lossMode=lossMode)
                        '''
                        model_ae = torch.load(path + date + '.pth')
                        # model_ae.load_state_dict(torch.load(path+date + '.tar'), strict=False)
                        # model=AE.Autoencoder(in_dim=gene_data_test.shape[1], h_dim=hidden_size)
                        '''
                        model_nn = torch.load(
                            date + '_fully-connected-network.pth')  # load network from parameters saved in network.pth @ 22-2-18
                        '''  # 2022-7 commented

                        # images = AE.to_var(gene_data_test.T.view(gene_data_test.T.size(0), -1))
                        # images = images.float()
                        gene_data_test = torch.from_numpy(gene_data_test)
                        gene_data_test = AE.to_var(gene_data_test.view(gene_data_test.size(0), -1))
                        gene_data_test = gene_data_test.float()
                        _, _, embedding = model_ae(gene_data_test.T)

                        print("predicting:after ae, embedding is ")
                        # print(embedding)
                        print(embedding.shape)
                        print("len(datasetNameList)")
                        print(len(datasetNameList))
                        if len(datasetNameList) > 1 and len(datasetNameList) == 6:
                            #ae_out, [pred_out1,pred_out2,pred_out3,pred_out4,pred_out5,pred_out6], _ = model_ae(gene_data_test.T)
                            ae_out, pred_out_list, _ = model_ae(
                                gene_data_test.T)
                            #[pred_out1, pred_out2, pred_out3, pred_out4, pred_out5, pred_out6]=pred_out_list
                            #for i in range(len(datasetNameList)):
                            print("prediction list is")
                            print(pred_out_list)
                            '''
                            print("prediction%d is" % 1)

                            print(pred_out1.shape)
                            print("prediction%d is" % 2)
                            print(pred_out2.shape)
                            print("prediction%d is" % 3)
                            print(pred_out3.shape)
                            print("prediction%d is" % 4)
                            print(pred_out4.shape)
                            print("prediction%d is" % 5)
                            print(pred_out5.shape)
                            print("prediction%d is" % 6)
                            print(pred_out6.shape)
                            '''
                            # data_test_pred = [pred_out1,pred_out2,pred_out3,pred_out4,pred_out5,pred_out6]
                            # data_test_pred = pred_out
                            pred_out_list=torch.Tensor([item.cpu().detach().numpy() for item in pred_out_list]).squeeze().T
                            
                            data_test_pred = pd.DataFrame(pred_out_list.detach().numpy())
                            data_test_pred.to_csv(path + date + "_" + code +"separateAE-NN=" +
                                    str(separatelyTrainAE_NN) + "pred_list.txt", sep='\t')
                            '''
                            data_test_pred = pd.DataFrame(pred_out1.detach().numpy())
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred1).txt", sep='\t')
                            data_test_pred = pd.DataFrame(pred_out2.detach().numpy())
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred2).txt", sep='\t')
                            data_test_pred = pd.DataFrame(pred_out3.detach().numpy())
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred3).txt", sep='\t')
                            data_test_pred = pd.DataFrame(pred_out4.detach().numpy())
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred4).txt", sep='\t')
                            data_test_pred = pd.DataFrame(pred_out5.detach().numpy())
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred5).txt", sep='\t')
                            data_test_pred = pd.DataFrame(pred_out6.detach().numpy())
                            data_test_pred.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "separateAE-NN=" +
                                str(separatelyTrainAE_NN) + "pred6).txt", sep='\t')
                            '''
                            data_test_ae_out = pd.DataFrame(ae_out.detach().numpy())
                            data_test_ae_out.to_csv(
                                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "AE_output).txt",
                                sep='\t')
                    else:
                        gene_data_test = np.array(gene_data_test)
                        hidden_size = 15
                        print("gene_data_test.shape")
                        print(gene_data_test.shape)
                        '''
                        model_ae = AE.MeiNN(config, path, date, code, gene_data_test.T, Y_test.T, platform, model_type, data_type,
                                    HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=AE_epoch_from_main,
                                    NN_epoch_from_main=NN_epoch_from_main, separatelyTrainAE_NN=separatelyTrainAE_NN,model_dir='./results/models',
                                    gene_to_residue_or_pathway_info=my_gene_to_residue_info,toAddGeneSite=toAddGeneSite,
                                    toAddGenePathway=toAddGenePathway,
                                    multiDatasetMode=multiDatasetMode,datasetNameList=datasetNameList,lossMode=lossMode)
                        '''
                        model_ae=torch.load(path+date + '.pth')
                        # model_ae.load_state_dict(torch.load(path+date + '.tar'), strict=False)
                        # model=AE.Autoencoder(in_dim=gene_data_test.shape[1], h_dim=hidden_size)
                        '''
                        model_nn = torch.load(
                            date + '_fully-connected-network.pth')  # load network from parameters saved in network.pth @ 22-2-18
                        '''#2022-7 commented

                        # images = AE.to_var(gene_data_test.T.view(gene_data_test.T.size(0), -1))
                        # images = images.float()
                        gene_data_test = torch.from_numpy(gene_data_test)
                        gene_data_test = AE.to_var(gene_data_test.view(gene_data_test.size(0), -1))
                        gene_data_test = gene_data_test.float()
                        _,_,embedding = model_ae(gene_data_test.T)

                        print("predicting:after ae, embedding is ")
                        #print(embedding)
                        print(embedding.shape)

                        out,prediction,_ = model_ae(gene_data_test.T)
                        print("prediction is")
                        print(prediction)
                        prediction = prediction.view(out.size(0), -1)
                        data_test_pred = prediction.detach().numpy()
                        print("after to numpy is")
                        print(data_test_pred)
                        data_test_pred = pd.DataFrame(np.array(data_test_pred))
                        data_test_pred.to_csv(
                            path+date + "prediction.txt", sep='\t')
                        # print('Now predicting ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(int(count * 100 / num)) + '% ...')

                        '''if count == 1:
                            data_test_pred = pred2.T
                        else:
                            print("data_test_pred")
                            print(data_test_pred)
                            print("pred2.T")
                            print(pred2.T)
                            data_test_pred = np.vstack([data_test_pred, pred2.T])'''
                        print('finish!')





            '''#7-9
            print("predicting:after ae, embedding is ")
            print(embedding)
            print(embedding.shape)
            '''

            '''
            normalized_data_test_pred = pd.DataFrame(np.array(normalized_data_test_pred))
            normalized_data_test_pred.to_csv(
                path + date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "normalized_pred).txt",
                sep='\t')
            '''


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
