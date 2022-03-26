#-*- coding : utf-8-*-
# coding:unicode_escape
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from train_keras import run
from predict_keras import predict
#from test import select_feature
import torch
torch.set_printoptions(profile="full")
#from torchsummary import summary
code = "GSE66695-series"#GSE42861_processed_methylation_matrix
platform = "platform.json"
model_type = "AE"#"RandomForest"
predict_model_type="L2"
data_type = "origin_data"
dataset_type="train"
isTrain=True
toTrainAE=True
toTrainNN=True
isPredict=False
model=None
AE_epoch=100
NN_epoch=100
ae=None
fcn=None
h_dim=13
date='3-25-kerasAE-regular-h_dim=%d-lr-epoch%d'%(h_dim,AE_epoch)
keras=True
path = r"./result/"
train_dataset_filename=r"./dataset/GSE66695_series_matrix.xlsx"#r"./dataset/data_train.txt"#GSE42861_processed_methylation_matrix.txt

train_label_filename=r"./dataset/label_train.txt"
test_dataset_filename=r"./dataset/data_test.txt"
test_label_filename=r"./dataset/label_test.txt"
just_check_data=True
'''
def print_model_summary_pytorch():
    print('###############################################################')
    file = open(date + "ae_detail.csv", mode='w', encoding='utf-8')
    model_ae=torch.load(date+'_auto-encoder.pth')
    summary(model_ae,input_size=(0,809))#, input_size=(3, 512, 512)
    #file.write(summary(model_ae,input_size=(0,809)))
    print(model_ae)
    for name,parameters in model_ae.named_parameters():
        print(name+':'+str(parameters.size()))
        print(parameters)

        file.write(name+':'+str(parameters.size()))
        file.write(str(parameters))
    print('###############################################################')
'''

# train
if isTrain:
    train_data = pd.read_excel(train_dataset_filename,skiprows=30)#, index_col=0,names=['0','1']#,delimiter='!|\t'
    #train_data['0'].str.split('\t', expand=True)
    print("read train_data.shape:")
    print(train_data.shape)
    print(train_data[0:15])
    train_data.head(10)
    print("finish read train data")
    train_data_splited,test_data_splited=train_test_split(train_data, train_size=0.8, random_state=10)
    print("train_data_splited.shape:")
    print(train_data_splited.shape)
    print(train_data_splited[0:10])
    print("test_data_splited.shape:")
    print(test_data_splited.shape)
    print(test_data_splited[0:10])
    train_label = pd.read_table(train_label_filename, index_col=0).values.ravel()
    print("finish read train label")
    print(train_data.head(10))
    if(not just_check_data):
        if keras:
            (ae,fcn)=run(path,date,code, train_data, train_label, platform, model_type, data_type,h_dim,toTrainAE,toTrainNN,AE_epoch,NN_epoch)
            ae.summary()
            fcn.summary()
        else:
            run(path,date, code, train_data, train_label, platform, model_type, data_type, h_dim, toTrainAE,toTrainNN, AE_epoch, NN_epoch)
'''
if keras:
    ae.summary()
    fcn.summary()
'''

# predict
if isPredict and (not just_check_data):
    test_data = pd.read_table(test_dataset_filename, index_col=0)
    predict(path,date,code, test_data, platform, date+"_"+code +"_"+model_type+"_"+data_type+dataset_type+"_model.pickle", model_type, data_type,model,predict_model_type)


'''
# test(feature selection)
data = pd.read_table(r"./dataset/"+date+"_"+code+"_gene_level("+data_type+"_"+model_type+").txt", index_col=0)
label = pd.read_table(test_label_filename, index_col=0).values.ravel()
'''
#select_feature(code, data, label, gene=True)
