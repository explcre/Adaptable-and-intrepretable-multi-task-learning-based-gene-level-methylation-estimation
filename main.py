import pandas as pd
import numpy as np
from train import run
from predict import predict
from test import select_feature
import torch
torch.set_printoptions(profile="full")
#from torchsummary import summary
code = "GSE66695"
platform = "platform.json"
model_type = "AE"#"RandomForest"
predict_model_type="L2"
data_type = "origin_data"
dataset_type="train"
isTrain=True
toTrainAE=True
toTrainNN=True
isPredict=True
model=None
AE_epoch=100
NN_epoch=100
ae=None
fcn=None
date='3-6-kerasAE-regular-try7-lr-epoch%d'%AE_epoch

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

print(np.random.randint(0,2,(100,200)))
# train
if isTrain:
    train_data = pd.read_table("data_train.txt", index_col=0)
    train_label = pd.read_table("label_train.txt", index_col=0).values.ravel()
    (ae,fcn)=run(date,code, train_data, train_label, platform, model_type, data_type,79*5,toTrainAE,toTrainNN,AE_epoch,NN_epoch)
    ae.summary()
    fcn.summary()


# predict

if isPredict:
    test_data = pd.read_table("data_test.txt", index_col=0)
    predict(date,code, test_data, platform, date+"_"+code +"_"+model_type+"_"+data_type+dataset_type+"_model.pickle", model_type, data_type,model,predict_model_type)

# test(feature selection)
data = pd.read_table(date+"_"+code+"_gene_level("+data_type+"_"+model_type+").txt", index_col=0)
label = pd.read_table("label_test.txt", index_col=0).values.ravel()
select_feature(code, data, label, gene=True)
