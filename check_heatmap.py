# -*- coding : utf-8-*-
# coding:unicode_escape
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

framework='keras'
if framework=='keras':
    from train_keras_redefined_loss import run
    from predict_keras_redefined_loss import predict
elif framework=='pytorch':
    from train_pytorch import run
    from predict_pytorch import predict

# from test import select_feature
import torch
import seaborn as sns
import matplotlib.pylab as plt

torch.set_printoptions(profile="full")
# from torchsummary import summary

platform = "platform.json"
model_type = "AE"  # "RandomForest"
predict_model_type = "L2"
data_type = "origin_data"
dataset_type = "train"
isTrain = True
toTrainAE = True
toTrainNN = True
isPredict = True
toTrainMeiNN = True
toAddGeneSite = False
toAddGenePathway = False
just_check_data = False

onlyGetPredictionFromLocalAndCheckAccuracy = False
lossMode = 'reg_mean'
# reg_mean: we set loss as mean of regularization+prediction loss
# auto_scale:
# no:no mode
selectNumResidueMode = 'num'
# num:define num of selected residue
# pvalue:define a threshold of pvalue
# min: index will be minimum of 1,num_of_selected and 2.(last index pvalue which < pvalueThreshold)
pvalueThreshold = 1e-5
num_of_selected_residue = 25
selectNumPathwayMode = 'equal_difference'  # '=num_gene'
# =num_gene: equal number of gene selected
# 'equal_difference' make pathway-gene-residue an arithmetic sequence
# num : give a value
num_of_selected_pathway = num_of_selected_residue / 2
isMultiDataset = True
multiDatasetMode = 'softmax'
# softmax: multi-class, with last layer of MeiNN is softmax
# multi-task: multi-task solution with network architecture for each task
datasetNameList = ['diabetes1', 'IBD', 'MS', 'Psoriasis', 'RA',
                   'SLE']  # "diabetes1","RA","Psoriasis"]#,"RA","Psoriasis"]#,"Psoriasis","IBD"]# ['diabetes1','Psoriasis','SLE']
model = None
AE_epoch = 100  # *len(datasetNameList)
NN_epoch = 100  # *len(datasetNameList)
separatelyTrainAE_NN = False
toAddSkipConnection = False
ae = None
fcn = None
myMeiNN = None

code = ''
for i in datasetNameList:
    code += (i + ' ')  # "GSE66695"#GSE42861_processed_methylation_matrix #"GSE66695-series"
num_of_selected_residue_list = [2000, 2000, 2000]
h_dim = 60 * len(datasetNameList)
date = '6-9%s-Aep%d-Nep%d-Site%sPath%s-res%d-lMod-%s-sep%s-multi%s-pMod%s' % (
    (len(datasetNameList) > 1), AE_epoch, NN_epoch, toAddGeneSite, toAddGenePathway, num_of_selected_residue, lossMode,
    separatelyTrainAE_NN, multiDatasetMode, selectNumPathwayMode)
keras = True
path = r"./result/"
selected_residue_name_list = set()

toCheckHeatMap=True
if toCheckHeatMap:

    gene_to_residue_map=np.load(
        path + date + "_" + code + "_gene_level" + "gene2residue_map)" + ".txt.npy")
    gene_pathway_needed=pd.read_csv(
        path + date + "_" + code + "_gene_level" + "gene_pathway_needed)" + ".csv",sep='\t')
    print("gene_pathway_needed")
    print(gene_pathway_needed.shape)
    print(gene_pathway_needed)
    print("gene_to_residue_map")
    print(gene_to_residue_map.shape)
    print(gene_to_residue_map)

    heat_map_gene_pathway_kown = sns.heatmap(gene_pathway_needed.iloc[:,2:], linewidth=1, annot=False)
    plt.title(path + date + 'multi-task-MeiNN gene-pathway known info HeatMap')
    plt.savefig(path + date + 'multi-task-MeiNN_gene_pathway_known_info_heatmap.png')
    plt.show()

    heat_map_gene_residue_known = sns.heatmap(gene_to_residue_map, linewidth=1, annot=False)
    plt.title(path + date + 'multi-task-MeiNN gene-residue known info HeatMap')
    plt.savefig(path + date + 'multi-task-MeiNN_gene_residue_known_info_heatmap.png')
    plt.show()

    from keras import backend as K, losses
    from keras.models import load_model
    def relu_advanced(x):
        return K.relu(x, threshold=0)
    def myLoss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred)
    loaded_fcn_multitask=load_model(path + date + 'multi-task-MeiNN.h5'
                                            ,custom_objects={'relu_advanced': relu_advanced, 'myLoss': myLoss})
    weight = loaded_fcn_multitask.get_weights()
    print("loaded_fcn_multitask")
    print(loaded_fcn_multitask.summary())

    for i in range(len(weight)):
        print("%d-th weight"%i)
        print(weight[i].shape)
        print(weight[i])


    plt.figure(figsize=(10,10))

    layer_gene_pathway=12
    print("weight[layer_gene_pathway]")
    print(weight[layer_gene_pathway].shape)
    print(weight[layer_gene_pathway])
    heat_map_gene_pathway = sns.heatmap( weight[layer_gene_pathway], linewidth = 1 , annot = False)
    plt.title(path + date + 'multi-task-MeiNN gene-pathway HeatMap' )
    plt.savefig(path + date + 'multi-task-MeiNN_gene_pathway_heatmap.png')
    plt.show()

    heat_map_gene_pathway_clustered = sns.clustermap( weight[layer_gene_pathway],row_cluster=True,standard_scale=1)
    plt.title(path + date + 'multi-task-MeiNN gene-pathway row-clustered cluster Map' )
    plt.savefig(path + date + 'multi-task-MeiNN_gene_pathway_row-clustered_cluster_map.png')
    plt.show()

    layer_gene_site=15
    print("weight[layer_gene_site]")
    print(weight[layer_gene_site].shape)
    print(weight[layer_gene_site])
    heat_map_gene_site = sns.heatmap( list(weight[layer_gene_site]), linewidth = 1 , annot = False)
    plt.title(path + date + 'multi-task-MeiNN gene-site HeatMap' )
    plt.savefig(path + date + 'multi-task-MeiNN_gene_site_heatmap.png')
    plt.show()