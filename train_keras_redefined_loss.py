# data_train.py
import re
# from resVAE.resvae import resVAE
# import resVAE.utils as cutils
# from resVAE.config import config
# import resVAE.reporting as report
import torchvision

from MeiNN.MeiNN import MeiNN, gene_to_residue_or_pathway_info
from MeiNN.MeiNN_pytorch import MeiNN_pytorch
from MeiNN.config import config
import os
import json
import numpy as np
import pandas as pd
import csv  # 调用数据保存文件
import pickle
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# import TabularAutoEncoder
# import VAE
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()#newly-added-3-27

import torch
from torch import nn
# import torchvision
from torch.autograd import Variable
# import AutoEncoder
import math
import warnings
import AutoEncoder as AE

from time import time

# import tensorflow.keras as keras

from keras import layers

# from keras import objectives
from keras import losses
from keras import regularizers
from keras import backend as K
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
from keras.models import load_model

warnings.filterwarnings("ignore")


def mkdir(path):
    import os
    # remove first blank space
    path = path.strip()
    # remove \ at the end
    path = path.rstrip("\\")
    # judge whether directory exists
    # exist     True
    # not exist   False
    isExists = os.path.exists(path)
    # judge the result
    if not isExists:
        # if not exist, then create directory
        os.makedirs(path)
        print(path + " directory created successfully.")
        return True
    else:
        # if directory exists, don't create and print it already exists
        print(path + " directory already exists.")
        return False


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


# Only train regression model, save parameters to pickle file
def run(path, date, code, X_train, y_train, platform, model_type, data_type, HIDDEN_DIMENSION, toTrainMeiNN,
        toAddGenePathway=False,toAddGeneSite=False,multiDatasetMode='multi-task',datasetNameList=[],
        num_of_selected_residue=1000,lossMode='reg_mean',selectNumPathwayMode = '=num_gene',
        num_of_selected_pathway = 500,
        AE_epoch_from_main=1000, NN_epoch_from_main=1000, separatelyTrainAE_NN=True,gene_pathway_dir="./dataset/GO term pathway/matrix.csv",
        pathway_name_dir="./dataset/GO term pathway/gene_set.txt",
        gene_name_dir="./dataset/GO term pathway/genes.txt",
        framework='keras'):
    data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                 'radical_data': radical_data, 'cube_data': cube_data}
    model_dict = {'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression,
                  'L1': Lasso, 'L2': Ridge, 'RandomForest': RandomForestRegressor, 'AE': AE.Autoencoder}

    if toAddGenePathway:
        # the following added 22-4-24 for go term pathway
        gene_pathway_csv_data = pd.read_csv(gene_pathway_dir, header=None, dtype='int')  # 防止弹出警告
        pathway_name_data = pd.read_csv(pathway_name_dir,header=None)#, dtype='str', sep=',')#, header=0, dtype='str')
        # pathway_name_data_df=pathway_name_data.values.tolist()
        gene_name_data = pd.read_csv(gene_name_dir,header=None)#, dtype='str', sep=',')#, header=0, dtype='str', sep=',')
        # gene_name_data_df = gene_name_data.values.tolist()
        gene_pathway_df = pd.DataFrame(gene_pathway_csv_data)  # , columns=gene_name_data, index=pathway_name_data)
        print("INFO: gene_pathway_df=")
        print(gene_pathway_df)
        print("INFO : pathway_name_data")
        print(pathway_name_data)
        print("INFO : gene_name_data")
        print(gene_name_data)
        gene_pathway_df.index=pathway_name_data[0].values.tolist()
        gene_pathway_df.columns=gene_name_data[0].values.tolist()#.index.values.tolist()
        print("INFO: gene_pathway_df after adding column and index")
        print(gene_pathway_df)
        #print("INFO: gene_pathway_df.index")
        #print(gene_pathway_df.index.values.tolist())
        print("gene_pathway_df.loc['GOBP_MITOCHONDRIAL_GENOME_MAINTENANCE']")
        print(gene_pathway_df.loc['GOBP_MITOCHONDRIAL_GENOME_MAINTENANCE'])
        # genename_to_genepathway_index_map=
        # gene_pathway_df.rename(columns=gene_name_data, index=pathway_name_data)
        # print(gene_pathway_df.head(10))
        # above added 22-4-24 for go term pathway

    with open(platform, 'r') as f:
        gene_dict = json.load(f)
        f.close()

    count = 0
    num = len(gene_dict)
    gene_list = []
    print('Now start training gene...')

    data_train = data_dict[data_type](X_train)
    print("data_train:")
    print(data_train)
    # print("gene_dict:")
    # print(gene_dict)
    gene_data_train = []
    residuals_name = []
    model = None
    count_gene = 0
    count_residue = 0
    gene_to_id_map = {}
    residue_to_id_map = {}
    gene_present_list = set()
    mode_all_gene_and_residue = False
    '''
    data_train_df=pd.DataFrame(data_train)
    print("data_train_df=")
    print(data_train_df)
    print("y_train")
    print(y_train)
    if code=="GSE66695":
        data_label_df0=pd.DataFrame(y_train,columns=['Ground Truth'],index=data_train_df.columns)
    else:
        data_label_df0 = pd.DataFrame(y_train,columns=['Ground Truth'])
    data_label_df=data_label_df0.T
    print("data_label_df=")
    print(data_label_df)
    data_train_label_df=data_train_df.append(data_label_df)#pd.concat([data_train_df, data_label_df], axis=0)
    print("after join data and label")
    print(data_train_label_df)
    from scipy import stats
    data_train_label_df_T=data_train_label_df.T
    print("data_train_label_df_T[data_train_label_df_T['Ground Truth']==1.0]")
    print(data_train_label_df_T[data_train_label_df_T['Ground Truth']==1.0])
    t_test_result=stats.ttest_ind(data_train_label_df_T[data_train_label_df_T['Ground Truth']==1.0], data_train_label_df_T[data_train_label_df_T['Ground Truth']==0.0])
    print("t_testresult=")
    print(t_test_result)
    print("t_testresult.pvalue=")
    print(t_test_result.pvalue)
    print("t_testresult.pvalue.shape=")
    print(t_test_result.pvalue.shape)

    data_train_label_df['pvalue']=t_test_result.pvalue
    print("data_train_label_df added pvalue")
    print(data_train_label_df)
    print("t_testresult.pvalue.sort()=")
    print(np.sort(t_test_result.pvalue))
    print("data_train_label_df.sort_values(by='pvalue',ascending=True)")
    data_train_label_df_sorted_by_pvalue=data_train_label_df.sort_values(by='pvalue', ascending=True)
    print(data_train_label_df_sorted_by_pvalue)
    print("data_train_label_df_sorted_by_pvalue.iloc[1:,:-1])")
    data_train_label_df_sorted_by_pvalue_raw=data_train_label_df_sorted_by_pvalue.iloc[1:, :-1]
    print(data_train_label_df_sorted_by_pvalue_raw)

    selected_residue_train_data=data_train_label_df_sorted_by_pvalue_raw.iloc[:num_of_selected_residue,:]
    print("selected_residue_train_data)")
    print(selected_residue_train_data)
    data_train=selected_residue_train_data
    '''
    #data_train_label_df.sort_values(by='pvalue',ascending=True)
    #t_test_result.pvalue.sort()
    toPrintInfo=False
    for (i, gene) in enumerate(gene_dict):
        count += 1
        if toPrintInfo:
            print("%s-th,gene=%s,gene_dict[gene]=%s" % (i, gene, gene_dict[gene]))
        # gene_data_train = []
        # residuals_name = []

        # following added 22-4-14
        if mode_all_gene_and_residue:
            gene_to_id_map[gene] = count_gene
            count_gene += 1
            for residue in gene_dict[gene]:
                # gene_to_residue_map[gene_to_id_map[gene]][residue_to_id_map[residue]] = 1  # added 22-4-14
                residue_to_id_map[residue] = count_residue  # added 22-4-14
                count_residue += 1  # added 22-4-14
        # above added 22-4-14

        for residue in data_train.index:
            if residue in gene_dict[gene]:
                if (residue not in (residuals_name)):  # added 2022-4-14
                    residuals_name.append(residue)
                    gene_data_train.append(data_train.loc[residue])
                # following added 22-4-14
                if not mode_all_gene_and_residue:
                    if gene not in gene_to_id_map:
                        gene_to_id_map[gene] = count_gene
                        count_gene += 1
                        gene_present_list.add(gene)
                        # gene_to_residue_map.append([])
                    if residue not in residue_to_id_map:
                        residue_to_id_map[residue] = count_residue  # added 22-4-14
                        count_residue += 1  # added 22-4-14
                        # gene_to_residue_map.append(1)

                # above added 22-4-14
        if len(gene_data_train) == 0:
            # print('Contained Nan data, has been removed!')
            continue

        # gene_data_train = np.array(gene_data_train)
        if gene not in gene_list:
            gene_list.append(gene)

        if toPrintInfo:
            print('No.' + str(i) + 'Now training ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(
                int(count * 100 / num)) + '% ...')
        # print("gene_data_train.shape[1]")
        # print(np.array(gene_data_train).shape[1])

        if count == 1:
            with open(path + date + "_" + code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'wb') as f:
                pickle.dump((gene, model), f)
        else:
            with open(path + date + "_" + code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'ab') as f:
                pickle.dump((gene, model), f)
        if toPrintInfo:
            print('finish!')

    #############2021-5-21##############

    if toAddGenePathway:
        gene_present_list_df = pd.DataFrame(list(gene_present_list), columns=['name'])
        gene_present_set=set(gene_present_list)
        where_input_gene_is_not_in_go_term_set=gene_present_set.difference(set(gene_pathway_df.columns.values.tolist()))
        print("where_input_gene_is_not_in_go_term_set")
        print(where_input_gene_is_not_in_go_term_set)
        gene_pathway_df_with_input_gene=gene_pathway_df.loc[gene_pathway_df.apply(np.sum, axis=1) > 0 ]
        gene_pathway_df_with_input_gene[list(where_input_gene_is_not_in_go_term_set)]=np.zeros((gene_pathway_df_with_input_gene.shape[0],len(where_input_gene_is_not_in_go_term_set)), dtype=np.int)
        print("gene_pathway_df_with_input_gene")
        print(gene_pathway_df_with_input_gene)
        print("selected present gene from go term:")
        gene_pathway_df_with_only_present_gene=gene_pathway_df_with_input_gene[gene_present_list]
        print(gene_pathway_df_with_only_present_gene)
        gene_pathway_needed=gene_pathway_df_with_only_present_gene.loc[gene_pathway_df_with_only_present_gene.apply(np.sum, axis=1) > 0 ]
        print(" remove rows that are all 0,gene_pathway_needed")
        print(gene_pathway_needed)
        gene_pathway_needed[list(where_input_gene_is_not_in_go_term_set)] = np.ones(
            (gene_pathway_needed.shape[0], len(where_input_gene_is_not_in_go_term_set)), dtype=np.int)
        print(" remove rows that are all 0,gene_pathway_needed,add never exist input gene")
        print(gene_pathway_needed)
        gene_pathway_needed['gene-pathway sum']=gene_pathway_needed.apply(lambda x:sum(x),axis=1)
        print(" remove rows that are all 0,gene_pathway_needed,add never exist input gene,with connection sum")
        print(gene_pathway_needed)
        gene_pathway_needed.sort_values(by='gene-pathway sum',ascending=False)
        print(" remove rows that are all 0,gene_pathway_needed,add never exist input gene,sorted by connection sum")
        print(gene_pathway_needed)
        if selectNumPathwayMode=='=num_gene':
            selected_pathway_num=gene_pathway_needed.shape[1]
        elif selectNumPathwayMode=='equal_difference':
            selected_pathway_num=count_gene-(count_residue-count_gene)
        elif selectNumPathwayMode=='equal_difference':
            selected_pathway_num=num_of_selected_pathway
        gene_pathway_needed=gene_pathway_needed.iloc[:selected_pathway_num-1,:-1]
        print(" remove rows that are all 0,gene_pathway_needed,add never exist input gene,sorted by connection sum,finally selected certain pathway:")
        print(gene_pathway_needed)
        gene_pathway_needed_reversed=gene_pathway_needed.replace([1,0],[0,1]).values.tolist()
        print("gene_pathway_needed_reversed:")
        print(gene_pathway_needed.replace([1,0],[0,1]))
        gene_pathway_needed.to_csv(
            path + date + "_" + code + "_gene_level" + "gene_pathway_needed).csv", sep='\t')
        import seaborn as sns
        import matplotlib.pylab as plt
        heat_map_gene_pathway = sns.heatmap( gene_pathway_needed, linewidth=1, annot=False)
        plt.title(path + date + 'multi-task-MeiNN gene-pathway known info HeatMap')
        plt.savefig(path + date + 'multi-task-MeiNN_gene_pathway_known_info_heatmap.png')
        #plt.show()
    ####################################
    if toAddGenePathway and False:
        gene_present_list_df = pd.DataFrame(list(gene_present_list), columns=['name'])
        temp_list = list(range(len(gene_name_data)))

        # for i, val in enumerate(temp_list):
        #    temp_list[i] = str(val)
        gene_present_index = gene_present_list_df.replace(gene_name_data.values.tolist(), temp_list)
        # gene_present_index_sorted=gene_present_index.sort_values('name')
        gene_present_index_list = gene_present_index.values.tolist()
        print("**********gene_present_index_list********")
        print(gene_present_index_list)
        print(len(gene_present_index_list))
        import re
        # for i, val in enumerate(gene_present_index_list):
        # gene_present_index_list[i] = str(val)
        # re.search(re_exp)
        re_exp = r"[^0-9\]\[]"
        where_input_gene_is_not_in_go_term = [False] * len(gene_present_index_list)
        for i, val in enumerate(gene_present_index_list):
            where_input_gene_is_not_in_go_term[i] = re.match(re_exp, str(val))
        # where_input_gene_is_not_in_go_term = list(filter(lambda x: re.match(re_exp, x) != None, gene_present_index_list))
        # where_input_gene_is_not_in_go_term = gene_present_index_list.str.contains(re_exp)
        print("**********where_input_gene_is_not_in_go_term********")
        print(where_input_gene_is_not_in_go_term)
        print(len(where_input_gene_is_not_in_go_term))
        gene_name_data_list = gene_name_data.values.tolist()
        print("**gene_name_data_list:**")
        print(gene_name_data_list)
        print(gene_name_data_list[0])
        print("**gene_to_id_map:**")
        print(gene_to_id_map)
        # print(gene_to_id_map['ABCDEFG'])
        print("**********in gene to id map but not in go term********")
        count1 = 0
        for gene in gene_present_list:
            if str(gene) not in gene_name_data_list:
                print(gene)
                print(count1)
                count1 += 1
        gene_pathway_present_gene_index = gene_pathway_csv_data.loc[:gene_present_index_list]
    # save the dictionary : following added 22-4-14



    np.save(
        path + date + "_" + code + "_gene_level" + "_original_residue_name_list)" + ".txt",
        residuals_name)#added 5-12
    np.save(
        path + date + "_" + code + "_gene_level" + "_original_gene_to_id_map)" + ".txt",
        gene_to_id_map)
    np.save(
        path + date + "_" + code + "_gene_level" + "_original_residue_to_id_map)" + ".txt",
        residue_to_id_map)

    print("len residue_to_id_map%d"% len(residue_to_id_map))
    print("len gene_to_id_map%d" % len(gene_to_id_map))
    gene_to_residue_map = [[0 for i in range(len(residue_to_id_map))] for i in range(len(gene_to_id_map))]
    gene_to_residue_map_reversed = [[1 for i in range(len(residue_to_id_map))] for i in range(len(gene_to_id_map))]
    count_connection = 0
    if toAddGeneSite:
        for id in gene_to_id_map:
            if (id in gene_dict):
                for residue in gene_dict[id]:
                    if residue in residue_to_id_map:
                        gene_to_residue_map[gene_to_id_map[str(id)]][residue_to_id_map[residue]] = 1
                        gene_to_residue_map_reversed[gene_to_id_map[str(id)]][residue_to_id_map[residue]] = 0
                        count_connection += 1

    np.save(
        path + date + "_" + code + "_gene_level" +  "gene2residue_map)" + ".txt",
        gene_to_residue_map)

    heat_map_gene_residue= sns.heatmap(gene_to_residue_map, linewidth=1, annot=False)
    plt.title(path + date + 'multi-task-MeiNN gene-residue known info HeatMap')
    plt.savefig(path + date + 'multi-task-MeiNN_gene_residue_known_info_heatmap.png')
    #plt.show()
    # above added 22-4-14
    #gene_data_train_Tensor=gene_data_train
    gene_data_train = np.array(gene_data_train)  # added line on 2-3
    gene_data_train_Tensor=torch.from_numpy(gene_data_train).float()
    print("gene_data_train=")
    print(gene_data_train)
    np.save(
        path + date + "_" + code +  "gene_data_train)" + ".txt",
        gene_data_train)
    # ae=None
    autoencoder = None
    fcn = None
    if True or (model_type == "VAE" or model_type == "AE" or model_type == "MeiNN"):
        # encoding_dim = 400
        latent_dim = HIDDEN_DIMENSION
        print("DEBUG INFO:we entered MeiNN code")
        if True or toTrainMeiNN:
            if framework=='keras':
                print("DEBUG INFO:we entered to train MeiNN keras code")
                gene_data_train=np.load(path + date + "_" + code + "gene_data_train)" + ".txt.npy")
                my_gene_to_residue_info = gene_to_residue_or_pathway_info(gene_to_id_map, residue_to_id_map, gene_to_residue_map,
                                                           count_connection, gene_to_residue_map_reversed,gene_pathway_needed,gene_pathway_needed_reversed)
                myMeiNN = MeiNN(config, path, date, code, gene_data_train.T, y_train.T, platform, model_type, data_type,
                            HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=AE_epoch_from_main,
                            NN_epoch_from_main=NN_epoch_from_main, separatelyTrainAE_NN=separatelyTrainAE_NN,model_dir='./results/models',
                            gene_to_residue_or_pathway_info=my_gene_to_residue_info,toAddGeneSite=toAddGeneSite,
                            toAddGenePathway=toAddGenePathway,
                            multiDatasetMode=multiDatasetMode,datasetNameList=datasetNameList,lossMode=lossMode)

                # myMeiNN.build()
                myMeiNN.compile()
                # myMeiNN.fcn.fit(gene_data_train.T, y_train.T, epochs=NN_epoch_from_main, batch_size=79, shuffle=True)
                myMeiNN.fit()
            elif framework=='pytorch':
                myMeiNN=None
                print("DEBUG INFO:we entered to train MeiNN pytorch code")
                gene_data_train = np.load(path + date + "_" + code + "gene_data_train)" + ".txt.npy")
                my_gene_to_residue_info = gene_to_residue_or_pathway_info(gene_to_id_map, residue_to_id_map,
                                                                          gene_to_residue_map,
                                                                          count_connection,
                                                                          gene_to_residue_map_reversed,
                                                                          gene_pathway_needed,
                                                                          gene_pathway_needed_reversed)
                ##########added from train_pytorch.py############
                num_epochs = AE_epoch_from_main
                batch_size = int(gene_data_train.shape[1])  # gene_data_train.shape[0]#100#809
                #hidden_size = 10
                dataset = gene_data_train.T  # .flatten()#gene_data_train.view(gene_data_train.size[0], -1)
                # dataset = gene_data_train  # dsets.MNIST(root='../data',

                # train=True,
                # transform=transforms.ToTensor(),
                # download=True)
                data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                          batch_size=batch_size,
                                                          shuffle=True)
                print("gene_data_train.shape")
                print(gene_data_train.shape)
                print("dataset.shape")
                print(dataset.shape)

                ae=AE.MeiNN(config, path, date, code, gene_data_train.T, y_train.T, platform, model_type, data_type,
                            HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=AE_epoch_from_main,
                            NN_epoch_from_main=NN_epoch_from_main, separatelyTrainAE_NN=separatelyTrainAE_NN,model_dir='./results/models',
                            gene_to_residue_or_pathway_info=my_gene_to_residue_info,toAddGeneSite=toAddGeneSite,
                            toAddGenePathway=toAddGenePathway,
                            multiDatasetMode=multiDatasetMode,datasetNameList=datasetNameList,lossMode=lossMode)

                #ae = AE.Autoencoder(in_dim=gene_data_train.shape[0],
                #                    h_dim=HIDDEN_DIMENSION)  # in_dim=gene_data_train.shape[1]

                if torch.cuda.is_available():
                    ae.cuda()

                criterion =nn.BCELoss()
                optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
                iter_per_epoch = len(data_loader)
                data_iter = iter(data_loader)

                # save fixed inputs for debugging
                fixed_x = next(data_iter)  # fixed_x, _ = next(data_iter)
                #mydir = './data/'
                # difine the directory to be created
                mkpath = ".\\result\\%s"%date
                mkdir(mkpath)
                myfile = "t_img_bth%d.png" % (i + 1)
                images_path = os.path.join(mkpath, myfile)
                torchvision.utils.save_image(Variable(fixed_x).data.cpu(), images_path)
                fixed_x = AE.to_var(fixed_x.view(fixed_x.size(0), -1))
                AE_loss_list = []
                y_train_T_tensor=torch.from_numpy(y_train.T.values).float()
                toPrintInfo = True
                for epoch in range(num_epochs):
                    t0 = time()
                    for i, (images) in enumerate(data_loader):  # for i, (images, _) in enumerate(data_loader):
                        # flatten the image
                        images = AE.to_var(images.view(images.size(0), -1))
                        images = images.float()
                        if toPrintInfo:
                            print("DEBUG INFO:before the input of model MeiNN")
                            print(images)
                        #out,prediction,embedding = ae(images)
                        if multiDatasetMode=="softmax":
                            out, y_pred, embedding = ae(gene_data_train_Tensor.T)
                        elif multiDatasetMode=="multi-task":
                            return out, [pred1, pred2, pred3, pred4, pred5, pred6], embedding
                        #loss= nn.BCELoss(prediction,y_train_T_tensor)+nn.BCELoss(out,images)

                        def BCE_loss_masked(y_pred, y):
                            # y_pred:预测标签，已经过sigmoid/softmax处理 shape is (batch_size, 1)
                            # y：真实标签（一般为0或1） shape is (batch_size)
                            mask= y.ne(0.5)
                            y_masked = torch.masked_select(y, mask)
                            y_pred_masked = torch.masked_select(y_pred, mask)

                            y_pred_masked = torch.cat((1 - y_pred_masked, y_pred_masked), 1)  # 将二种情况的概率都列出，y_hat形状变为(batch_size, 2)
                            # 按照y标定的真实标签，取出预测的概率，来计算损失
                            return - torch.log(y_pred_masked.gather(1, y_masked.view(-1, 1))).mean()
                            # 函数返回loss均值
                        toMask=True
                        y_masked=y_train_T_tensor
                        y_pred_masked=y_pred
                        if toMask:
                            mask = y_train_T_tensor.ne(0.5)
                            #y_masked = torch.masked_select(y_train_T_tensor, mask)
                            y_masked=y_train_T_tensor*mask
                            #y_pred_masked = torch.masked_select(prediction, mask)
                            y_pred_masked = y_pred*mask
                        reg_loss=0
                        for i,param in enumerate(ae.parameters()):
                            if toPrintInfo:
                                print("%d-th layer:"%i)
                                #print(name)
                                print("param:")
                                print(param.shape)
                            reg_loss+=torch.sum(torch.abs(param))

                        loss = reg_loss*0.0001+criterion(y_pred_masked,y_masked)*10000+criterion(out, images)*1

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if toPrintInfo:
                            print("prediction")
                            print(y_pred.shape)
                            print(y_pred)
                            print("y_train.T")
                            print(y_train.T.shape)
                            print(y_train.T)
                            print("y_pred_masked:")
                            print(y_pred_masked.shape)
                            print(y_pred_masked)
                            print("y_masked")
                            print(y_masked.shape)
                            print(y_masked)
                            toPrintInfo=False
                        print("reg_loss%f,ae loss%f,prediction loss-masked%f,prediction loss%f" % (reg_loss,
                        criterion(out, images), criterion(y_pred_masked,y_masked),criterion(y_pred,y_train_T_tensor)))

                        print("loss: %f"%loss.item())
                        #print(loss.item())
                        AE_loss_list.append(loss.item())

                        if (i + 1) % 10 == 0:
                            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs'
                                  % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.item(),
                                     time() - t0))  # original version: loss.item() was loss.data[0]

                    if (epoch + 1) % 1 == 0:
                        # save the reconstructed images
                        fixed_x = fixed_x.float()
                        reconst_images,y_pred,embedding = ae(fixed_x)#prediction
                        reconst_images = reconst_images.view(reconst_images.size(0), gene_data_train.shape[
                            0])  # reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
                        #mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/test_sample/data/'
                        mkpath = ".\\result\\%s" % date
                        mkdir(mkpath)
                        myfile = 'rcnst_img_bt%d_ep%d.png' % ( i + 1, (epoch + 1))
                        reconst_images_path = os.path.join(mkpath, myfile)
                        torchvision.utils.save_image(reconst_images.data.cpu(), reconst_images_path)
                    ##################
                    model = model_dict[model_type]()
                torch.save({"epoch": num_epochs,  # 一共训练的epoch
                 "model_state_dict": ae.state_dict(),  # 保存模型参数×××××这里埋个坑××××
                 "optimizer": optimizer.state_dict()}, path+date + '.tar')

                torch.save(ae, path+date + '.pth')  # save the whole autoencoder network
                AE_loss_list_df = pd.DataFrame(AE_loss_list)
                AE_loss_list_df.to_csv(path+date + "_AE_loss_list).csv",sep='\t')
                if count == 1:
                    with open(path+date + '_train_model_pytorch.pickle', 'wb') as f:
                        pickle.dump((gene, ae), f)  # pickle.dump((gene, model), f)
                else:
                    with open(path+date + '_train_model_pytorch.pickle', 'ab') as f:
                        pickle.dump((gene, ae), f)  # pickle.dump((gene, model), f)





                '''
                myMeiNN = MeiNN_pytorch(config, path, date, code, gene_data_train.T, y_train.T, platform, model_type, data_type,
                                HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=AE_epoch_from_main,
                                NN_epoch_from_main=NN_epoch_from_main, separatelyTrainAE_NN=separatelyTrainAE_NN,
                                model_dir='./results/models',
                                gene_to_residue_or_pathway_info=my_gene_to_residue_info, toAddGeneSite=toAddGeneSite,
                                toAddGenePathway=toAddGenePathway,
                                multiDatasetMode=multiDatasetMode, datasetNameList=datasetNameList, lossMode=lossMode)

                # myMeiNN.build()
                myMeiNN.compile()
                # myMeiNN.fcn.fit(gene_data_train.T, y_train.T, epochs=NN_epoch_from_main, batch_size=79, shuffle=True)
                myMeiNN.fit()'''




            if toAddGenePathway and False:
                print("***************************")
                print("len residuals_name%d" % len(residuals_name))

                print("****gene_pathway_df***********************")
                print(gene_pathway_df)
                print("****csv_data***********************")
                print(gene_pathway_csv_data)
                print("****gene_name_data***********************")
                print(gene_name_data)
                print(gene_name_data.iloc[1])
                print("****gene_present_list_df***********************")
                print(gene_present_list_df)
                print("****gene_present_index***********************")
                print(gene_present_index)
                # print("****gene_present_index_sorted***********************")
                # print(gene_present_index_sorted)
                print("****gene_pathway_present_gene_index***********************")
                print(gene_pathway_present_gene_index)

                # pathway_name_data

            '''
            decoder_regularizer='var_l1'
            decoder_regularizer_initial=0.0001
            activ = 'relu'
            latent_scale=5
            l_rate=K.variable(0.01)
            relu_thresh=0
            decoder_bn=False
            decoder_bias='last'
            last_activ='tanh'#'softmax'
            in_dim = 809
            # Compress the dimension to latent_dim
            mid_dim = math.sqrt(in_dim *  latent_dim)
            q3_dim =math.sqrt(in_dim * mid_dim)
            q1_dim=math.sqrt( latent_dim * mid_dim)
            decoder_shape=[ latent_dim,q1_dim,mid_dim,q3_dim]
            input_shape=(809)
            if decoder_regularizer == 'dot_weights':
                dot_weights = np.zeros(shape=(latent_scale * latent_dim, latent_scale * latent_dim))
                for s in range(latent_dim):
                    dot_weights[s * latent_scale:s * latent_scale + latent_scale,
                    s * latent_scale:s * latent_scale + latent_scale] = 1
            # L1 regularizer with the scaling factor updateable through the l_rate variable (callback)
            def variable_l1(weight_matrix):
                return l_rate * K.sum(K.abs(weight_matrix))
            # L2 regularizer with the scaling factor updateable through the l_rate variable (callback)
            def variable_l2(weight_matrix):
                return l_rate * K.sum(K.square(weight_matrix))
            # Mixed L1 and L2 regularizer, updateable scaling. TODO: Consider implementing different scaling factors for L1 and L2 part
            def variable_l1_l2(weight_matrix):
                return l_rate * (K.sum(K.abs(weight_matrix)) + K.sum(K.square(weight_matrix))) * 0.5
            # Dot product-based regularizer
            def dotprod_weights(weights_matrix):
                penalty_dot = l_rate * K.mean(K.square(K.dot(weights_matrix,
                                                                  K.transpose(weights_matrix)) * dot_weights))
                penalty_l1 = 0.000 * l_rate * K.sum(K.abs(weights_matrix))
                return penalty_dot + penalty_l1
            def dotprod(weights_matrix):
                penalty_dot = l_rate * K.mean(K.square(K.dot(weights_matrix, K.transpose(weights_matrix))))
                penalty_l1 = 0.000 * l_rate * K.sum(K.abs(weights_matrix))
                return penalty_dot + penalty_l1
            def dotprod_inverse(weights_matrix):
                penalty_dot = 0.1 * K.mean(
                    K.square(K.dot(K.transpose(weights_matrix), weights_matrix) * dot_weights))
                penalty_l1 = 0.000 * l_rate * K.sum(K.abs(weights_matrix))
                return penalty_dot + penalty_l1
            def relu_advanced(x):
                return K.relu(x, threshold=relu_thresh)
            if activ == 'relu':
                activ = relu_advanced
            # assigns the regularizer to the scaling factor. TODO: Look for more elegant method
            if decoder_regularizer == 'var_l1':
                reg = variable_l1
                reg1 = variable_l1
            elif decoder_regularizer == 'var_l2':
                reg = variable_l2
                reg1 = variable_l2
            elif decoder_regularizer == 'var_l1_l2':
                reg = variable_l1_l2
                reg1 = variable_l1_l2
            elif decoder_regularizer == 'l1':
                reg = regularizers.l1(decoder_regularizer_initial)
                reg1 = regularizers.l1(decoder_regularizer_initial)
            elif decoder_regularizer == 'l2':
                reg = regularizers.l2(decoder_regularizer_initial)
                reg1 = regularizers.l2(decoder_regularizer_initial)
            elif decoder_regularizer == 'l1_l2':
                reg = regularizers.l1_l2(l1=decoder_regularizer_initial, l2=decoder_regularizer_initial)
                reg1 = regularizers.l1_l2(l1=decoder_regularizer_initial, l2=decoder_regularizer_initial)
            elif decoder_regularizer == 'dot':
                reg = dotprod
                reg1 = dotprod
            elif decoder_regularizer == 'dot_weights':
                reg1 = dotprod_weights
                reg = dotprod
            else:
                reg = None
                reg1 = None
            # this is our input placeholder
            input = Input(shape=(in_dim,))
            # 编码层
            encoded = Dense(q3_dim, activation='relu')(input)
            encoded = Dense(mid_dim, activation='relu')(encoded)
            encoded = Dense(q1_dim, activation='relu')(encoded)
            encoder_output = Dense( latent_dim,name="input_to_encoding")(encoded)
            decoded = layers.Dense(q1_dim,
                             activation=activ,
                             name='Decoder1',
                             activity_regularizer=reg1)(encoder_output)
            if decoder_bn:
                decoded = layers.BatchNormalization()(decoded)
            # adds layers to the decoder. See encoder layers
            if len(decoder_shape) > 1:
                for i in range(len(decoder_shape) - 1):
                    if decoder_bias == 'all':
                        decoded = layers.Dense(decoder_shape[i + 1],
                                         activation=activ,
                                         name='Dense_D' + str(i + 2),
                                         use_bias=True,
                                         activity_regularizer=reg)(decoded)
                    else:
                        decoded = layers.Dense(decoder_shape[i + 1],
                                         activation=activ,
                                         name='Dense_D' + str(i + 2),
                                         use_bias=False,
                                         kernel_regularizer=reg)(decoded)
                    if decoder_bn:
                        decoded = layers.BatchNormalization()(decoded)
            if decoder_bias == 'none':
                ae_outputs = layers.Dense(input_shape,
                                              activation=last_activ,
                                              use_bias=False,name='ae_output')(decoded)
            else:
                ae_outputs = layers.Dense(input_shape,
                                              activation=last_activ,name='ae_output')(decoded)
            # 解码层
            #decoded = Dense(q1_dim, activation='relu')(encoder_output)
            #decoded = Dense(mid_dim, activation='relu')(decoded)
            #decoded = Dense(q3_dim, activation='relu')(decoded)
            #decoded = Dense(in_dim, activation='tanh')(decoded)
            # 构建自编码模型
            autoencoder = Model(inputs=input, outputs=ae_outputs)
            # 构建编码模型
            encoder = Model(inputs=input, outputs=encoder_output)
            # compile autoencoder
            autoencoder.compile(optimizer='adam', loss='binary_crossentropy') #loss='mse'
            class CustomAutoencoder(layers.Layer):
                def __init__(self):
                    super(CustomAutoencoder, self).__init__()
                    # this is our input placeholder
                    input = Input(shape=(in_dim,))
                    # 编码层
                    encoded = Dense(q3_dim, activation='relu')(input)
                    encoded = Dense(mid_dim, activation='relu')(encoded)
                    encoded = Dense(q1_dim, activation='relu')(encoded)
                    encoder_output = Dense(latent_dim, name="input_to_encoding")(encoded)
                    decoded = layers.Dense(q1_dim,
                                           activation=activ,
                                           name='Decoder1',
                                           activity_regularizer=reg1)(encoder_output)
                    if decoder_bn:
                        decoded = layers.BatchNormalization()(decoded)
                    # adds layers to the decoder. See encoder layers
                    if len(decoder_shape) > 1:
                        for i in range(len(decoder_shape) - 1):
                            if decoder_bias == 'all':
                                decoded = layers.Dense(decoder_shape[i + 1],
                                                       activation=activ,
                                                       name='Dense_D' + str(i + 2),
                                                       use_bias=True,
                                                       activity_regularizer=reg)(decoded)
                            else:
                                decoded = layers.Dense(decoder_shape[i + 1],
                                                       activation=activ,
                                                       name='Dense_D' + str(i + 2),
                                                       use_bias=False,
                                                       kernel_regularizer=reg)(decoded)
                            if decoder_bn:
                                decoded = layers.BatchNormalization()(decoded)
                    if decoder_bias == 'none':
                        self.ae_outputs = layers.Dense(input_shape,
                                                  activation=last_activ,
                                                  use_bias=False, name='ae_output')(decoded)
                    else:
                        self.ae_outputs = layers.Dense(input_shape,
                                                  activation=last_activ, name='ae_output')(decoded)
                    self.autoencoder = Model(inputs=input, outputs=ae_outputs)
                    # 构建编码模型
                    self.encoder = Model(inputs=input, outputs=encoder_output)
                def call(self, inputs,option='ae_output'):
                    if(option=='ae_output'):
                        return self.autoencoder(input)
                    elif(option=='embedding'):
                        return self.encoder(input)
            # training
            '''

            '''
            MeiNN.autoencoder.fit(gene_data_train.T, gene_data_train.T, epochs=AE_epoch_from_main, batch_size=79, shuffle=True)
            print("AE finish_fitting")
            MeiNN.autoencoder.save(date+'AE.h5')
            print("AE finish saving model")


            ################################################################
            #the following is the embedding to y prediction
            #ae=torch.load(date+'_auto-encoder.pth')
            #loaded_autoencoder = load_model(date + 'AE.h5',custom_objects={'variable_l1': variable_l1,'relu_advanced':relu_advanced})
            input_to_encoding_model = Model(inputs=autoencoder.input,
                                       outputs=autoencoder.get_layer('input_to_encoding').output)
            print("input_to_encoding_model.predict(gene_data_train.T)")
            print(input_to_encoding_model.predict(gene_data_train.T))
            # embedding=ae.code(torch.tensor(gene_data_train.T).float())
            embedding = input_to_encoding_model.predict(gene_data_train.T)
            embedding_df = pd.DataFrame(embedding)
            embedding_df.to_csv(path+date+"_"+code + "_gene_level" + "(" + data_type + '_' + model_type + "_embedding_original).txt", sep='\t')
            print("embedding is ")
            print(embedding)
            print(embedding.shape)
            in_dim =  latent_dim
            # output dimension is 1
            out_dim = 1
            mid_dim = math.sqrt(in_dim *  latent_dim)
            q3_dim = math.sqrt(in_dim * mid_dim)

            q1_dim = math.sqrt( latent_dim * mid_dim)
            # this is our input placeholder
            #input = Input(shape=(in_dim,))
            # 编码层
            out_x = Dense(q3_dim, activation='relu')(encoder_output)
            out_x = Dense(mid_dim, activation='relu')(out_x)
            out_x = Dense(q1_dim, activation='relu')(out_x)
            output = Dense(out_dim,activation='sigmoid',name="prediction")(out_x)#originally sigmoid
            def reconstruct_and_predict_loss(x,ae_outputs,output,y_train,y_index):
                reconstruct_loss = losses.binary_crossentropy(x, ae_outputs)
                print(output[0])
                print("y_train.T=")
                print(y_train.T)
                print("y_train.T.shape=")
                print(y_train.T.shape)
                predict_loss =losses.binary_crossentropy(y_pred=output,y_true=y_train.T[y_index]) #- 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
                return reconstruct_loss + predict_loss
            def my_reconstruct_and_predict_loss(y_true, y_pred, lam=0.5):
                reconstruct_loss = losses.binary_crossentropy(y_true=autoencoder.input, y_pred=autoencoder.get_layer('ae_output').output)
                predict_loss = losses.binary_crossentropy(y_true,
                                                          y_pred)  # - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
                return K.mean(lam * reconstruct_loss + (1 - lam) * predict_loss)
            # build the fcn model
            fcn = Model(inputs=input, outputs=output)
            # compile fcn
            fcn.compile(optimizer='adam', loss= my_reconstruct_and_predict_loss,experimental_run_tf_function=False)  # loss='mse'#'binary_crossentropy'
            # training
            #fcn.fit(embedding, y_train.T, epochs=NN_epoch_from_main, batch_size=79, shuffle=True)
            fcn.fit(gene_data_train.T, y_train.T, epochs=NN_epoch_from_main, batch_size=79, shuffle=True)
            print("FCN finish_fitting")
            fcn.save(path+date + 'FCN.h5')
            print("FCN finish saving model")
            embedding = input_to_encoding_model.predict(gene_data_train.T)  # input_to_encoding_model.predict(gene_data_train.T)
            embedding_df = pd.DataFrame(embedding)
            embedding_df.to_csv(
                path+date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "_embedding_trained).txt",
                sep='\t')
            print("embedding is ")
            print(embedding)
            print(embedding.shape)
            '''
            '''
            num_epochs = NN_epoch_from_main
            batch_size = 79 # gene_data_train.shape[0]#100#809
            hidden_size = 10
            dataset = gene_data_train.T  # .flatten()#gene_data_train.view(gene_data_train.size[0], -1)
            # dataset = gene_data_train  # dsets.MNIST(root='../data',
            # train=True,
            # transform=transforms.ToTensor(),
            # download=True)
            data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True)
            print("gene_data_train.shape")
            print(gene_data_train.shape)
            print("dataset.shape")
            print(dataset.shape)
            #ae = AE.Autoencoder(in_dim=gene_data_train.shape[0], h_dim=79 * 5)  # in_dim=gene_data_train.shape[1]
            fcn=AE.NN(in_dim=HIDDEN_DIMENSION, h_dim=1)
            if torch.cuda.is_available():
                fcn.cuda()
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
            iter_per_epoch = len(data_loader)
            data_iter = iter(data_loader)
            # save fixed inputs for debugging
            fixed_x = next(data_iter)  # fixed_x, _ = next(data_iter)
            mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/test_sample/data/'
            myfile = '%snn_real_image_%s_batch%d.png' % (date,code, i + 1)
            images_path = os.path.join(mydir, myfile)
            torchvision.utils.save_image(Variable(fixed_x).data.cpu(), images_path)
            fixed_x = AE.to_var(fixed_x.view(fixed_x.size(0), -1))
            NN_loss_list=[]
            for epoch in range(num_epochs):
                t0 = time()
                for i, (images) in enumerate(data_loader):  # for i, (images, _) in enumerate(data_loader):
                    # flatten the image
                    images = AE.to_var(images.view(images.size(0), -1))
                    images = images.float()
                    #embedding
                    embedding_=ae.code(images)
                    out = fcn(embedding_)
                    #print("out at tain.py nn ")
                    #print(out)
                    #print("torch.tensor(y_train).float() at tain.py nn ")
                    #print(torch.tensor(y_train).float())
                    out=torch.reshape(out, (-1,))
                    loss = criterion(out, torch.tensor(y_train).float().T)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("training nn, epoch %d : loss= "% epoch)
                    print(loss.item())
                    NN_loss_list.append(loss.item())
                    if (i + 1) % 100 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs'
                              % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.item(),
                                 time() - t0))  # original version: loss.item() was loss.data[0]
                        print("out after reshape")
                        print(out.shape)
                        print(out)
                if (epoch + 1) % 1 == 0:
                    fixed_x = fixed_x.float()
                    embedding_out = ae.code(torch.tensor(fixed_x).float())
                    reconst_images = fcn(embedding_out)
                    reconst_images = reconst_images.view(reconst_images.size(0),
                                                         -1)  # reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
                    mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/test_sample/data/'
                    myfile = '%snn_reconst_images_%s_batch%d_epoch%d.png' % (date,code, i + 1, (epoch + 1))
                    reconst_images_path = os.path.join(mydir, myfile)
                    torchvision.utils.save_image(reconst_images.data.cpu(), reconst_images_path)
            torch.save(fcn, date+'_fully-connected-network.pth')
            NN_loss_list_df = pd.DataFrame(NN_loss_list)
            NN_loss_list_df.to_csv(
                date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "_NN_loss).csv",
                sep='\t')
        '''

    else:
        model = model_dict[model_type]()
        model.fit(gene_data_train.T, y_train)
        if model_type == "RandomForest":
            print("The number of residuals involved in the gene {} is {}".format(gene, len(gene_data_train)))
            print("The feature importance is ")
            print(model.feature_importances_)
            print("The names of residuals are ")
            print(residuals_name)
            print(15 * '-')

        if count == 1:
            with open(path + date + "_" + code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'wb') as f:
                pickle.dump((gene, model), f)
        else:
            with open(path + date + "_" + code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'ab') as f:
                pickle.dump((gene, model), f)
    print("Training finish!")
    return myMeiNN,residuals_name


def train_VAE(model, train_db, optimizer=tf.keras.optimizers.Adam(0.001), n_input=80):
    for epoch in range(1000):
        for step, x in enumerate(train_db):
            x = tf.reshape(x, [-1, n_input])
            with tf.GradientTape() as tape:
                x_rec_logits, mean, log_var = model(x)
                rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
                rec_loss = tf.reduce_mean(rec_loss)
                # compute kl divergence (mean, val) ~ N(0, 1)
                kl_div = -0.5 * (log_var + 1 - mean ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_mean(kl_div) / x.shape[0]
                # loss
                loss = rec_loss + 1.0 * kl_div

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 10 == 0:
                print(epoch, step, 'kl_div:', float(kl_div), 'rec_loss:', rec_loss)


if __name__ == '__main__':
    # Parameter description：
    # code: dataSet ID such as GSE66695 ( string )
    # train_file: train data filename( .txt )
    # label_file: train label filename(.txt)
    # platform: Gene correspond to methylation characteristics( json file )
    # model_type: type of regression model ( string )
    # data_type: type of data ( string )

    # example

    code = "GSE66695"
    train_file = "data_train.txt"
    label_file = "label_train.txt"
    platform = "platform.json"
    model_type = "LinearRegression"
    data_type = "origin_data"

    train_data = pd.read_table(train_file, index_col=0)
    train_label = pd.read_table(label_file, index_col=0).values.ravel()

    run(code, train_data, train_label, platform, model_type, data_type)