# data_train.py
import re
#from resVAE.resvae import resVAE
#import resVAE.utils as cutils
#from resVAE.config import config
#import resVAE.reporting as report
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
from keras.models import load_model
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
#import TabularAutoEncoder
#import VAE
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import torch
from torch import nn
#import torchvision
from torch.autograd import Variable
#import AutoEncoder
import math
import warnings
import AutoEncoder as AE
from time import time
from keras import layers
from keras import losses
from keras import regularizers
from keras import backend as K

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


# Only train regression model, save parameters to pickle file
def run(path,date,code, X_train, y_train, platform, model_type, data_type,HIDDEN_DIMENSION,toTrainAE,toTrainNN,AE_epoch_from_main=1000,NN_epoch_from_main=1000):

    data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                 'radical_data': radical_data, 'cube_data': cube_data}
    model_dict = {'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression,
                  'L1': Lasso, 'L2': Ridge, 'RandomForest': RandomForestRegressor,'AE':AE.Autoencoder}

    with open(platform, 'r') as f:
        gene_dict = json.load(f)
        f.close()

    count = 0
    num = len(gene_dict)
    gene_list = []
    print('Now start training gene...')

    data_train = data_dict[data_type](X_train)

    gene_data_train = []
    residuals_name = []
    model=None
    for (i,gene) in enumerate(gene_dict):
        count += 1
        #gene_data_train = []
        #residuals_name = []
        for residue in data_train.index:
            if residue in gene_dict[gene]:
                gene_data_train.append(data_train.loc[residue])
                residuals_name.append(residue)
        if len(gene_data_train) == 0:
            # print('Contained Nan data, has been removed!')
            continue

        #gene_data_train = np.array(gene_data_train)
        gene_list.append(gene)

        print('No.'+str(i)+'Now training ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(
                int(count * 100 / num)) + '% ...')
        #print("gene_data_train.shape[1]")
        #print(np.array(gene_data_train).shape[1])

        if count == 1:
            with open(path+date+"_"+code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'wb') as f:
                pickle.dump((gene, model), f)
        else:
            with open(path+date+"_"+code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'ab') as f:
                pickle.dump((gene, model), f)
        print('finish!')

    gene_data_train = np.array(gene_data_train)#added line on 2-3
    print("gene_data_train=")
    print(gene_data_train)
    #ae=None
    autoencoder=None
    fcn=None
    if (model_type == "VAE" or model_type == "AE"or model_type == "MeiNN"):
        #encoding_dim = h_dim
        latent_dim = HIDDEN_DIMENSION
        if toTrainAE:
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
            # 压缩特征维度至400维
            mid_dim = math.sqrt(in_dim * latent_dim)
            q3_dim =math.sqrt(in_dim * mid_dim)
            q1_dim=math.sqrt(latent_dim * mid_dim)
            decoder_shape=[latent_dim,q1_dim,mid_dim,q3_dim]
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
            encoder_output = Dense(latent_dim,name="input_to_encoding")(encoded)

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
                                              use_bias=False)(decoded)
            else:
                ae_outputs = layers.Dense(input_shape,
                                              activation=last_activ)(decoded)
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

            # training
            autoencoder.fit(gene_data_train.T, gene_data_train.T, epochs=AE_epoch_from_main, batch_size=79, shuffle=True)
            print("AE finish_fitting")
            autoencoder.save(path+date+'AE.h5')
            print("AE finish saving model")
            #loaded_autoencoder = load_model(date+'AE.h5')
            '''
            num_epochs = AE_epoch_from_main
            batch_size = 79#gene_data_train.shape[0]#100#809
            hidden_size = 10
            dataset = gene_data_train.T#.flatten()#gene_data_train.view(gene_data_train.size[0], -1)
            #dataset = gene_data_train  # dsets.MNIST(root='../data',

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
            ae = AE.Autoencoder(in_dim=gene_data_train.shape[0], h_dim=HIDDEN_DIMENSION)#in_dim=gene_data_train.shape[1]
            if torch.cuda.is_available():
                ae.cuda()

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
            iter_per_epoch = len(data_loader)
            data_iter = iter(data_loader)


            # save fixed inputs for debugging
            fixed_x = next(data_iter)  # fixed_x, _ = next(data_iter)
            mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/test_sample/data/'
            myfile = '%sreal_image_%s_batch%d.png' % (date,code, i + 1)
            images_path = os.path.join(mydir, myfile)
            torchvision.utils.save_image(Variable(fixed_x).data.cpu(), images_path)
            fixed_x = AE.to_var(fixed_x.view(fixed_x.size(0), -1))
            AE_loss_list=[]
            for epoch in range(num_epochs):

                t0 = time()
                for i, (images) in enumerate(data_loader):  # for i, (images, _) in enumerate(data_loader):

                    # flatten the image
                    images = AE.to_var(images.view(images.size(0), -1))
                    images = images.float()
                    out = ae(images)
                    loss = criterion(out, images)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(loss.item())
                    AE_loss_list.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs'
                              % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.item(),
                                 time() - t0))  # original version: loss.item() was loss.data[0]


                if (epoch + 1) % 1 == 0:
                    # save the reconstructed images
                    fixed_x = fixed_x.float()
                    reconst_images = ae(fixed_x)
                    reconst_images = reconst_images.view(reconst_images.size(0), gene_data_train.shape[0])  # reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
                    mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/test_sample/data/'
                    myfile = '%sreconst_images_%s_batch%d_epoch%d.png' % (date,code, i+1, (epoch + 1))
                    reconst_images_path = os.path.join(mydir, myfile)
                    torchvision.utils.save_image(reconst_images.data.cpu(), reconst_images_path)
                ##################
                model = model_dict[model_type]()


            AE_loss_list_df = pd.DataFrame(AE_loss_list)
            AE_loss_list_df.to_csv(
                date + "_" + code + "_gene_level" + "(" + data_type + '_' + model_type + "_AE_loss).csv",
                sep='\t')
            if count == 1:
                with open(date+"_"+code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'wb') as f:
                    pickle.dump((gene, ae), f)  # pickle.dump((gene, model), f)
            else:
                with open(date+"_"+code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'ab') as f:
                    pickle.dump((gene, ae), f)  # pickle.dump((gene, model), f)

            torch.save(ae, date+'_auto-encoder.pth')#save the whole autoencoder network
        '''
################################################################
        #the following is the embedding to y prediction
        if(toTrainNN):
            #ae=torch.load(date+'_auto-encoder.pth')
            loaded_autoencoder = load_model(path+date + 'AE.h5',custom_objects={'variable_l1': variable_l1,'relu_advanced':relu_advanced})

            input_to_encoding_model = Model(inputs=loaded_autoencoder.input,
                                       outputs=loaded_autoencoder.get_layer('input_to_encoding').output)
            # embedding=ae.code(torch.tensor(gene_data_train.T).float())
            embedding = input_to_encoding_model.predict(gene_data_train.T)


            embedding_df = pd.DataFrame(embedding)
            embedding_df.to_csv(path+date+"_"+code + "_gene_level" + "(" + data_type + '_' + model_type + "_embedding).txt", sep='\t')

            print("embedding is ")
            print(embedding)
            print(embedding.shape)

            in_dim = latent_dim
            # output dimension is 1
            out_dim = 1

            mid_dim = math.sqrt(in_dim * latent_dim)
            q3_dim = math.sqrt(in_dim * mid_dim)


            
            q1_dim = math.sqrt(latent_dim * mid_dim)
            # this is our input placeholder
            input = Input(shape=(in_dim,))

            # 编码层
            out_x = Dense(q3_dim, activation='relu')(input)
            out_x = Dense(mid_dim, activation='relu')(out_x)
            out_x = Dense(q1_dim, activation='relu')(out_x)
            output = Dense(out_dim,activation='sigmoid',name="prediction")(out_x)#originally sigmoid


            # build the fcn model
            fcn = Model(inputs=input, outputs=output)
            # compile fcn
            fcn.compile(optimizer='adam', loss='binary_crossentropy')  # loss='mse'
            # training
            fcn.fit(embedding, y_train.T, epochs=NN_epoch_from_main, batch_size=79, shuffle=True)
            print("FCN finish_fitting")
            fcn.save(path+date + 'FCN.h5')
            print("FCN finish saving model")

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
            with open(path+date+"_"+code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'wb') as f:
                pickle.dump((gene, model), f)
        else:
            with open(path+date+"_"+code + "_" + model_type + "_" + data_type + 'train_model.pickle', 'ab') as f:
                pickle.dump((gene, model), f)
    print("Training finish!")
    return (autoencoder,fcn)


def train_VAE(model,train_db,optimizer=tf.keras.optimizers.Adam(0.001),n_input=80):
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
