# data_train.py
import re
import os
import json
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from models.vanilla_vae import VanillaVAE as VAE
from models import *
from data_dict import *
import yaml
import argparse
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning import Trainer
import warnings
warnings.filterwarnings("ignore")


# Only train regression model, save parameters to pickle file
def run(code,X_train,y_train,platform,model_type, data_type,latent_dim):
    data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                 'radical_data': radical_data, 'cube_data': cube_data}
    model_dict = {'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression,
                  'L1': Lasso, 'L2': Ridge,'VAE': VAE(len(X_train),latent_dim)}

    with open(platform, 'r') as f:
        gene_dict = json.load(f)
        f.close()

    count = 0
    num = len(gene_dict)
    gene_list = []
    print('Now start training gene...')

    data_train = data_dict[data_type](X_train)
    for gene in gene_dict:
        count += 1
        gene_data_train = []
        for residue in data_train.index:
            if residue in gene_dict[gene]:
                gene_data_train.append(data_train.loc[residue])
        if len(gene_data_train) == 0:
            print('Contained Nan data, has been removed!')
            continue

        gene_data_train = np.array(gene_data_train)
        gene_list.append(gene)
        print('Now training ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(
            int(count * 100 / num)) + '% ...')
        if not (model_type == 'VAE'):
            model = model_dict[model_type]()
            model.fit(gene_data_train.T,y_train)
        else:
            parser = argparse.ArgumentParser(description='Generic runner for VAE models')
            parser.add_argument('--config', '-c',
                                dest="filename",
                                metavar='FILE',
                                help='path to the config file',
                                default='configs/vae.yaml')

            args = parser.parse_args()
            with open(args.filename, 'r') as file:
                try:
                    config = yaml.safe_load(file)
                except yaml.YAMLError as exc:
                    print(exc)

            tt_logger = TestTubeLogger(
                save_dir=config['logging_params']['save_dir'],
                name=config['logging_params']['name'],
                debug=False,
                create_git_tag=False,
            )

            # For reproducibility
            torch.manual_seed(config['logging_params']['manual_seed'])
            np.random.seed(config['logging_params']['manual_seed'])
            cudnn.deterministic = True
            cudnn.benchmark = False

            model = vae_models[config['model_params']['name']](**config['model_params'])
            experiment = VAEXperiment(model,
                                      config['exp_params'])

            runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                             min_nb_epochs=1,
                             logger=tt_logger,
                             log_save_interval=100,
                             train_percent_check=1.,
                             val_percent_check=1.,
                             num_sanity_val_steps=5,
                             early_stop_callback=False,
                             **config['trainer_params'])

            print(f"======= Training {config['model_params']['name']} =======")
            runner.fit(experiment)

        if count == 1:
            with open(code+"_"+model_type + "_" + data_type + 'train_model.pickle', 'wb') as f:
                pickle.dump((gene, model), f)
        else:
            with open(code+"_"+model_type + "_" + data_type + 'train_model.pickle', 'ab') as f:
                pickle.dump((gene, model), f)
        print('finish!')

    print("Training finish!")


if __name__=='__main__':
    # Parameter description：
    # code: dataSet ID such as GSE66695 ( string )
    # train_file: train data filename( .txt )
    # label_file: train label filename(.txt)
    # platform: Gene correspond to methylation characteristics( json file )
    # model_type: type of regression model ( string )
    # data_type: type of data ( string )

    # example

    code="GSE66695"
    train_file="data_train.txt"
    label_file="label_train.txt"
    platform="platform.json"
    #model_type="LinearRegression"
    model_type = "VAE"
    data_type="origin_data"

    train_data = pd.read_table(train_file, index_col=0)
    train_label = pd.read_table(label_file, index_col=0).values.ravel()

    run(code,train_data, train_label, platform, model_type, data_type)
