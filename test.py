# encoding=UTF-8
from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
from elm import ELMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import sys
stdsc = StandardScaler()

def load_data(filename, labelname):
    data = pd.read_table(filename, index_col=0)
    label = pd.read_table(labelname, index_col=0).values.ravel()
    return data, label

def classfiers(X, Y,times,fold):
    #Using six classifiers to do cross validation for X and Y times fold.
    classifier = [SVC(kernel='linear'), GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=10),
                  GradientBoostingClassifier(random_state=10), ELMClassifier()]
    acc_res = []
    for clf in classifier:
        if clf == classifier[-1]:
            X = stdsc.fit_transform(X)
        each_score = []
        for i in range(times):
            acc_temp = []
            skf = StratifiedKFold(n_splits=fold, random_state=i, shuffle=True)
            for train_index, test_index in skf.split(X, Y):
                # print('Train: ',train_index,'Test: ',test_index)
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                clf.fit(X_train, Y_train)
                acc1 = accuracy_score(Y_test, clf.predict(X_test))
                acc_temp.append(acc1)
            each_score.append(np.mean(acc_temp))
        acc_res.append(np.mean(each_score))
    return acc_res

def IFS_validation(X, Y, times,fold,select,tag):
    feature_order = X.columns
    list_gene = []
    result = []
    print("feature_number:")
    for i in range(1, len(feature_order) + 1):
        print(i)
        list_gene.append(feature_order[:i])
        result.append(classfiers(X[feature_order[:i]].values, Y,times,fold))
    score = pd.DataFrame(result)
    col = ['SVM', 'NBayes', 'KNN', 'DTree', 'GBDT', 'ELM']
    score.columns = col
    score.index = list(map(lambda x: ','.join(x), list_gene))
    score.to_csv(tag + "_" + select + "_Acc_" + str(times) + "_runs_" + str(fold) + "_fold.csv")

def TRank(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "Trank"
    X = data
    print(select + ' selection start...')
    p_index = np.argwhere(label == 1).ravel()
    n_index = np.argwhere(label == 0).ravel()
    X.columns = list(range(len(label)))
    p_data = X[p_index]
    n_data = X[n_index]
    res = []
    for gene in p_data.index:
        data1 = p_data.loc[gene]
        data2 = n_data.loc[gene]
        res.append(stats.ttest_ind(data1,data2).pvalue)
    result = pd.DataFrame({geneOrResidue: p_data.index, 'pvalue': res}).sort_values(by='pvalue')
    print(select + " result.head()")
    print(result.head())
    col = [geneOrResidue, 'pvalue']
    result.to_csv(tag + '_Trank_rank.txt', index=None, columns=col, sep="\t")
    print(select + ' selection finish...')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = X.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    # print(cur_X)
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def WRank(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "Wrank"
    X = data
    print(select + ' selection start...')
    p_index = np.argwhere(label == 1).ravel()
    n_index = np.argwhere(label == 0).ravel()
    X.columns = list(range(len(label)))
    p_data = X[p_index]
    n_data = X[n_index]
    res = []
    for gene in p_data.index:
        data1 = p_data.loc[gene]
        data2 = n_data.loc[gene]
        res.append(stats.ranksums(data1, data2).pvalue)
    result = pd.DataFrame({geneOrResidue: p_data.index, 'pvalue': res}).sort_values(by='pvalue')
    print(select + " result.head()")
    print(result.head())
    col = [geneOrResidue, 'pvalue']
    result.to_csv(tag + '_Wrank_rank.txt', index=None, columns=col, sep="\t")
    print(select + ' selection finish...')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = X.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    # print(cur_X)
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def Chi2(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "Chi2"
    print('Chi2 selection start...')
    X = data.T

    minMaxScaler = preprocessing.MinMaxScaler()
    minMax = minMaxScaler.fit_transform(X)
    X = pd.DataFrame(minMax, columns=X.columns.values)
    print('after standardized...')
    print(X.head())
    (chi, pval) = chi2(X, label)
    res = pd.DataFrame({geneOrResidue: data.index.tolist(), 'chi2': chi}).sort_values(by='chi2', ascending=False)
    print('Chi2 result.head()')
    print(res.head())
    col = [geneOrResidue, 'chi2']
    res.to_csv(tag + '_Chi2_rank.txt', sep="\t", index=None, columns=col)
    print('Chi2 selection finish...')
    print(select + " IFS validation start...")
    result = res.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    IFS_validation(cur_X, label, times,fold,select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def ROCRank(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "ROCrank"
    print('ROCRank start...')
    X = data.T
    feat_labels = X.columns.tolist()
    auc_list = []
    clf = GaussianNB()
    random_seed = 0
    for item in feat_labels:
        X_train, X_test, y_train, y_test = train_test_split(X[item].values, label, test_size=0.3,
                                                            random_state=random_seed)
        clf.fit(X_train.reshape(-1, 1), y_train)
        y_pred = clf.predict_proba(X_test.reshape(-1, 1))
        auc_list.append(roc_auc_score(y_test, y_pred[:, 1]))
    result = pd.DataFrame({geneOrResidue: feat_labels, 'AUC': auc_list}).sort_values(by='AUC', ascending=False)
    print('ROCRank result.head()')
    print(result.head())
    col = [geneOrResidue, 'AUC']
    result.to_csv(tag + '_ROCRank_rank.txt', sep="\t", index=None, columns=col)
    print('ROCRank selection finish')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def RF(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "RF"
    print('RF selection start...')
    X = data.T
    forest = RandomForestClassifier(n_estimators=1000, random_state=10, n_jobs=1)
    forest.fit(X, label)
    importance = forest.feature_importances_
    result = pd.DataFrame({geneOrResidue: X.columns.values, "importance": importance})
    result = result.sort_values(by='importance', ascending=False)
    print('RF result.head()')
    print(result.head())
    col = [geneOrResidue, "importance"]
    result.to_csv(tag + "_RF_rank.txt", sep="\t", index=None, columns=col)
    print('RF selection finish...')
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def SVM_RFE(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "SVM_RFE"
    print("SVM_RFE selection start...")

    if geneOrResidue != "gene"and len(data)>20000:  # top 20000 pvalue
        order = pd.read_csv(tag + '_Trank_rank.txt', index_col=0).iloc[list(range(20000))]
        # print(order)
        X = data.loc[order.index]
        X = X.T
    else:
        X = data.T
    # use svm as the model
    clf = SVC(kernel='linear', C=1)
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, label)
    result = pd.DataFrame({geneOrResidue: X.columns.values, "rank": rfe.ranking_}).sort_values(by='rank')
    print("SVM_RFE result.head()")
    print(result.head())
    col = [geneOrResidue, 'rank']
    result.to_csv(tag + '_SVM_RFE_rank.txt', sep="\t", index=None, columns=col)
    print("SVM_RFE selection finish...")
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def LR_RFE(data, label,times,fold,cnt,tag,geneOrResidue):
    select = "LR_RFE"
    print(select + " selection start...")
    if geneOrResidue != "gene" and len(data)>20000:
        order = pd.read_csv(tag + '_Trank_rank.txt', index_col=0).iloc[list(range(20000))]
        # print(order)
        X = data.loc[order.index]
        X = X.T
    else:
        X = data.T
    # use linear regression as the model
    lr = LinearRegression()
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X, label)
    result = pd.DataFrame({geneOrResidue: X.columns.values, "rank": rfe.ranking_}).sort_values(by='rank')
    print(select + " result.head()")
    print(result.head())
    col = [geneOrResidue, 'rank']
    result.to_csv(tag + '_LR_RFE_rank.txt', sep="\t", index=None, columns=col)
    print(select + " selection finish...")
    print(select + " IFS validation start...")
    result = result.iloc[list(range(cnt))]
    cur_X = data.loc[result[geneOrResidue]]
    cur_X.to_csv(tag + "_" + select + "_" + str(cnt) + "_result.csv")
    cur_X = cur_X.T
    IFS_validation(cur_X, label,times,fold, select,tag)
    print(select + ' IFS validation finish...')
    print("------------------------")

def select_feature(code,data,label,gene=True,times=20,fold=5,cnt=100):
    if gene:
        geneOrResidue="gene"
    else:
        geneOrResidue="residue"

    tag = code + "_" + geneOrResidue # Prefix for result file
    #methods
    methods= [TRank(data, label,times,fold,cnt,tag,geneOrResidue),
              WRank(data, label,times,fold,cnt,tag,geneOrResidue),
              Chi2(data, label,times,fold,cnt,tag,geneOrResidue),
              ROCRank(data, label,times,fold,cnt,tag,geneOrResidue),
              RF(data, label,times,fold,cnt,tag,geneOrResidue),
              SVM_RFE(data, label,times,fold,cnt,tag,geneOrResidue),
              LR_RFE(data, label,times,fold,cnt,tag,geneOrResidue)]

    for method in methods:
        method

if __name__=='__main__':
    # Parameter description：
    # code: dataSet ID such as GSE66695
    # filename:The name of the feature data file for feature selection(.txt)
    # labelname: Label file name(.txt)
    # gene: feature type ( bool )
    # times: Number of cross validation cycles
    # fold：fold of cross validation score
    # cnt:Number of features for IFS

    #example
    code="GSE66695"
    filename="GSE66695_gene_level(origin_data_LinearRegression).txt"
    labelname="label_test.txt"
    data, label = load_data(filename, labelname)
    select_feature(code,data,label,gene=True) # feature type is residue
