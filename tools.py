

def evaluate_accuracy_list(datasetNameList,Y_test,pred_out,toPrint=True):
    normalized_pred_out = pred_out#[[0] * len(datasetNameList) for i in range(len(pred_out))]
    num_wrong_pred = 0
    num_wrong_pred_each_dataset=[0]*len(pred_out)
    num_total_each_dataset = [0] * len(pred_out)
    if len(datasetNameList) > 1:
        for i, predict_out_i in enumerate(pred_out):#i means i-th dataset
            for j, pred_out_i_j in enumerate(predict_out_i):# j means j-th input dimension
                if Y_test.iloc[i,j]!=0.5:#when the label is 0.5, it's not considered.
                    num_total_each_dataset[i]+=1
                    '''
                    print("DEBUG:Y_test.iloc[%d,%d]!=0.5"%(i,j))
                    print("DEBUG:Y_test.iloc[%d,%d]=%f"%(i,j,Y_test.iloc[i,j].item()))
                    print("DEBUG:pred_out[%d,%d]=%f"%(i,j,pred_out[i][j].item()))
                    '''
                    normalized_pred_out[i][j] = 1.0 if pred_out[i][j]>=0.5 else 0.0
                    num_wrong_pred += 1.0 if (abs(Y_test.iloc[i, j].item() - normalized_pred_out[i][j]))>=0.5 else 0.0
                    num_wrong_pred_each_dataset[i] += 1.0 if (abs(Y_test.iloc[i, j].item() - normalized_pred_out[i][j])) else 0.0
                    '''
                    if pred_out_i_j >= 0.5:
                        normalized_pred_out[i][j] = 1
                        num_wrong_pred += round(abs(Y_test.iloc[i,j] - 1.0))
                        num_wrong_pred_each_dataset[i] += round(abs(Y_test.iloc[i,j] - 1.0))
                    elif pred_out_i_j < 0.5:
                        normalized_pred_out[i][j] = 0
                        num_wrong_pred += round(abs(Y_test.iloc[i,j] - 0.0))
                        num_wrong_pred_each_dataset[i] += round(abs(Y_test.iloc[i, j] - 0.0))'''
    elif len(datasetNameList) == 1:
        for i, item in enumerate(pred_out):
            if item >= 0.5:
                normalized_pred_out.append(1)
                num_wrong_pred += round(abs(Y_test[0].iloc[i] - 1.0))
            elif item < 0.5:
                normalized_pred_out.append(0)
                num_wrong_pred += round(abs(Y_test[0].iloc[i] - 0.0))
    split_accuracy_dict = [0.0]*6
    for i in range(len(pred_out)):
        split_accuracy_dict[i] = (1.0 - num_wrong_pred_each_dataset[i] / num_total_each_dataset[i])  # [datasetNameList[i]]
    if toPrint:
        print("Y_test")
        print(Y_test.shape)
        print(Y_test)
        print("prediction")
        #print(pred_out.shape)
        #print(pred_out)
        print(len(pred_out[0]))
        print("for each dataset")

        for i in range(len(pred_out)):
            print("dataset %s"% datasetNameList[i])
            print("total case %d"%num_total_each_dataset[i])
            print("wrong prediction %d"%num_wrong_pred_each_dataset[i])
            print("accuracy for this dataset=%f"% (1.0-num_wrong_pred_each_dataset[i]/num_total_each_dataset[i]))
            #split_accuracy_dict[i]=(1.0-num_wrong_pred_each_dataset[i]/num_total_each_dataset[i])#[datasetNameList[i]]
        print("num of wrong prediction")
        print(num_wrong_pred)
        print("num of test total")
        print(len(Y_test.iloc[0]))
        print("accuracy")
        print(1.0 - num_wrong_pred / len(Y_test.iloc[0]))
    return normalized_pred_out, num_wrong_pred, 1.0 - num_wrong_pred / len(Y_test.iloc[0]),split_accuracy_dict



def print_parameters_settings(code,date,h_dim,toTrainMeiNN, toAddGenePathway,toAddGeneSite, multiDatasetMode,
                                                     datasetNameList,
                                                     num_of_selected_residue,
                                                     lossMode, selectNumPathwayMode,
                                                     num_of_selected_pathway,
                                                     AE_epoch, NN_epoch,
                                                     batch_size_mode,batch_size_ratio,
                                                     separatelyTrainAE_NN, toMask,
                                                     framework, skip_connection_mode,
                                                     split_accuracy_list,total_accuracy,
                                                     split_accuracy_list2,total_accuracy2,
                                                     split_accuracy_list3,total_accuracy3,
                                                     toValidate,multi_task_training_policy,learning_rate_list,preprocess_time,train_time,predict_time):
    print("code,date,h_dim,toTrainMeiNN, toAddGenePathway,toAddGeneSite, multiDatasetMode,\
                                                     datasetNameList,\
                                                     num_of_selected_residue,\
                                                     lossMode, selectNumPathwayMode,\
                                                     num_of_selected_pathway,\
                                                     AE_epoch, NN_epoch,\
                                                     batch_size_mode,batch_size_ratio,\
                                                     separatelyTrainAE_NN, toMask,\
                                                     framework, skip_connection_mode,\
                                                     split_accuracy_list,total_accuracy,\
                                                     split_accuracy_list2,total_accuracy2,\
                                                     split_accuracy_list3,total_accuracy3\
                                                      toValidate,multi_task_training_policy,learning_rate_list,\
                                                       preprocess_time,train_time,predict_time")
    print(code,date,h_dim,toTrainMeiNN, toAddGenePathway,toAddGeneSite, multiDatasetMode,
                                                     datasetNameList,
                                                     num_of_selected_residue,
                                                     lossMode, selectNumPathwayMode,
                                                     num_of_selected_pathway,
                                                     AE_epoch, NN_epoch,
                                                     batch_size_mode,batch_size_ratio,
                                                     separatelyTrainAE_NN, toMask,
                                                     framework, skip_connection_mode,
                                                     split_accuracy_list,total_accuracy,
                                                     split_accuracy_list2,total_accuracy2,
                                                     split_accuracy_list3,total_accuracy3,
                                                    toValidate,multi_task_training_policy,learning_rate_list,
                                                    preprocess_time,train_time,predict_time)


def add_to_result_csv(code,date,h_dim,toTrainMeiNN, toAddGenePathway,toAddGeneSite, multiDatasetMode,
                                                     datasetNameList,
                                                     num_of_selected_residue,
                                                     lossMode, selectNumPathwayMode,
                                                     num_of_selected_pathway,
                                                     AE_epoch, NN_epoch,
                                                     batch_size_mode,batch_size_ratio,
                                                     separatelyTrainAE_NN, toMask,
                                                     framework, skip_connection_mode,
                                                     split_accuracy_list,total_accuracy,
                                                     split_accuracy_list2,total_accuracy2,
                                                     split_accuracy_list3,total_accuracy3,
                                                    toValidate,multi_task_training_policy,learning_rate_list,
                                                    preprocess_time,train_time,predict_time):
    import csv

    input_path ="./result-all/"  # campaign file path
    output_file_name="1-10results-together.csv"
    input_csv = open(input_path + output_file_name, 'a')#original  'ab'
    diabetes1,IBD,MS,Psoriasis,RA,SLE=split_accuracy_list
    single_diabetes1, single_IBD, single_MS, single_Psoriasis, single_RA, single_SLE = split_accuracy_list2
    whole_diabetes1, whole_IBD, whole_MS, whole_Psoriasis, whole_RA, whole_SLE = split_accuracy_list3
    a0=["code","date","h_dim","toTrainMeiNN", "toAddGenePathway","toAddGeneSite", "multiDatasetMode",
                                                     "datasetNameList",
                                                     "num_of_selected_residue",
                                                     "lossMode", "selectNumPathwayMode",
                                                     "num_of_selected_pathway",
                                                     "AE_epoch", "NN_epoch",
                                                     "batch_size_mode","batch_size_ratio",
                                                     "separatelyTrainAE_NN", "toMask",
                                                     "framework", "skip_connection_mode",
                                                     "diabetes1","IBD","MS","Psoriasis","RA","SLE","total_accuracy",
        "single_diabetes1", "single_IBD", "single_MS", "single_Psoriasis", "single_RA", "single_SLE","single_total_accuracy",
        "whole_diabetes1", "whole_IBD", "whole_MS", "whole_Psoriasis", "whole_RA", "whole_SLE",
        "whole_total_accuracy","toValidate","multi_task_training_policy","learning_rate_list","preprocess_time","train_time","predict_time"]
    a = [code,date,h_dim,toTrainMeiNN, toAddGenePathway,toAddGeneSite, multiDatasetMode,
                                                     datasetNameList,
                                                     num_of_selected_residue,
                                                     lossMode, selectNumPathwayMode,
                                                     num_of_selected_pathway,
                                                     AE_epoch, NN_epoch,
                                                     batch_size_mode,batch_size_ratio,
                                                     separatelyTrainAE_NN, toMask,
                                                     framework, skip_connection_mode,
                                                     diabetes1,IBD,MS,Psoriasis,RA,SLE,total_accuracy,
         single_diabetes1, single_IBD, single_MS, single_Psoriasis, single_RA, single_SLE,total_accuracy2,
         whole_diabetes1, whole_IBD, whole_MS, whole_Psoriasis, whole_RA, whole_SLE,total_accuracy3,toValidate,
         multi_task_training_policy,learning_rate_list,preprocess_time,train_time,predict_time]

    csv_write = csv.writer(input_csv, dialect='excel')
    csv_write.writerow(a0)
    csv_write.writerow(a)
    #csv_write.writerow(b)