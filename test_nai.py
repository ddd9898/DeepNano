import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
import logging
# from torch.utils.tensorboard import SummaryWriter 
import random
import os
from torch.utils.data import Dataset
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from tqdm import tqdm
from models.models import baseline, DeepNano_seq,DeepNano
from utils.dataloader import seqData_NBAT_Test
from utils.evaluate import evaluate

ESM2_MODEL = 'esm2_t6_8M_UR50D'
# ESM2_MODEL = 'esm2_t12_35M_UR50D'
# ESM2_MODEL = 'esm2_t30_150M_UR50D'
# ESM2_MODEL = 'esm2_t33_650M_UR50D'


def predicting(model, device, loader, Model_type):
    model.eval()
    total_preds_ave = torch.Tensor()
    total_preds_min = torch.Tensor()
    total_preds_max = torch.Tensor()
    total_labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        # for data in tqdm(loader):
        for data in loader:
            #Get input
            seqs_nanobody = data[0]
            seqs_antigen = data[1]

            #Calculate output
            g = data[2]
            if Model_type == 0:
                predictions = model(seqs_nanobody,seqs_antigen,device)

                total_preds_ave = torch.cat((total_preds_ave, predictions.cpu()), 0)
            elif Model_type == 1:
                p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)
                
                total_preds_ave = torch.cat((total_preds_ave, p_ave.cpu()), 0)
                total_preds_min = torch.cat((total_preds_min, p_min.cpu()), 0)
                total_preds_max = torch.cat((total_preds_max, p_max.cpu()), 0)

            total_labels = torch.cat((total_labels, g), 0)
            
            
    if Model_type == 1:
        return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten(),total_preds_min.numpy().flatten(),total_preds_max.numpy().flatten()
    else:
        return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten()

    
###装载训练好的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print('##########################在nai数据上测试DeepNano-seq(DScriptData){}模型：'.format(ESM2_MODEL))
###装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0).to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0).to(device)

model_dir = './output/checkpoint/'
model_name = 'DeepNano_seq({})_DScriptData_finetune1_best.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path) # map_location=torch.device('cpu')
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})

###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
# testDataset = seqData_NBAT()
testDataset  = seqData_NBAT_Test(seq_path='./data/Nanobody_Antigen-main/all_pair_data.seqs.fasta',
                                pair_path = './data/Nanobody_Antigen-main/all_pair_data.pair.tsv')
test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=1)
p1 = (p_ave+p_min+p_max)/3
# np.save('./output/results_DeepNano_seq(DScriptData)_{}.npy'.format('NBAT'),[g,p1,p_ave,p_min,p_max])

##Ensemble
precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p1)
print("{}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
    'Ensemble',Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))



print('##########################在nai数据上测试DeepNano-seq(SabdabData){}模型：'.format(ESM2_MODEL))
###装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0).to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0).to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model = DeepNano_seq(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0).to(device)

model_dir = './output/checkpoint/'
model_name = 'DeepNano_seq({})_SabdabData_finetune1_TF0_best.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu')) # 
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})

###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
# testDataset = seqData_NBAT()
testDataset  = seqData_NBAT_Test(seq_path='./data/Nanobody_Antigen-main/all_pair_data.seqs.fasta',
                                pair_path = './data/Nanobody_Antigen-main/all_pair_data.pair.tsv')
test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)
print(len(testDataset))

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=1)
p1 = (p_ave+p_min+p_max)/3
# np.save('./output/results_DeepNano_seq(SabdabData)_{}.npy'.format('NBAT'),[g,p1,p_ave,p_min,p_max])

##Ensemble
precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p1)
print("{}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
    'Ensemble',Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))




print('##########################在nai数据上测试DeepNano(SabdabData){}模型：'.format(ESM2_MODEL))
###装载训练好的模型
if ESM2_MODEL == 'esm2_t6_8M_UR50D':
    model = DeepNano(pretrained_model=r'./models/esm2_t6_8M_UR50D',hidden_size=320, finetune=0,
                        Model_BSite_path='./output/checkpoint/DeepNano_site(esm2_t6_8M_UR50D)_SabdabData_finetune1_TF0_best.model').to(device)
if ESM2_MODEL == 'esm2_t12_35M_UR50D':
    model = DeepNano(pretrained_model=r'./models/esm2_t12_35M_UR50D',hidden_size=480,finetune=0,
                        Model_BSite_path='./output/checkpoint/DeepNano_site(esm2_t12_35M_UR50D)_SabdabData_finetune1_TF0_best.model').to(device)
if ESM2_MODEL == 'esm2_t30_150M_UR50D':
    model = DeepNano(pretrained_model=r'./models/esm2_t30_150M_UR50D',hidden_size=640,finetune=0,
                        Model_BSite_path='./output/checkpoint/DeepNano_site(esm2_t30_150M_UR50D)_SabdabData_finetune1_TF0_best.model').to(device)
if ESM2_MODEL == 'esm2_t33_650M_UR50D':
    model = DeepNano(pretrained_model=r'./models/esm2_t33_650M_UR50D',hidden_size=1280,finetune=0,
                        Model_BSite_path='./output/checkpoint/DeepNano_site(esm2_t33_650M_UR50D)_SabdabData_finetune1_TF0_best.model').to(device)
if ESM2_MODEL == 'esm2_t36_3B_UR50D':
    model = DeepNano(pretrained_model=r'./models/esm2_t36_3B_UR50D',hidden_size=2560,finetune=0,
                        Model_BSite_path='./output/checkpoint/DeepNano_site(esm2_t36_3B_UR50D)_SabdabData_finetune1_TF0_best.model').to(device)
if ESM2_MODEL == 'esm2_t48_15B_UR50D':
    model = DeepNano(pretrained_model=r'./models/esm2_t48_15B_UR50D',hidden_size=5120,finetune=0,
                        Model_BSite_path='./output/checkpoint/DeepNano_site(esm2_t48_15B_UR50D)_SabdabData_finetune1_TF0_best.model').to(device)


model_dir = './output/checkpoint/'
model_name = 'DeepNano({})_SabdabData_finetune1_TF0_best.model'.format(ESM2_MODEL)
model_path = model_dir + model_name
weights = torch.load(model_path,map_location=torch.device('cpu')) # 
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
# testDataset = seqData_NBAT()
testDataset  = seqData_NBAT_Test(seq_path='./data/Nanobody_Antigen-main/all_pair_data.seqs.fasta',
                                pair_path = './data/Nanobody_Antigen-main/all_pair_data.pair.tsv')
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=1)
p1 = (p_ave+p_min+p_max)/3
print(len(p1))
# np.save('./output/results_DeepNano(SabdabData)_{}.npy'.format('NBAT'),[g,p1,p_ave,p_min,p_max])

##Ensemble
precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p1)
print("{}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
    'Ensemble',Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))


