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
from models.models import DeepNano_seq,DeepNano
from utils.dataloader import seqData_Dscript
from utils.evaluate import evaluate

# ESM2_MODEL = 'esm2_t6_8M_UR50D'
# ESM2_MODEL = 'esm2_t12_35M_UR50D'
# ESM2_MODEL = 'esm2_t30_150M_UR50D'
ESM2_MODEL = 'esm2_t33_650M_UR50D'

def predicting(model, device, loader, Model_type):
    model.eval()
    total_preds_ave = torch.Tensor()
    total_preds_min = torch.Tensor()
    total_preds_max = torch.Tensor()
    total_labels = torch.Tensor()

    logging.info('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader):
        # for data in loader:
            #Get input
            seqs_nanobody = data[0]
            seqs_antigen = data[1]

            #Calculate output
            g = data[2]

            p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)

            total_preds_ave = torch.cat((total_preds_ave, p_ave.cpu()), 0)
            total_preds_min = torch.cat((total_preds_min, p_min.cpu()), 0)
            total_preds_max = torch.cat((total_preds_max, p_max.cpu()), 0)

            total_labels = torch.cat((total_labels, g), 0)
            
            

    return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten(),total_preds_min.numpy().flatten(),total_preds_max.numpy().flatten()
 

    
###装载训练好的模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import SeqIO
import random
import torch
from torch import nn


class seqData_HSA(Dataset):
    def __init__(self,):
        super(seqData_HSA,self).__init__()
        
        self.seq_data = list()
        
        ##Load pair data
        data = pd.read_csv('./data/case_study/mmc4_HSA.csv').values.tolist()

        ##Antigen: HSA, from Uniprot P02768
        seq2 = 'MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL'
        for n,item in enumerate(data):
            # ID,sequence,cdr1,cdr2,cdr3,lowph,highph,salt = item
            # print(item)
            
            seq1 = item[0]
            elisa = item[1]
            
            # if (not isinstance(seq1,str)) or ('.' not in elisa) :
            #     continue
            # elisa = float(elisa)
            if '/' in elisa:
                continue
            if 'No binding' in elisa:
                elisa = 0
            else:
                elisa = float(elisa)

            
            if len(seq1)>800 or len(seq2)>800:
                continue
            
            self.seq_data.append([seq1,seq2,elisa])

    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label
print(len(seqData_HSA()))


print('##########################在HSA数据上测试DeepNano-seq(DScriptData)_{}模型：'.format(ESM2_MODEL))
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
weights = torch.load(model_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
testDataset = seqData_HSA()
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
# np.save('./compare/D-SCRIPT/my_test/section1/results(PPI_seqMLP_ESM2_Ensemble).npy',[g,p,p_ave,p_min,p_max])
np.save('./output/results_DeepNano_seq(DScriptData)_{}_{}.npy'.format('HSA',ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
# print(p1)

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate((g>0)+0,p1)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))

# find_count = np.sum(p1>0.5)
# print('{}/{} were found.'.format(find_count,len(p1)))

##Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))


print('##########################在HSA数据上测试DeepNano-seq(SabdabData)_{}模型：'.format(ESM2_MODEL))
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
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
testDataset = seqData_HSA()
print(len(testDataset))
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
# np.save('./compare/D-SCRIPT/my_test/section1/results(PPI_seqMLP_ESM2_Ensemble).npy',[g,p,p_ave,p_min,p_max])
np.save('./output/results_DeepNano_seq(SabdabData)_{}_{}.npy'.format('HSA',ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
print(p1)

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate((g>0)+0,p1)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))


# find_count = np.sum(p1>0.5)
# print('{}/{} were found.'.format(find_count,len(p1)))

##Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))



print('##########################在HSA数据上测试DeepNano(SabdabData)_{}模型：'.format(ESM2_MODEL))
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
weights = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
testDataset = seqData_HSA()
print(len(testDataset))
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
# np.save('./compare/D-SCRIPT/my_test/section1/results(PPI_seqMLP_ESM2_Ensemble).npy',[g,p,p_ave,p_min,p_max])
np.save('./output/results_DeepNano(SabdabData)_{}_{}.npy'.format('HSA',ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
print(p1)

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate((g>0)+0,p1)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))


# find_count = np.sum(p1>0.5)
# print('{}/{} were found.'.format(find_count,len(p1)))

##Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))





class seqData_GST(Dataset):
    def __init__(self,):
        super(seqData_GST,self).__init__()
        
        self.seq_data = list()
        
        ##Load pair data
        data = pd.read_csv('./data/case_study/mmc3_GST.csv').values.tolist()
        
        ##Antigen: GST, from Uniprot Q16772
        seq2 = 'MAGKPKLHYFNGRGRMEPIRWLLAAAGVEFEEKFIGSAEDLGKLRNDGSLMFQQVPMVEIDGMKLVQTRAILNYIASKYNLYGKDIKERALIDMYTEGMADLNEMILLLPLCRPEEKDAKIALIKEKTKSRYFPAFEKVLQSHGQDYLVGNKLSRADISLVELLYYVEELDSSLISNFPLLKALKTRISNLPTVKKFLQPGSPRKPPADAKALEEARKIFRF'
        
        for n,item in enumerate(data):   
            # BM_id,BL_id,Sequence,CDR3,CDR1,CDR2,Salt,LowpH,HighpH = item
            seq1 = item[0]
            elisa = item[1]
            
            # if (not isinstance(seq1,str)) or ('.' not in elisa) :
            #     continue
            # elisa = float(elisa)
            if '/' in elisa:
                continue
            if 'No binding' in elisa:
                elisa = 0
            else:
                elisa = float(elisa)
            
            
            
            if len(seq1)>800 or len(seq2)>800:
                continue
            
            self.seq_data.append([seq1,seq2,elisa])

    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label
print(len(seqData_GST()))


print('##########################在GST数据上测试DeepNano-seq(DScriptData)_{}模型：'.format(ESM2_MODEL))
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
weights = torch.load(model_path,map_location=torch.device('cpu')) # 
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
testDataset = seqData_GST()
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano_seq(DScriptData)_{}_{}.npy'.format('GST',ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
# np.save('./compare/D-SCRIPT/my_test/section1/results(PPI_seqMLP_ESM2_Ensemble).npy',[g,p,p_ave,p_min,p_max])
print(p1)

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate((g>0)+0,p1)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))


# find_count = np.sum(p1>0.5)
# print('{}/{} were found.'.format(find_count,len(p1)))

##Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))




print('##########################在GST数据上测试DeepNano-seq(SabdabData)_{}模型：'.format(ESM2_MODEL))
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
weights = torch.load(model_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})

###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
testDataset = seqData_GST()
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano_seq(SabdabData)_{}_{}.npy'.format('GST',ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
# np.save('./compare/D-SCRIPT/my_test/section1/results(PPI_seqMLP_ESM2_Ensemble).npy',[g,p,p_ave,p_min,p_max])
print(p1)

precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate((g>0)+0,p1)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))


# find_count = np.sum(p1>0.5)
# print('{}/{} were found.'.format(find_count,len(p1)))

##Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))



print('##########################在GST数据上测试DeepNano(SabdabData)_{}模型：'.format(ESM2_MODEL))
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
weights = torch.load(model_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')
model.load_state_dict(weights)
# model.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})


###装载测试数据
# testDataset = seqData_NBAT(pair_path='./data/Nanobody_Antigen-main/all_pair_data.csv')
testDataset = seqData_GST()
test_loader = DataLoader(testDataset, batch_size=128, shuffle=False)

#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
# np.save('./compare/D-SCRIPT/my_test/section1/results(PPI_seqMLP_ESM2_Ensemble).npy',[g,p,p_ave,p_min,p_max])
np.save('./output/results_DeepNano(SabdabData)_{}_{}.npy'.format('GST',ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
print(p1)


precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate((g>0)+0,p1)
print("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))

# find_count = np.sum(p1>0.5)
# print('{}/{} were found.'.format(find_count,len(p1)))

##Ensemble
from scipy.stats import pearsonr,spearmanr,kendalltau
r_row,p_value = pearsonr(g,p1)
print('pearsonr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = spearmanr(g,p1)
print('spearmanr: r_row={}, p_value={}'.format(r_row, p_value))
r_row,p_value = kendalltau(g,p1)
print('kendalltau: r_row={}, p_value={}'.format(r_row, p_value))