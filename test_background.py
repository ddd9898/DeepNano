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


ESM2_MODEL = 'esm2_t6_8M_UR50D'
# ESM2_MODEL = 'esm2_t12_35M_UR50D'
# ESM2_MODEL = 'esm2_t30_150M_UR50D'
# ESM2_MODEL = 'esm2_t33_650M_UR50D'

def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--size', dest='size', type=str, default='10w',
                        help='size',metavar='E')
    

    return parser.parse_args()

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


class seqData_background(Dataset):
    def __init__(self,antigen_seq):
        super(seqData_background,self).__init__()
        
        self.seq_data = list()
        
        ##Load sequence data
        nb_backgound = list()
        for fa in SeqIO.parse('./data/INDI/INDI_{}_nanobody.fasta'.format(DATA_SIZE),'fasta'):
            seq = ''.join(list(fa.seq))
            nb_backgound.append(seq)

        ##Antigen: background, from Uniprot P02768
        seq2 = antigen_seq
        # seq2 = 'MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL'
        for n,item in enumerate(nb_backgound):
            seq1 = item
            if len(seq1)>150:
                continue

            self.seq_data.append([seq1,seq2,0])

    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label

args = get_args()
DATA_SIZE = args.size

print('##########################在humanHSA background数据上测试DeepNano-seq(DScriptData)_{}模型：'.format(ESM2_MODEL))
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


###装载测试数据background
testDataset = seqData_background(antigen_seq='MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano_seq(DScriptData)_{}_{}.npy'.format('background({})_humanHSA'.format(DATA_SIZE),ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
##Ensemble
print('True Negative rate = {}'.format(np.sum(p1<0.5)/len(p1)))




print('##########################在humanHSA background数据上测试DeepNano-seq(SabdabData)_{}模型：'.format(ESM2_MODEL))
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


###装载测试数据background
testDataset = seqData_background(antigen_seq='MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano_seq(SabdabData)_{}_{}.npy'.format('background({})_humanHSA'.format(DATA_SIZE),ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
##Ensemble
print('True Negative rate = {}'.format(np.sum(p1<0.5)/len(p1)))




print('##########################在humanHSA background数据上测试DeepNano(SabdabData)_{}模型：'.format(ESM2_MODEL))
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


###装载测试数据background
testDataset = seqData_background(antigen_seq='MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano(SabdabData)_{}_{}.npy'.format('background({})_humanHSA'.format(DATA_SIZE),ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
##Ensemble
print('True Negative rate = {}'.format(np.sum(p1<0.5)/len(p1)))






print('##########################在GST background数据上测试DeepNano-seq(DScriptData)_{}模型：'.format(ESM2_MODEL))
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


###装载测试数据background
testDataset = seqData_background(antigen_seq='MAGKPKLHYFNGRGRMEPIRWLLAAAGVEFEEKFIGSAEDLGKLRNDGSLMFQQVPMVEIDGMKLVQTRAILNYIASKYNLYGKDIKERALIDMYTEGMADLNEMILLLPLCRPEEKDAKIALIKEKTKSRYFPAFEKVLQSHGQDYLVGNKLSRADISLVELLYYVEELDSSLISNFPLLKALKTRISNLPTVKKFLQPGSPRKPPADAKALEEARKIFRF')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano_seq(DScriptData)_{}_{}.npy'.format('background({})_GST'.format(DATA_SIZE),ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
##Ensemble
print('True Negative rate = {}'.format(np.sum(p1<0.5)/len(p1)))




print('##########################在GST background数据上测试DeepNano-seq(SabdabData)_{}模型：'.format(ESM2_MODEL))
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


###装载测试数据background
testDataset = seqData_background(antigen_seq='MAGKPKLHYFNGRGRMEPIRWLLAAAGVEFEEKFIGSAEDLGKLRNDGSLMFQQVPMVEIDGMKLVQTRAILNYIASKYNLYGKDIKERALIDMYTEGMADLNEMILLLPLCRPEEKDAKIALIKEKTKSRYFPAFEKVLQSHGQDYLVGNKLSRADISLVELLYYVEELDSSLISNFPLLKALKTRISNLPTVKKFLQPGSPRKPPADAKALEEARKIFRF')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano_seq(SabdabData)_{}_{}.npy'.format('background({})_GST'.format(DATA_SIZE),ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
##Ensemble
print('True Negative rate = {}'.format(np.sum(p1<0.5)/len(p1)))




print('##########################在GST background数据上测试DeepNano(SabdabData)_{}模型：'.format(ESM2_MODEL))
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


###装载测试数据background
testDataset = seqData_background(antigen_seq='MAGKPKLHYFNGRGRMEPIRWLLAAAGVEFEEKFIGSAEDLGKLRNDGSLMFQQVPMVEIDGMKLVQTRAILNYIASKYNLYGKDIKERALIDMYTEGMADLNEMILLLPLCRPEEKDAKIALIKEKTKSRYFPAFEKVLQSHGQDYLVGNKLSRADISLVELLYYVEELDSSLISNFPLLKALKTRISNLPTVKKFLQPGSPRKPPADAKALEEARKIFRF')
test_loader = DataLoader(testDataset, batch_size=256, shuffle=False)
#Test
g,p_ave,p_min,p_max = predicting(model, device, test_loader,Model_type=3)
p1 = (p_ave+p_min+p_max)/3
np.save('./output/results_DeepNano(SabdabData)_{}_{}.npy'.format('background({})_GST'.format(DATA_SIZE),ESM2_MODEL),[g,p1,p_ave,p_min,p_max])
##Ensemble
print('True Negative rate = {}'.format(np.sum(p1<0.5)/len(p1)))

