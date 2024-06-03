import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import logging
# from torch.utils.tensorboard import SummaryWriter
import random
import os
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from models.models import DeepNano_seq,DeepNano
from utils.dataloader import split_Train_Test,seqData_NBAT_Test
from utils.evaluate import evaluate

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import SeqIO
import random
from scipy.spatial.transform import Rotation as R
import torch
from torch import nn



'''
1. DeepNano_seq
CUDA_VISIBLE_DEVICES=0 python train_Sabdab.py --Model 0 --finetune 1 &


2. DeepNano
CUDA_VISIBLE_DEVICES=0 python train_Sabdab.py --Model 1 --finetune 1 &

'''

class seqData_Sabdab(Dataset):
    def __init__(self, data, addNeg=True,flip=False):
        super(seqData_Sabdab,self).__init__()
        
        # data = pd.read_csv(pair_path).values.tolist()
        self.seq_data = list()
        for item in data:
            _,_,seq1,_,_,seq2,_,_,_ = item
            if len(seq1)>800 or len(seq2)>800:
                continue
            self.seq_data.append([seq1,seq2,1])

        ##Add negative
        seq_data_neg = list()
        candidates = list(range(len(self.seq_data)))
        if addNeg:
            for idx1 in candidates:
                for t in range(10):
                    idx2 = random.choice(candidates)
                    seq_data_neg.append([self.seq_data[idx2][0],self.seq_data[idx1][1],0])
                
            self.seq_data.extend(seq_data_neg)
            
        ##Flip
        if flip:
            flip_pairs = list()
            for item in self.seq_data:
                seq1, seq2, label = item
                flip_pairs.append([seq2,seq1,label])
            self.seq_data.extend(flip_pairs)


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]

        return seq1,seq2,label


def get_args():
    parser = argparse.ArgumentParser(description='Train the model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  

    parser.add_argument('--Model', dest='Model', type=int, default=0,
                        help='Model',metavar='E')
    
    parser.add_argument('--finetune', dest='finetune', type=int, default=1,
                        help='finetune',metavar='E')
    
    parser.add_argument('--pretrained', dest='pretrained', type=str, default=None,
                        help='pretrained',metavar='E')
    

    return parser.parse_args()



def train(model, device, train_loader, optimizer, epoch, Model_type):
    '''
    training function at each epoch
    '''
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    logging.info('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #Get input
        seqs_nanobody = data[0]
        seqs_antigen = data[1]
        
        #Calculate output
        optimizer.zero_grad()
        
        p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)
        
        ###Calculate loss
        gt = data[2].float().to(device)
        loss1 = F.binary_cross_entropy(p_ave.squeeze(),gt)
        loss2 = F.binary_cross_entropy(p_min.squeeze(),gt)
        loss3 = F.binary_cross_entropy(p_max.squeeze(),gt)
        
        loss = (loss1 + loss2 + loss3)/3

        train_loss = train_loss + loss.item()

        #Optimize the model
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                            batch_idx * BATCH_SIZE,
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss.item()))
    train_loss = train_loss / len(train_loader)
    return train_loss


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
            p_ave, p_min, p_max = model(seqs_nanobody,seqs_antigen,device)
            
            total_preds_ave = torch.cat((total_preds_ave, p_ave.cpu()), 0)
            total_preds_min = torch.cat((total_preds_min, p_min.cpu()), 0)
            total_preds_max = torch.cat((total_preds_max, p_max.cpu()), 0)
            
            g = data[2]
            total_labels = torch.cat((total_labels, g), 0)

    return total_labels.numpy().flatten(),total_preds_ave.numpy().flatten(),total_preds_min.numpy().flatten(),total_preds_max.numpy().flatten()




def set_seed(seed = 1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


if __name__ == '__main__':
    
    set_seed()

    #Train setting
    BATCH_SIZE = 32
    LR = 0.00005 
    LOG_INTERVAL = 20000 
    NUM_EPOCHS = 10 
    
        
    #Get argument parse
    args = get_args()

    if args.Model == 0:
        model_name = 'DeepNano_seq(test_flip)'
    elif args.Model == 1:
        model_name = 'DeepNano(test_flip)'

    #Set log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Output name
    add_name = '(esm2_t6_8M_UR50D)_SabdabData_finetune{}_TF{}'.format(args.finetune,(args.pretrained is not None)+0)
    
    
    logfile = './output/log/log_' + model_name + add_name + '.txt'
    fh = logging.FileHandler(logfile,mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    
    #Step 1:Prepare dataloader
    trainData,valData = split_Train_Test(data_path='./data/Sabdab/all_binding_site_data_5A.csv',test_ratio=0.05)

    trainDataset = seqData_Sabdab(trainData,addNeg=True,flip=True)
    valDataset   = seqData_Sabdab(valData,addNeg=True,flip=False)
    testDataset  = seqData_NBAT_Test(seq_path='./data/Nanobody_Antigen-main/all_pair_data.seqs.fasta',
                                pair_path = './data/Nanobody_Antigen-main/all_pair_data.pair.tsv')
    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True,drop_last=True)
    val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)
    test_loader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)

    #Step 2: Set  model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.Model == 0:
        model = DeepNano_seq(finetune=args.finetune).to(device)
    elif args.Model == 1:
        model = DeepNano(finetune=args.finetune).to(device)


    ##Load pretrained models
    if args.pretrained is not None:
        model_dir = './output/checkpoint/' 
        model_path = model_dir + args.pretrained
        weights = torch.load(model_path,map_location=torch.device('cpu')) # map_location=torch.device('cpu')

        model.load_state_dict(weights)
        # ##Get model params 
        # model2_dict = model.state_dict()
        # state_dict = {k:v for k,v in weights.items() if k in model2_dict.keys()}
        # model2_dict.update(state_dict)
        # model.load_state_dict(model2_dict)

    #Step 3: Train the model
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01) #0.001
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.0001)

                                                    
    logging.info(f'''Starting training:
    Model_name:      {model_name}
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LR}
    Training size:   {len(trainDataset)}
    Validating size: {len(valDataset)}
    Testing size:    {len(testDataset)}
    Finetune:        {args.finetune}
    Pretrained:      {args.pretrained}
    Device:          {device.type}
    ''')
    

    best_AUC_PR = -1
    best_epoch = 0
    model_file_name =  './output/checkpoint/' + model_name + add_name

    for epoch in range(NUM_EPOCHS):
        #Train
        train_loss = train(model, device, train_loader, optimizer, epoch, args.Model)

        ## Val
        g,p_ave,p_min,p_max = predicting(model, device, val_loader, args.Model)
        #ensemble
        p = (p_ave+p_min+p_max)/3

        precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR = evaluate(g,p)
        logging.info("Val: epoch {}: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
            epoch,Top10,Top20,Top50,accuracy,recall,precision,F1_score,AUC_ROC,AUC_PR))

        #Save model
        if best_AUC_PR<AUC_PR:
            best_AUC_PR = AUC_PR
            best_epoch = epoch
            #Save model
            torch.save(model.state_dict(), model_file_name +'_best.model')
            
            ## Test
            g,p_ave,p_min,p_max = predicting(model, device, test_loader, args.Model)
            #ensemble
            p = (p_ave+p_min+p_max)/3
           
            precision_test,recall_test,accuracy_test,F1_score_test,Top10_test,Top20_test,Top50_test,AUC_ROC_test,AUC_PR_test = evaluate(g,p)
            
        logging.info("Best val epoch {} for ensemble with AUC_PR = {:.4f}".format(best_epoch,best_AUC_PR))
        logging.info("Test: Top10 = {:.4f},Top20 = {:.4f},Top50 = {:.4f},accuracy={:.4f},Recall = {:.4f},Precision={:.4f},F1 score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                Top10_test,Top20_test,Top50_test,accuracy_test,recall_test,precision_test,F1_score_test,AUC_ROC_test,AUC_PR_test))





            


            

        