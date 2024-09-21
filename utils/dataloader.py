from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import SeqIO
import random
from scipy.spatial.transform import Rotation as R
import torch
from torch import nn

class seqData_Dscript(Dataset):
    def __init__(self, pair_path, seqs_path, addNeg=True,augment=False):
        super(seqData_Dscript,self).__init__()
        
        self.seq_data = list()
        self.augment = augment
        
        ##Load sequence data
        PPI_seq_dict = dict()
        for fa in SeqIO.parse(seqs_path,'fasta'):
            ID = fa.description
            seq = ''.join(list(fa.seq))
            PPI_seq_dict[ID] = seq
        
        ##Load pair data
        data = pd.read_csv(pair_path,sep='\t',header=None).values.tolist()
        
        for item in data:
            ID1,ID2,label = item
            
            if (not addNeg) and label == 0:
                continue
            
            seq1 = PPI_seq_dict[ID1]
            seq2 = PPI_seq_dict[ID2]
            
            if len(seq1)>800 or len(seq2)>800:
                    continue
            
            self.seq_data.append([seq1,seq2,label])


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        if not self.augment:
            return seq1,seq2,label
        else:
            p = random.random()
            if p<0.5:
                return seq1,seq2,label
            else:
                return seq2,seq1,label

    
    

class seqData_NBAT(Dataset):
    def __init__(self, pair_path, split_path, data_split='0-train', addNeg=True):
        super(seqData_NBAT,self).__init__()
        
        self.seq_data = list()

        idx, split = data_split.split('-')
        
        ##Load pair data
        data = pd.read_csv(pair_path).values.tolist()
        split_data = pd.read_csv(split_path).values.tolist()[int(idx)]
        
        for n,item in enumerate(data):
            label,seq1,seq2 = item
            if label == 'Yes':
                label = 1
            else:
                label = 0
            
            if (not addNeg) and label == 0:
                continue

            if (split=='train' and split_data[n]==0) or (split=='test' and split_data[n]==1):
                self.seq_data.append([seq1,seq2,label])


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label

class seqData_NBAT_Test(Dataset):
    def __init__(self,seq_path,pair_path):
        super(seqData_NBAT_Test,self).__init__()
        
        self.seq_data = list()
        
        ##Load all PPI sequence
        PPI_seq_dict = dict()
        # for fa in SeqIO.parse('./data/Nanobody_Antigen-main/all_pair_data.seqs.fasta','fasta'):
        for fa in SeqIO.parse(seq_path,'fasta'):
            ID = fa.description
            seq = ''.join(list(fa.seq))
            PPI_seq_dict[ID] = seq
        
        ##Load pair data
        # data = pd.read_csv('./data/Nanobody_Antigen-main/all_pair_data.pair.tsv',sep='\t').values.tolist()
        data = pd.read_csv(pair_path,sep='\t',header=None).values.tolist()
        
        for n,item in enumerate(data):
            ID1, ID2, label = item
            seq1 = PPI_seq_dict[ID1]
            seq2 = PPI_seq_dict[ID2]
            
            if len(seq1)>800 or len(seq2)>800:
                continue
            
            self.seq_data.append([seq1,seq2,label])

    def __len__(self):
        return len(self.seq_data)
    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]
       
        return seq1,seq2,label


class seqData_Sabdab(Dataset):
    def __init__(self, pair_path):
        super(seqData_Sabdab,self).__init__()
        
        data = pd.read_csv(pair_path).values.tolist()
        self.seq_data = list()
        for item in data:
            ID1,seq1,ID2,seq2,label = item

            self.seq_data.append([seq1,seq2,label])


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self,i):
        seq1,seq2,label = self.seq_data[i]

        return seq1,seq2,label

class infaData_Sabdab(Dataset):
    def __init__(self, pair_path, augment = False):
        super(infaData_Sabdab,self).__init__()
        
        data = pd.read_csv(pair_path).values.tolist()
        self.seq_data = list()
        for item in data:
            _,_,seq1,_,_,seq2,BSite2,_,_ = item
            BSite2 = self.__augmentBSite(BSite2,seq2,augment=augment)
            self.seq_data.append([seq1,seq2,1,BSite2])
        

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self,i):
        seq1,seq2,label,BSite2 = self.seq_data[i]
       
        return seq1,seq2,label,BSite2
    
    def __augmentBSite(self,BSite2,seq2,augment=True):
        BSite2 = [int(idx) for idx in BSite2.split(',')]
        ##对结合界面进行数据增广
        max_len = len(seq2)
        if augment:
            num_add = random.choice([0,1,2,3,4,5])
        else:
            num_add = 0
        add_idxs = list()
        for idx in BSite2:
            for new_idx in range(idx-num_add,idx+num_add+1):
                if new_idx>=0 and new_idx<max_len and new_idx not in BSite2:
                    add_idxs.append(new_idx)
        BSite2.extend(add_idxs)

        temp = np.zeros(len(seq2))
        for idx in BSite2:
            temp[idx] = 1
        BSite2 = list(temp)

        return BSite2



def collate_fn_infaData(batch):
    
    seq1_batch = [item[0] for item in batch]
    
    seq2_batch = [item[1] for item in batch]
    
    label = torch.FloatTensor([item[2] for item in batch])

    
    
    max_len = max([len(item[3]) for item in batch])
    aa_BSite2_batch = list()
    for item in batch:
        BSite2 = torch.LongTensor(np.pad(item[3],(0,max_len-len(item[3])),'constant')).unsqueeze(0)
        aa_BSite2_batch.append(BSite2)

    aa_BSite2_batch = torch.cat(aa_BSite2_batch,dim=0)
    

    return seq1_batch,seq2_batch,label,aa_BSite2_batch
