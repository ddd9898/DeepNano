from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
import numpy as np

def evaluate(GT,Pre,thresh = 0.5):
    GT = np.array(GT)
    Pre = np.array(Pre)
    
    #Top 10/20/50
    idx_list = np.argsort(-Pre)
    GT_new = [GT[idx] for idx in idx_list]
    

    Top10 = np.sum(GT_new[:10])/10
    Top20 = np.sum(GT_new[:20])/20
    Top50 = np.sum(GT_new[:50])/50
    
    AUC_ROC = roc_auc_score(GT,Pre)
    precision_list, recall_list, _ = precision_recall_curve(GT, Pre)
    AUC_PR = auc(recall_list, precision_list)

    # Pre = [1 if item>0.5 else 0 for item in Pre]
    Pre = [1 if item>thresh else 0 for item in Pre]
    accuracy = accuracy_score(GT,Pre)
    recall = recall_score(GT,Pre)
    precision = precision_score(GT,Pre)
    F1_score = f1_score(GT,Pre,average='binary')
    
    return precision,recall,accuracy,F1_score,Top10,Top20,Top50,AUC_ROC,AUC_PR
 

def evaluate_site(labels,BSites,thresh = 0.5):
    
    accuracy = 0
    recall = 0
    precision = 0
    F1_score = 0
    for n in range(len(labels)):
        GT = np.array(labels[n])
        Pre = np.array(BSites[n])
        Pre = np.array([1 if item>thresh else 0 for item in Pre])
        
        accuracy += accuracy_score(GT,Pre)
        recall += recall_score(GT,Pre)
        precision += precision_score(GT,Pre)
        F1_score += f1_score(GT,Pre,average='macro')
        
    accuracy = accuracy/len(labels)
    recall = recall/len(labels)
    precision = precision/len(labels)
    F1_score = F1_score/len(labels)

    return accuracy,precision,recall,F1_score
 
