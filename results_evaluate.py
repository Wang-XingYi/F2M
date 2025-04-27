import csv
import os
import pickle

from utils import matrixPlot, plotPictrue, auc1, plot_confusion_matrix, get_roc_auc, index_calculation, ROC_curve,plot_confusion_matrixV2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
"""
calculate evaluation indicators
"""


def read_txt_file(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            parts = line.strip().split(' ')
            pre_probability = float(parts[0])
            pre_classes = int(parts[1])
            label = int(parts[2])
            data.append(( pre_probability, pre_classes, label))
    return data

def evaluate_model(data, predictions,model_name):

    trueLabel = []
    labels = ['eczema','others', 'psoriasis']


    for item in data:
        pre_probability, pre_classes, label = item
        trueLabel.append(label)


    predictions_np = [pred.cpu().numpy() for pred in predictions]


    Y_pred = np.array(predictions_np)
    trueLabel = np.array(trueLabel)


    Y_pred_classes = np.argmax(Y_pred, axis=1)


    confusion_mtx = confusion_matrix(trueLabel, Y_pred_classes)

    log = open(f'./logs/01_{model_name}_classify_log.txt', 'w', encoding='utf-8')
    report = classification_report(trueLabel, Y_pred_classes,
                                   target_names=labels ,
                                   digits=3)
    print(report)
    log.write('--------------------F2M-------------------------\n')
    log.write(f"Accuracy：{np.mean(Y_pred_classes == trueLabel):.4f} \n")
    log.write(f'\n{report}\n')

    index_calculation(Y_pred, Y_pred_classes, trueLabel, log,num_classes=3)

    # 根据我们的算法，改成了0,2,1的顺序，因为0的标签代表others
    cm=confusion_matrix(trueLabel, Y_pred_classes,labels=[0,2,1])
    log.write("fusion_matrix: \n{}\n".format(cm))

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.round(cm_normalized, 3)
    print(cm_normalized)
    log.write(f'normalized confusion matrix：\n{cm_normalized}\n')

    df = pd.DataFrame(confusion_mtx, index=labels, columns=['col1', 'col2', 'col3'])
    df.to_csv(f'./logs/01_{model_name}_confusion_matrix.csv', sep='\t', header=False)
    log.close()

if __name__ == '__main__':

    input_file = './logs/dataset_record_log.txt'
    predictions = []
    with open('./logs/pred_file.pkl', 'rb') as f:
        while True:
            try:
                pred = pickle.load(f)
                predictions.append(pred)
            except EOFError:
                break
    data = read_txt_file(input_file)

    evaluate_model(data,predictions,model_name="00_ours")
