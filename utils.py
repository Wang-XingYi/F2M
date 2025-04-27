import itertools

from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from evaluate import FusionMatrix
from torch.nn import init
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 10})
classNum=3
rename='mamba'

def auc1(trueLabel,abiliable,classes=classNum):
    tempTrueLabel=[0]*len(trueLabel)
    tempAbiliable=[0]*len(trueLabel)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        i+=1
        for j in range(len(trueLabel)):
            if trueLabel[j]==i:
                tempTrueLabel[j]=1
            tempAbiliable[j]=abiliable[j][i-1]
        fpr[i-1], tpr[i-1], thresholds = roc_curve(tempTrueLabel, tempAbiliable, pos_label=1)
        roc_auc[i-1]=auc(fpr[i-1], tpr[i-1])
        tempTrueLabel = [0] * len(trueLabel)
        tempAbiliable = [0] * len(trueLabel)
    return fpr,tpr,roc_auc
def plotPictrue(fpr,tpr,roc_auc,model_name):
    lw = 2
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue','red','blue','green','black','bisque','burlywood','antiquewhite','tan','navajowhite',
     'goldenrod','gold','khaki','ivory','forestgreen','limegreen',
     'springgreen','lightcyan','teal','royalblue',
     'navy','slateblue','indigo','darkorchid','darkviolet','thistle']
    save_name=['./logs/'+model_name+'-Algorithm-to-0-3-class-disease-sizeClass.png']
    lens=1
    for temp_save_name in save_name:
        for i in range(classNum):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,label='ROC curve of class{0} (AUC area = {1:0.2f})'.format(str(i), roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Sensitivity')
        plt.ylabel('Specificity')
        plt.title(rename)
        plt.legend(loc="lower right")
        plt.savefig(temp_save_name, format='png')
        plt.clf()
        lens+=1
    #plt.show()

def ROC_curve(y_true,y_scores,model_name):
    # 将实际标签二值化（One-vs-Rest），即每个类别的标签独立成一列
    y_true_binary = label_binarize(y_true, classes=[0, 1, 2])

    # 初始化绘图
    plt.figure()

    # 针对每个类别绘制 ROC 曲线
    # for i in range(3):  # 这里假设类别是 0、1 和 2
    #     # 计算每个类别的 FPR 和 TPR
    #     fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
    #     roc_auc = auc(fpr, tpr)
    #
    #     # 绘制 ROC 曲线
    #     plt.plot(fpr, tpr, lw=2, label=f'ROC curve for class {i} (area = {roc_auc:.2f})')
    # 存储每个类别的 FPR、TPR 和 AUC 值
    fpr_list = []
    tpr_list = []
    auc_list = []

    # 对每个类别进行 One-vs-Rest ROC 曲线计算
    for i in range(3):  # 对于每个类别
        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc(fpr, tpr))

    # 计算平均 AUC
    mean_auc = np.mean(auc_list)

    # 对三个类别的 FPR 和 TPR 求平均（简单处理方式）
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= 3  # 平均 TPR

    # 绘制平均 ROC 曲线
    plt.plot(all_fpr, mean_tpr, lw=2, label=f'algo_name (AUC = {mean_auc:.3f})')
    # 保存 all_fpr 和 mean_tpr 为 txt 文件
    np.savetxt(f'./logs/02_{model_name}_all_fpr.txt', all_fpr, fmt='%.5f')
    np.savetxt(f'./logs/02_{model_name}_mean_tpr.txt', mean_tpr, fmt='%.5f')


    # 添加对角线参考线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置图形参数
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'./logs/{model_name}_AUC_curve.png', format='png')
    # plt.show()

def matrixPlot(imagesModelRes,trueLabel,model_name):
    cm=confusion_matrix(trueLabel, imagesModelRes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(rename)
    plt.colorbar()
    labels_name=[ str(i) for i in range(classNum)]
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./logs/{model_name}_confusion_matrix.png', format='png')
    # plt.show()
def plot_confusion_matrix(cm,model_name, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    decoy = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > decoy else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./logs/{model_name}_confusion_matrixV2.png', format='png')


def plot_confusion_matrixV2(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # y_pred = y_pred.argmax(axis=1)
    # y_true = y_true.argmax(axis=1)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f'./logs/confusion_matrixV22.png', format='png')
    return ax
def get_roc_auc(all_preds, all_labels):
    one_hot = label_to_one_hot(all_labels, all_preds.shape[1])

    fpr = {}
    tpr = {}
    roc_auc = np.zeros([all_preds.shape[1]])
    for i in range(all_preds.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(one_hot[:, i], all_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc


def label_to_one_hot(label, num_class):
    one_hot = F.one_hot(torch.from_numpy(label).long(), num_class).float()
    one_hot = one_hot.numpy()

    return one_hot


def index_calculation(all_preds, all_result, all_labels, log, num_classes=3):
    fusion_matrix = FusionMatrix(num_classes)
    fusion_matrix.update(all_result, all_labels)
    roc_auc = get_roc_auc(all_preds, all_labels)

    metrics = {}
    metrics["sensitivity"] = fusion_matrix.get_rec_per_class()
    metrics["specificity"] = fusion_matrix.get_pre_per_class()
    metrics["f1_score"] = fusion_matrix.get_f1_score()
    metrics["roc_auc"] = roc_auc
    metrics["fusion_matrix"] = fusion_matrix.matrix

    metrics["acc"] = fusion_matrix.get_accuracy()
    metrics["bacc"] = fusion_matrix.get_balance_accuracy()
    auc_mean = np.mean(metrics["roc_auc"])
    spec_mean = np.mean(metrics["specificity"])
    sens_mean = np.mean(metrics["sensitivity"])

    print("\n-------  Valid result: Valid_Acc: {:>6.3f}%  Balance_Acc: {:>6.3f}%  -------".format(
        metrics["acc"] * 100, metrics["bacc"] * 100))
    log.write(("\n-------  Valid result: Valid_Acc: {:>6.3f}%  Balance_Acc: {:>6.3f}%  -------\n".format(
        metrics["acc"] * 100, metrics["bacc"] * 100)))

    print("         roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}     ".format(
        auc_mean, metrics["f1_score"]))
    log.write("roc_auc.mean: {:>6.3f}  f1_score: {:>6.4f}\n".format(
        auc_mean, metrics["f1_score"]))

    print("         roc_auc:       {}  ".format(metrics["roc_auc"]))
    log.write("roc_auc:       {}\n".format(metrics["roc_auc"]))

    print("sensitivity:   {}   mean:   {}\n".format(metrics["sensitivity"],sens_mean))
    log.write("sensitivity:   {}   mean:   {}\n".format(metrics["sensitivity"],sens_mean))

    print("         specificity:   {}   mean:   {}  ".format(metrics["specificity"], spec_mean))
    log.write("specificity:   {}   mean:   {}\n".format(metrics["specificity"], spec_mean))

    print("         fusion_matrix: \n{}  ".format(metrics["fusion_matrix"]))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))  # 通过平均池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        max_out = self.mlp(self.max_pool(x))  # 通过最大池化压缩全局空间信息: (B,C,H,W)--> (B,C,1,1) ,然后通过MLP降维升维:(B,C,1,1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        x = self.conv1(x)  # 通过卷积层得到注意力权重:(B,2,H,W)-->(B,1,H,W)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, inp, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inp, ratio)
        self.sa = SpatialAttention(kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = x * self.ca(x)  # 通过通道注意力机制得到的特征图,x:(B,C,H,W),ca(x):(B,C,1,1),out:(B,C,H,W)
        result = out * self.sa(out)  # 通过空间注意力机制得到的特征图,out:(B,C,H,W),sa(out):(B,1,H,W),result:(B,C,H,W)
        return result