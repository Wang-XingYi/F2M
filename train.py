import argparse
import copy
import os
import random
from math import log10

import cv2
import numpy as np
import torch.optim as optim
import torch.utils.data

from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data_utils import TrainDataset, ValDataset
# from loss import GeneratorLoss
from model import Model


loss_function = torch.nn.CrossEntropyLoss()
# name of log
train_log_dir = 'train_log_Oneline-FastDLT'
from torchvision.transforms import Grayscale
exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
exp_train_log_dir = os.path.join(exp_name, train_log_dir)
work_dir = os.path.join(exp_name, 'dataset')
pair_list = list(open(os.path.join(work_dir, 'Val_List.txt')))

LOG_DIR = os.path.join(exp_train_log_dir, 'logs')


writer = SummaryWriter(log_dir=LOG_DIR)
model_save='./train_logs/epochs'
if not os.path.exists(model_save):
    os.makedirs(model_save)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def geometricDistance(correspondence, h):
    """
    Correspondence err
    :param correspondence: Coordinate
    :param h: Homography
    :return: L2 distance
    """

    p1 = np.transpose(np.matrix([correspondence[0][0], correspondence[0][1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([correspondence[1][0], correspondence[1][1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def train(args):
    train_path = os.path.join(exp_name, 'dataset/Train_List.txt')
    val_path = os.path.join(exp_name, 'dataset/Val_List.txt')


    UPSCALE_FACTOR = args.upscale_factor
    NUM_EPOCHS = args.max_epoch

    train_data = TrainDataset(data_path=train_path, exp_path=exp_name, patch_w=args.patch_size_w,
                              patch_h=args.patch_size_h, rho=8, WIDTH=args.img_w, HEIGHT=args.img_h)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.cpus, shuffle=True,
                              drop_last=True)

    val_data = ValDataset(data_path=exp_name, patch_w=args.patch_size_w, patch_h=args.patch_size_h, rho=8,
                          WIDTH=args.img_w, HEIGHT=args.img_h)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=True)

    net=Model()
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))

    if torch.cuda.is_available():
        net.cuda()


    optimizerG = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-4)  # default as 0.0001
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.8)


    results = {'loss': [],'acc': []}
    count = 0

    model_save_fre = 5000
    score_print_fre = 50
    max_ace = 0
    min_loss=15
    glob_iter = 0
    min_loss_val = 15
    best_model = None

    for epoch in range(args.max_epoch):
        train_bar = tqdm(train_loader)
            # Adversarial Loss
        running_results = {'batch_sizes': 0, 'loss': 0, 'acc':0}

        net.train()

        accu_num = torch.zeros(1).cuda()
        sample_num = 0
        print(epoch, 'lr={:.6f}'.format(schedulerG.get_last_lr()[0]))
        for i, batch_value in enumerate(train_bar):
            org_imges = batch_value[0].float()
            labels = batch_value[1]
            sample_num += labels.shape[0]

            if torch.cuda.is_available():
                org_imges = org_imges.cuda()
                labels = labels.cuda()

            batch_size = org_imges.size(0)
            running_results['batch_sizes'] += batch_size

            pred = net(org_imges)
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num = torch.eq(pred_classes, labels).sum()
            loss = loss_function(pred, labels)
            optimizerG.zero_grad()
            g_loss = loss
            g_loss.backward(retain_graph=True)
            optimizerG.step()

            running_results['loss'] += loss.item()
            running_results['acc'] += accu_num.item()



            train_bar.set_description(desc='[%d/%d] Loss: %.4f acc: %.4f' % (
                epoch, NUM_EPOCHS, running_results['loss'] / (i+1),
                running_results['acc'] / sample_num
            ))
            writer.add_scalar('loss', running_results['loss'] / (i+1),count)
            writer.add_scalar('acc', running_results['acc'] / sample_num,count)

            count =count+ 1
            glob_iter += 1
            if g_loss.item() < min_loss_val:
                min_loss_val = g_loss.item()
                best_model = copy.deepcopy(net)
            writer.add_scalar('learning rate', schedulerG.get_last_lr()[0], glob_iter)

        schedulerG.step()
        net.eval()
        val_bar = tqdm(val_loader)
        valing_results = {'acc': 0,  'batch_sizes': 0}
        accu_num = torch.zeros(1).cuda()
        accu_loss = torch.zeros(1).cuda()
        sample_num = 0
        for i, batch_value in enumerate(val_bar):

            org_imges = batch_value[0].float()
            labels = batch_value[1]
            batch_size = org_imges.size(0)
            valing_results['batch_sizes'] += batch_size
            sample_num += org_imges.shape[0]


            with torch.no_grad():
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    org_imges = org_imges.cuda()

                pred = net(org_imges)  # 生成图片
                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels).sum()

                loss = loss_function(pred, labels)
                accu_loss += loss

                val_bar.desc = "[valid epoch {}] loss: {:.3f}, accu_num: {:.3f}".format(epoch,
                                                                                       accu_loss.item() / (i + 1),
                                                                                       accu_num.item() / sample_num)

        # save model parameters
        torch.save(net.state_dict(), './train_logs/epochs/net_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))  # 存储网络参数
        if max_ace < (accu_num.item() / sample_num) and epoch > 10:
            max_ace = accu_num.item() / sample_num
            torch.save(net.state_dict(), f'{model_save}/ace_best_model.pth')

        results['loss'].append(accu_loss.item() / sample_num)
        results['acc'].append(accu_num.item() / sample_num)
        with open(r'./log.txt', 'a', encoding='utf-8') as file:
            file.write(
                f'epoch={epoch},val loss={accu_loss.item() / (i + 1)},val acc={accu_num.item() / sample_num}\n')

    writer.close()
    # 保存最好的模型
    model = best_model
    filename = 'best_' + str(glob_iter) + '.pth'
    model_save_path = os.path.join(model_save, filename)
    torch.save(model.state_dict(), model_save_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    set_seed(3)
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--gpus', type=int, default=2, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=8, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--patch_size_h', type=int, default=224)
    parser.add_argument('--patch_size_w', type=int, default=224)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained waights?')
    parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 2, 4],
                        help='super resolution upscale factor')
    print('<==================== Loading data ===================>\n')

    opt = parser.parse_args()
    print(opt)
    train(opt)