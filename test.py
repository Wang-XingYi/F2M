import argparse
import os
import pickle
import random
import torch.nn.functional as F
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TestDataset,ValDataset
from model import Model




def test(args):
    UPSCALE_FACTOR = args.upscale_factor  # 上采样
    # NUM_EPOCHS = args.num_epochs #轮数
    exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

    net = Model().eval()
    if torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load('./train_logs/epochs/ace_best_model.pth'))

    test_data = TestDataset(data_path=exp_name,dataset_txt=args.dataset_txt,patch_w=args.patch_size_w,
                            patch_h=args.patch_size_h, rho=8, WIDTH=args.img_w, HEIGHT=args.img_h)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=8, shuffle=False, drop_last=False)
    # val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    epoch = 1
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    val_bar = tqdm(test_loader)  # 验证集的进度条
    sample_num = 0
    accu_num = torch.zeros(1).cuda()  # 累计预测正确的样本数
    f = open(args.save_txt, "w", encoding='utf-8')
    f.write(f'pre_probability pre_classes label\n')
    pred_file=open(args.pred_file, 'wb')
    for i, batch_value in enumerate(val_bar):
        org_imges = batch_value[0].float()
        labels = batch_value[1]
        batch_size = org_imges.size(0)
        sample_num += org_imges.shape[0]

        with torch.no_grad():
            if torch.cuda.is_available():
                labels = labels.cuda()
                org_imges = org_imges.cuda()
            pred = net(org_imges)  # 生成图片
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            val_bar.desc = "[valid epoch {}] accu: {:.3f}".format(epoch, accu_num.item() / sample_num)
            # 将 logits 转换为概率
            probabilities = F.softmax(pred, dim=1)
            # 获取最大概率值
            max_probability, max_index = torch.max(probabilities, dim=1)
            for j in range(batch_size):
                f.write(f'{max_probability[j].item()} {max_index[j].item()} {labels[j]}\n')
                pickle.dump(pred[j].cpu(), pred_file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False          # 是否优化运行速度
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    set_seed(3)  # 设置随机数种子
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=2, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=8, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--patch_size_h', type=int, default=224)
    parser.add_argument('--patch_size_w', type=int, default=224)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')

    parser.add_argument('--model_name', type=str, default='mobilenetV4')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--dataset_txt', type=str, default='Test_List.txt')
    parser.add_argument('--save_txt', type=str, default="./logs/dataset_record_log.txt")
    parser.add_argument('--pred_file', type=str, default="./logs/pred_file.pkl")

    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained waights?')
    parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained waights?')
    parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 2, 4],
                        help='super resolution upscale factor')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
