import os.path

import numpy
import torch


from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import cv2
import math
from sklearn.metrics.cluster import  mutual_info_score
threshold=6
def AFRR(img1,img2):
    sift=cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    Accuracy1=0
    length = len(good)
    if length == 0: # 无法提取特征点
        accu1 = 0
        print('没有匹配点')
        return accu1
    for i in range(len(good)):
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小。
        queryIdex=good[i].queryIdx
        trainIdx=good[i].trainIdx
        x1,y1=kp1[queryIdex].pt
        x2,y2=kp2[trainIdx].pt
        x = np.asarray([x1, y1])
        y = np.asarray([x2, y2])
        eucl = Euclidean(x, y)
        if eucl >= 10:
            length -= 1
            continue
        if eucl<threshold:
            Accuracy1 += 1
    # if length <2:
    if length==0: # 特征点都很大
        # return 0
        print('匹配点不足')
        return 0
    accu1 = Accuracy1 / length
    print(f"corrent={Accuracy1},sum={len(good)},accu={accu1}")
    return accu1



# 以特征的方式计算准确率
def distance(img1,img2,index):
    # sift = cv2.xfeatures2d_SIFT.create()
    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)
    Accuracy1=0
    Accuracy2 = 0
    Accuracy3 = 0
    Accuracy4 = 0
    length = len(good)
    if length == 0:
        accu1 = 0
        accu2 = 0
        accu3 = 0
        accu4 = 0
        print(index + '没有匹配点')
        return accu1,accu2,accu3,accu4
    for i in range(len(good)):
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小。
        queryIdex=good[i].queryIdx
        trainIdx=good[i].trainIdx
        x1,y1=kp1[queryIdex].pt
        x2,y2=kp2[trainIdx].pt
        if abs(x1-x2)<=threshold and abs(y1-y2)<=threshold:
            Accuracy1+=1
        x = np.asarray([x1,y1])
        y = np.asarray([x2,y2])
        eucl=Euclidean(x,y)
        man=Manhattan(x,y)
        che=Chebyshev(x,y)
        if eucl<=threshold:
            Accuracy2+=1
        if man<=threshold:
            Accuracy3+=1
        if che<=threshold:
            Accuracy4+=1
    accu1=Accuracy1/len(good)
    accu2=Accuracy2/len(good)
    accu3 = Accuracy3 / len(good)
    accu4 = Accuracy4 / len(good)
    print(f"corrent={Accuracy1},sum={len(good)},accu={accu1}")
    print(f"corrent={Accuracy2},sum={len(good)},accu={accu2}")
    print(f"corrent={Accuracy3},sum={len(good)},accu={accu3}")
    print(f"corrent={Accuracy4},sum={len(good)},accu={accu4}")
    return accu1,accu2,accu3,accu4
    # return accu1

def sift(img1,img2,index,save_path):

    MIN_MATCH_COUNT = 1

    # sift = cv2.xfeatures2d_SIFT.create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    # 显示图像
    if len(good) >= MIN_MATCH_COUNT:

        h1, w1= img1.shape
        h2, w2= img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        # plt.imshow(newimg)
        # plt.show()
        plt.imsave(os.path.join(save_path,"sift_"+index+".tiff"),newimg)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    Accuracy1=0
    length = len(good)
    error=[]
    if length == 0:
        accu1 = 0
        print(index + '没有匹配点')
        return accu1,error
    for i in range(len(good)):
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小。
        queryIdex=good[i].queryIdx
        trainIdx=good[i].trainIdx
        x1,y1=kp1[queryIdex].pt
        x2,y2=kp2[trainIdx].pt
        x = np.asarray([x1, y1])
        y = np.asarray([x2, y2])
        eucl = Euclidean(x, y)
        error.append(eucl)
        # if eucl <= threshold:
        #     Accuracy1 += 1
        if abs(x1-x2)<=threshold and abs(y1-y2)<=threshold:
            Accuracy1+=1
    accu1=Accuracy1/len(good)
    print(f"corrent={Accuracy1},sum={len(good)},accu={accu1}")
    return accu1,error

def ORB(image1,image2,index):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matcher = bf.knnMatch(des1, des2, k=2)
    # Lowe's ratio test
    good = []
    for m, n in matcher:
        if m.distance < 0.7 * n.distance:
            good.append(m)


    # img_mathes = cv2.drawMatches(image1, kp1, image2, kp2, matcher, None, (255, 0, 0))
    #
    # cv2.imwrite(os.path.join(r"F:\DeepHomography_version2\DeepHomography_modify6\images\result", "ORB_"+index + ".bmp"), img_mathes)
    # 显示图像
    MIN_MATCH_COUNT=1
    if len(good) >= MIN_MATCH_COUNT:

        h1, w1 = image1.shape
        h2, w2 = image2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for j in range(3):
            newimg[hdif:hdif + h1, :w1, j] = image1
            newimg[:h2, w1:w1 + w2, j] = image2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        # plt.imshow(newimg)
        # plt.show()
        plt.imsave(
            os.path.join(r"../AFRE_result", "ORB_" + index + ".tiff"),
            newimg)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    correct1=0
    length=len(good)
    error = []
    if length==0:
        accu1 = 0
        print(index + '没有匹配点')
        return accu1,error
    for i in range(length):
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小。
        queryIdex=good[i].queryIdx
        trainIdx=good[i].trainIdx
        x1,y1=kp1[queryIdex].pt
        x2,y2=kp2[trainIdx].pt
        x = np.asarray([x1, y1])
        y = np.asarray([x2, y2])
        eucl = Euclidean(x, y)
        error.append(eucl)
        # if eucl <= threshold:
        #     correct1 += 1
        if abs(x1-x2)<=threshold and abs(y1-y2)<=threshold:
            correct1+=1

    accu1=correct1/length
    print(f"corrent={correct1},sum={length},accu={accu1}")
    return accu1,error


# 还暂时没有测试对不对
def SURF(img1,img2,index):
    minHessian = 1000
    surf = cv2.xfeatures2d_SURF.create()

    keyPoint1, descriptors1 = surf.detectAndCompute(img1, None)
    keyPoint2, descriptors2 = surf.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good = []

    for m, n in matches:

        if m.distance < 0.7* n.distance:
            good.append(m)


    MIN_MATCH_COUNT = 1
    if len(good) >= MIN_MATCH_COUNT:

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for j in range(3):
            newimg[hdif:hdif + h1, :w1, j] = img1
            newimg[:h2, w1:w1 + w2, j] = img2

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(keyPoint1[m.queryIdx].pt[0]), int(keyPoint1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(keyPoint2[m.trainIdx].pt[0] + w1), int(keyPoint2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        # plt.imshow(newimg)
        # plt.show()
        plt.imsave(
            os.path.join(r"../AFRE_result", "SURF_" + index + ".tiff"),
            newimg)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    correct1 = 0
    length = len(good)
    error = []
    if length == 0:
        accu1 = 0
        print(index + '没有匹配点')
        return accu1,error
    for i in range(length):
        # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
        # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
        # .pt:关键点坐标，.angle：表示关键点方向，.response表示响应强度，.size:标书该点的直径大小。
        queryIdex = good[i].queryIdx
        trainIdx = good[i].trainIdx
        x1, y1 = keyPoint1[queryIdex].pt
        x2, y2 = keyPoint2[trainIdx].pt
        x = np.asarray([x1, y1])
        y = np.asarray([x2, y2])
        eucl = Euclidean(x, y)
        error.append(eucl)
        # if eucl <= threshold:
        #     correct1 += 1
        if abs(x1 - x1) <=threshold and abs(y1 - y2) <= threshold:
            correct1 += 1

    accu1 = correct1 / length
    print(f"corrent={correct1},sum={length},accu={accu1}")
    return accu1,error


# 欧式距离
def Euclidean(x,y):
    return np.sqrt(np.sum(np.square(x-y)))

# 曼哈顿距离
def Manhattan(x,y):
    return np.sum(np.abs(x-y))
# 切比雪夫距离
def Chebyshev(x,y):
    return np.max(np.abs(x-y))



def MI(img1,img2):

    # path1=r'F:\PythonSIFT-master\Test\RGB\004.bmp'
    # path2=r'F:\PythonSIFT-master\Test\GT\004.bmp'
    # img1=cv2.imread(path1)
    # img2=cv2.imread(path2)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_ref = np.array(img1, dtype=np.int32)
    img_sen = np.array(img2, dtype=np.int32)
    img_ref=img_ref .reshape(-1)
    img_sen_roi=img_sen .reshape(-1)
    MIValue=mutual_info_score(img_ref, img_sen_roi)
    return MIValue
    # print('MI',MIValue)
def psnr(img1,img2):
    mse=numpy.mean((img1-img2)**2)
    if mse==0:
        return 100
    PIXEL_MAX=255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))



def ssim_new(x, y, size=3):
    # C = (K*L)^2 with K = max of intensity range (i.e. 255). L is very small
    x=torch.from_numpy(x)
    y=torch.from_numpy(y)

    x = np.transpose(x, [2, 1, 0])
    y = np.transpose(y, [2, 1, 0])
    x = x.unsqueeze(0).float()
    y = y.unsqueeze(0).float()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, size, 1, padding=0)
    mu_y = F.avg_pool2d(y, size, 1, padding=0)

    sigma_x = F.avg_pool2d(x ** 2, size, 1, padding=0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, size, 1, padding=0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, size, 1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)
# FIXME there seems to be a problem with this code
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return numpy.mean(ssim_map)

def compareError():
    sift_error= np.load(r'../AFRE_result/sift.npy', allow_pickle=True)
    orb_error = np.load(r'../AFRE_result/orb.npy', allow_pickle=True)
    surf_error = np.load(r'../AFRE_result/surf.npy', allow_pickle=True)
    values1=[]
    values3 = []
    values5=[]
    values7 = []
    values9 = []
    values11=[]
    name='surf'
    for items in surf_error:
        num1=0
        num3=0
        num5=0
        num7=0
        num9=0
        num11=0
        if len(items)==0:
            values1.append(0)
            values3.append(0)
            values5.append(0)
            values7.append(0)
            values9.append(0)
            values11.append(0)
            continue
        for value in items:
            if value<=1:
                num1+=1
            if value<=3:
                num3+=1
            if value<=5:
                num5+=1
            if value<=7:
                num7+=1
            if value<=9:
                num9+=1
            if value<=11:
                num11+=1
        length=len(items)
        values1.append(num1/length)
        values3.append(num3 / length)
        values5.append(num5 / length)
        values7.append(num7 / length)
        values9.append(num9 / length)
        values11.append(num11/length)

    print()
    np.savetxt(os.path.join(r'../AFRE_result', name+'_1.txt'), values1, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_3.txt'), values3, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_5.txt'), values5, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_7.txt'), values7, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_9.txt'), values9, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_11.txt'), values11, fmt='%s', newline='\n')


def erroeMatchRate():
    sift_error= np.load(r'../AFRE_result/sift.npy', allow_pickle=True)
    orb_error = np.load(r'../AFRE_result/orb.npy', allow_pickle=True)
    surf_error = np.load(r'../AFRE_result/surf.npy', allow_pickle=True)
    values1=[]
    values3 = []
    values5=[]
    values7 = []
    values9 = []
    values11=[]
    name='surf'
    for items in surf_error:
        num1=0
        num3=0
        num5=0
        num7=0
        num9=0
        num11=0
        if len(items)==0:
            values1.append(0)
            values3.append(0)
            values5.append(0)
            values7.append(0)
            values9.append(0)
            values11.append(0)
            continue
        for value in items:
            if value>=12:
                num1+=1
            if value>=16:
                num3+=1
            if value>=20:
                num5+=1
            if value>=32:
                num7+=1
            if value>=48:
                num9+=1
            if value>=64:
                num11+=1
        length=len(items)
        values1.append(num1/length)
        values3.append(num3 / length)
        values5.append(num5 / length)
        values7.append(num7 / length)
        values9.append(num9 / length)
        values11.append(num11/length)

    print()
    np.savetxt(os.path.join(r'../AFRE_result', name+'_12.txt'), values1, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_16.txt'), values3, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_20.txt'), values5, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_32.txt'), values7, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_48.txt'), values9, fmt='%s', newline='\n')
    np.savetxt(os.path.join(r'../AFRE_result', name + '_64.txt'), values11, fmt='%s', newline='\n')

if __name__ == '__main__':
    # img1=cv2.imread(r'../images/result_image/0_GT.tiff')
    # img2=cv2.imread(r'../images/result_image/0_pre.tiff')
    # img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #
    # result=AFRE(img1,img2,'1')
    # # print(result)
    # sift(img1,img2,'1')
    # ORB(img1, img2, '1')
    # SURF(img1, img2, '1')

    # compareError()
    erroeMatchRate()

    # path1=r'../images/result'
    # figs=os.listdir(path1)
    # # GT_ACC=[]
    # # pre_ACC=[]
    # ACC=[]
    # HVCD=[]
    # EUCL = []
    # MAN = []
    # CHE = []
    # DValue=[]
    # for i in range(42):
    #     index=str(i)
    #     print(i)
    #     GT_path=os.path.join(path1,index+'_GT.tiff')
    #     pre_path = os.path.join(path1, index + '_pre.tiff')
    #     img1=cv2.imread(pre_path)
    #     img2=cv2.imread(GT_path)
    #     img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #     acc,error=SURF(img1,img2,index)
    #     ACC.append(acc)
    #     DValue.append(error)
    # np.save(os.path.join(r'../AFRE_result',"surf"),DValue)
    #     # hvcd,eucl,man,che=distance(img1,img2,index)
    #     # HVCD.append(hvcd)
    #     # EUCL.append(eucl)
    #     # MAN.append(man)
    #     # CHE.append(che)
    # print(np.mean(ACC))
    #
    # # h=np.mean(HVCD)
    # # e=np.mean(EUCL)
    # # m=np.mean(MAN)
    # # c=np.mean(CHE)
    # # print(f'HVCD={h},EUCL={e},MAN={m},CHE={c}')

