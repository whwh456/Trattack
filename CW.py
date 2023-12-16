#import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import models
import os
import PIL as Image

# import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def show_images_diff(original_img, original_label, adversarial_img, adversarial_label):
    plt.figure()
    # 归一化
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    if adversarial_img.any() > 1.0:
        adversarial_img = adversarial_img / 255.0
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')
    plt.subplot(132)
    plt.title(adversarial_img)
    plt.title('Adversarial')
    plt.imshow(adversarial_img)
    plt.axis('off')
    plt.subplot(133)
    plt.title('Adversarial-Original')
    difference = adversarial_img - original_img
    # (-1,1)->(1,0)
    difference = difference / abs(difference).max() / 2.0 + 0.5
    plt.imshow(difference, cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_images(original_img):
    plt.figure()
    # 归一化
    if original_img.any() > 1.0:
        original_img = original_img / 255.0
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 数据集地址
images = os.listdir('D:\pycharm\CWattack\pic\MNIST/test/')
length = len(images)

# 选择模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.alexnet(pretrained=True).to(device).eval()

# 设定最大迭代次数、学习率、二分查找最大迭代次数、c的初始值、k值等参数
max_iterations = 1000  # 1000次可以完成95%的优化工作
learning_rate = 0.01
binary_search_steps = 10
initial_const = 1e2
confidence = initial_const
k = 40

# 像素值区间
boxmin = -3.0
boxmax = 3.0
num_labels = 1000  # 类别数

# 攻击目标标签，必须使用one hot编码
target_label = 6
tlab = Variable(torch.from_numpy(np.eye(num_labels)[target_label]).to(device).float())
# np.eye 生成对角矩阵
shape = (1, 3, 224, 224)
# 均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 初始化c的边界
lower_bound = 0
c = initial_const
upper_bound = 1e10
# 保存最佳的l2值、预测概率值和对抗样本
o_bestl2 = 1e10
o_bestscore = -1
o_bestattack = [np.zeros(shape)]
# 根据像素值的边界计算标准差boxmul和均值boxplus，把对抗样本转换成新的输入newimg
boxmul = (boxmax - boxmin) / 2.0
boxplus = (boxmin + boxmax) / 2.0

# 每次导入一张图片进行CW攻击
for i in range(length):
    #   图片信息导入以及格式转换
    #img = cv2.imread('D:\pycharm\CWattack\pic\MNIST/test/' + images[i])  # uint8(64,64,3)
    #img2 = np.asarray(img, dtype='float32')  # numpy (64,64,3)BGR
    img2 = img2[..., ::-1]  # (64,64,3)RGB
    # img2 = np.asarray(img2,dtype='float32')
    #img2 = cv2.resize(img2, (224, 224))
    img1 = img2.copy().astype(np.float32)
    img1 /= 255.0
    img1 = (img1 - mean) / std
    img1 = img1.transpose(2, 0, 1)
    img1 = np.expand_dims(img1, axis=0)

    # CW攻击过程
    for outer_step in range(binary_search_steps):
        print("o_bestl2={} confidence={}".format(o_bestl2, confidence))
        # 把原始图像转换成图像数据和扰动的形态
        timg = Variable(torch.from_numpy(np.arctanh((img1 - boxplus) / boxmul * 0.99999)).to(device).float())
        modifier = Variable(torch.zeros_like(timg).to(device).float())
        # 图像数据的扰动量梯度可以获取
        modifier.requires_grad = True
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)  # 优化器
        for iteration in range(1, max_iterations + 1):
            optimizer.zero_grad()
            # 定义新输入
            newimg = torch.tanh(modifier + timg) * boxmul + boxplus
            output = model(newimg)
            # 定义cw中的损失函数  loss2：计算对抗样本和原始数据之间的距离
            loss2 = torch.dist(newimg, (torch.tanh(timg) * boxmul + boxplus), p=2)
            # loss1：挑选指定分类标签和剩下其它分类中概率最大者，计算两者之间概率差
            real = torch.max(output * tlab)
            other = torch.max((1 - tlab) * output)
            loss1 = other - real + k
            loss1 = torch.clamp(loss1, min=0)  # 限制范围，截取功能
            loss1 = confidence * loss1
            # 计算总的损失函数
            loss = loss1 + loss2
            loss.backward(retain_graph=True)
            optimizer.step()
            l2 = loss2
            sc = output.data.cpu().numpy()
            # 显示当前运行结果
            if (iteration % (max_iterations // 10) == 0):
                print("iteration={} loss={} loss1={} loss2={}".format(
                    iteration, loss, loss1, loss2))
            if (l2 < o_bestl2) and (np.argmax(sc) == target_label):
                print('attack success l2={} target_label={}'.format(l2, target_label))
                o_bestl2 = l2
                o_bestscore = np.argmax(sc)
                o_bestattack = newimg.data.cpu().numpy()

        # 叠加噪声之后的攻击样本
        adv = o_bestattack[0]
        adv = adv.transpose(1, 2, 0)
        adv = (adv * std) + mean
        adv = adv * 255.0
        adv = np.clip(adv, 0, 255).astype(np.uint8)
        # 调用展示函数展示该图片的原始样本、攻击样本和噪声
        show_images_diff(img2, 0, adv, 0)

        # 每轮二分查找结束后，更新c值以及对应的边界
        confidence_old = -1
        if (o_bestscore == target_label) and o_bestscore != -1:
            # 攻击成功，减小c的值
            upper_bound = min(upper_bound, confidence)
            if upper_bound < 1e9:
                print()
                confidence_old = confidence
                confidence = (lower_bound + upper_bound) / 2
        else:
            lower_bound = max(lower_bound, confidence)
            confidence_old = confidence
            if upper_bound < 1e9:
                confidence = (lower_bound + upper_bound) / 2
            else:
                confidence *= 10

        print("outer_step={} confidence {}->{}".format(outer_step,
                                                       confidence_old, confidence))