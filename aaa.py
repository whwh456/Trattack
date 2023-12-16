from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
# 定义扰动值列表
epsilons = [0, .05, .1, .15, .2, .25, .3]
# 预训练模型路径(训练好的模型文件的存储路径)
pretrained_model = "checkpoint/lenet_mnist_model.pth"
device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")



def fgsm_attack(image, epsilon, data_grad):
    """
    :param image: 需要攻击的图像
    :param epsilon: 扰动值的范围
    :param data_grad: 图像的梯度
    :return: 扰动后的图像
    """
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad
    # 添加剪切以维持[0,1]范围

    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image
