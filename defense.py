import torch
from torch.autograd.grad_mode import F
import numpy as np
import matplotlib.pyplot as plt

def fgsm_attack(data, eps, data_grad):
    sign_data_grad = np.sign(data_grad)
    perturbed_data = torch.tensor(data + eps*sign_data_grad)
    #perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# 拉普拉斯噪声机制
def laplace_function(x, beta):
    result = (1 / (2 * beta)) * np.e ** (-1 * (np.abs(x) / beta))
    return result


def noisyCount(x, sensitivety, epsilon):
    beta = sensitivety / epsilon
    n_value = (1 / (2 * beta)) * np.e ** (-1 * (np.abs(x) / beta))
    #print(n_value)
    return x

if __name__ == "__main__":
    t = np.load("./data/geolife_features/multi_feature_segs_test.npy")
    t1 = np.load("./data/geolife_features/multi_feature_segs_test_normalized.npy")
    t4 = np.load("./data/geolife_features/pedcc.npy")
    t6 = np.load("./data/geolife_features/multi_feature_seg_labels_test.npy")
    t7 = np.load("./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy")



    t4 = np.array(t4)
    t6 = np.array(t6)
    k = 4
    w = 0.3
    t[:, :, k] = t[:, :, k] + w
    t1[:, :, k] = t1[:, :, k] + w
    t7[:, :, k] = t7[:, :, k] + w

    t6[:, k] = t6[:, k] + w
    t4[k, :] = t4[k, :] + w

    np.save('./data/geolife_features/multi_feature_segs_test.npy', t)
    np.save('./data/geolife_features/multi_feature_segs_test_normalized.npy', t1)
    np.save('./data/geolife_features/multi_feature_seg_labels_test.npy', t6)
    np.save('./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy', t7)
    np.save('./data/geolife_features/pedcc.npy', t4)

    '''
        s1 = 15
        s2 = 5
        t = noisyCount(test1, s1, s2)
        t1 = noisyCount(test3, s1, s2)
        t4 = noisyCount(p, s1, s2)
        # t5 = noisyCount(s, s1, s2)
        t6 = noisyCount(test4, s1, s2)
        t7 = noisyCount(test5, s1, s2)



        z = 0.1
        np.save('./data/SHL_features/multi_feature_segs_test.npy', t+z)
        np.save('./data/SHL_features/multi_feature_segs_test_normalized.npy', t1+z)
        np.save('./data/SHL_features/pedcc.npy', t4+z)
        # np.save('./data/SHL_features/std.npy', t5)
        np.save('./data/SHL_features/multi_feature_seg_labels_test.npy', t6+z)
        np.save('./data/SHL_features/multi_feature_segs_test_all_features_normalized.npy', t7+z)
        #s = np.load("./data/geolife_features/std.npy")
    '''







# 拉普拉斯噪声


# 对抗绕动

'''
    s1 = 15
    s2 = 5
    t = noisyCount(test1, s1, s2)
    t1 = noisyCount(test3, s1, s2)
    t4 = noisyCount(p, s1, s2)
    # t5 = noisyCount(s, s1, s2)
    t6 = noisyCount(test4, s1, s2)
    t7 = noisyCount(test5, s1, s2)
'''





'''
ss = 5
t = fgsm_attack(test1, ss, test1)
t1 = fgsm_attack(test3, ss, test3)
t4 = fgsm_attack(p, ss, p)
# t5 = fgsm_attack(s, ss, s)
t6 = fgsm_attack(test4, ss, test4)
t7 = fgsm_attack(test5, ss, test5)
'''