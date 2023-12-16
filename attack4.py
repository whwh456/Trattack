import torch
from torch.autograd.grad_mode import F
import numpy as np


def fgsm_attack(data, eps, data_grad):
    # data_grad=data_grad.tolist()
    #sign_data_grad = data_grad.sign()
    sign_data_grad = np.sign(data_grad)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    perturbed_data = torch.tensor(data + eps * sign_data_grad)
    #perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

'''
    torch.sign()
    torch.sigmoid()
    torch.tanh_()
'''

if __name__ == "__main__":
    #train1 = np.load("./data/geolife_features/multi_feature_segs_train.npy")
    #train3 = np.load("./data/geolife_features/multi_feature_segs_train_normalized.npy")
    #train4 = np.load("./data/geolife_features/multi_feature_seg_labels_train.npy")
    #train5 = np.load("./data/geolife_features/multi_feature_segs_train_all_features_normalized.npy")
    test1 = np.load("./data/SHL_features/multi_feature_segs_test.npy")
    test3 = np.load("./data/SHL_features/multi_feature_segs_test_normalized.npy")
    test4 = np.load("./data/SHL_features/multi_feature_seg_labels_test.npy")
    test5 = np.load("./data/SHL_features/multi_feature_segs_test_all_features_normalized.npy")
    p = np.load("./data/SHL_features/pedcc.npy")
    s = np.load("./data/SHL_features/std.npy")


    ss = 16
    t = fgsm_attack(test1, ss, test1)
    t1 = fgsm_attack(test3, ss, test3)
    #t2 = fgsm_attack(test1, ss, test1)
    #t3 = fgsm_attack(test3, ss, test3)
    t4 = fgsm_attack(p, ss, p)
    t5 = fgsm_attack(s, ss, s)
    t6 = fgsm_attack(test4, ss, test4)
    t7 = fgsm_attack(test5, ss, test5)


    np.save('./data/SHL_features/multi_feature_segs_test.npy', t )
    np.save('./data/SHL_features/multi_feature_segs_test_normalized.npy', t1)


    np.save('./data/SHL_features/pedcc.npy', t4)
    np.save('./data/SHL_features/std.npy', t5)
    np.save('./data/SHL_features/multi_feature_seg_labels_test.npy', t6)
    np.save('./data/SHL_features/multi_feature_segs_test_all_features_normalized.npy', t7)
  #np.save('./data/geolife_features/multi_feature_segs_train.npy', t2)
    #np.save('./data/geolife_features/multi_feature_segs_train.npy', t3)
'''
def test( model, device, test_loader, eps ):
    for data, true_ in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, eps, data_grad)
        output = model(perturbed_data)

'''
