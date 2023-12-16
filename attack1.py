import numpy as np

# 读取.npy文件

'''

train1 = np.load("./data/geolife_features/multi_feature_segs_train.npy")
# train2 = np.load("./data/geolife_features/multi_feature_segs_train_all_features_normalized.npy")
train3 = np.load("./data/geolife_features/multi_feature_segs_train_normalized.npy")
test1 = np.load("./data/geolife_features/multi_feature_segs_test.npy")
# test2 = np.load("./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy")
test3 = np.load("./data/geolife_features/multi_feature_segs_test_normalized.npy")
p = np.load("./data/geolife_features/pedcc.npy")
s = np.load("./data/geolife_features/std.npy")
'''
test1 = np.load("./data/geolife_features/multi_feature_segs_test.npy")
test3 = np.load("./data/geolife_features/multi_feature_segs_test_normalized.npy")
test4 = np.load("./data/geolife_features/multi_feature_seg_labels_test.npy")
test5 = np.load("./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy")
p = np.load("./data/geolife_features/pedcc.npy")
s = np.load("./data/geolife_features/std.npy")


'''
z = 0.1
np.save('./data/geolife_features/multi_feature_segs_test.npy', test1+z)
np.save('./data/geolife_features/multi_feature_segs_test_normalized.npy', test3+z)
np.save('./data/geolife_features/multi_feature_seg_labels_test.npy', test4+z)
np.save('./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy', test5+z)
np.save('./data/geolife_features/pedcc.npy', p+z)
np.save('./data/geolife_features/std.npy', s+z)
'''



'''
print(train1)

# print(train2.shape)
print(train3.shape)
print(test1.shape)
# print(test2.shape)
print(test3.shape)

print(p.shape)
print(s.shape)

'''


p = np.array(p)
test4 = np.array(test4)
t = 2
w = 0.5
test1[:, :, t] = test1[:, :, t] + w
test3[:, :, t] = test3[:, :, t] + w
test5[:, :, t] = test5[:, :, t] + w
s[t] = s[t] + w

test4[:, t] = test4[:, t] + w
p[t, :] = p[t, :] + w

np.save('./data/geolife_features/multi_feature_segs_test.npy', test1)
np.save('./data/geolife_features/multi_feature_segs_test_normalized.npy', test3)
np.save('./data/geolife_features/multi_feature_seg_labels_test.npy', test4)
np.save('./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy', test5)
np.save('./data/geolife_features/pedcc.npy', p)
np.save('./data/geolife_features/std.npy', s)


'''
np.save('./data/geolife_features/multi_feature_segs_train.npy', train1-0.1)
# np.save('./data/geolife_features/multi_feature_segs_train_all_features_normalized.npy', train2 )
np.save('./data/geolife_features/multi_feature_segs_train_normalized.npy', train3-0.1)
np.save('./data/geolife_features/multi_feature_segs_test.npy', test1-0.1)
# np.save('./data/geolife_features/multi_feature_segs_test_all_features_normalized.npy', test2 )
np.save('./data/geolife_features/multi_feature_segs_test_normalized.npy', test3-0.1)
np.save('./data/geolife_features/pedcc.npy', p-0.1)
np.save('./data/geolife_features/std.npy', s-0.1)
'''




