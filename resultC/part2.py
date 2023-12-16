import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
position = ['Epoch0', 'Epoch1', 'Epoch2', 'Epoch3', 'Epoch4']
data = [0.27162, 0.23373, 0.2322, 0.23192, 0.2314]
plt.bar(x=position, height=data)
plt.xlabel("epoch")
plt.ylabel("avg train_loss")
plt.show()
'''

name_list = ["all", "lng", "lat", "t", "v", "a","deg"]

'''
num_list = [0.60208, 0.6812, 0.7041, 0.6514, 0.6741, 0.6481,0.66235]
num_list1 = [0.55605, 0.6101, 0.6474, 0.6623, 0.6041, 0.6747,0.58415]
'''

num_list = [0.67508, 0.5702, 0.5417, 0.5742, 0.6145, 0.6074,0.5763]
num_list1 = [0.74565, 0.5546, 0.5624, 0.5967, 0.5871, 0.5641,0.6145]




x = list(range(len(num_list)))
width = 0.3;  # 柱子的宽度
index = np.arange(len(name_list));
plt.bar(index, num_list, width, color='steelblue', tick_label=name_list, label='Geolife-defense')
plt.bar(index + width, num_list1, width, color='green',  label='SHL-defense')
# plt.legend(['avg train_loss', 'avg test_loss'], labelspacing=1)

for a, b in zip(index, num_list):  # 柱子上的数字显示
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7);
for a, b in zip(index + width, num_list1):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7);
axes = plt.gca()
axes.set_ylim([0, 0.9])
plt.xlabel('recall of different trajectory features')
plt.ylabel("recall")
plt.legend()

plt.show()


