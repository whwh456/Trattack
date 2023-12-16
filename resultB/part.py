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
num_list = [22.9, 20.4, 18.7, 18.8, 20.4, 21.7,16.7]
num_list1 = [12.6, 11.7, 10.5, 12.1, 10.1, 9.7,12.4]
x = list(range(len(num_list)))
width = 0.3;  # 柱子的宽度
index = np.arange(len(name_list));
plt.bar(index, num_list, width, color='steelblue', tick_label=name_list, label='Geolife-attack')
plt.bar(index + width, num_list1, width, color='green',  label='SHL-attack')
# plt.legend(['avg train_loss', 'avg test_loss'], labelspacing=1)

for a, b in zip(index, num_list):  # 柱子上的数字显示
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7);
for a, b in zip(index + width, num_list1):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7);
axes = plt.gca()
axes.set_ylim([0, 30])
plt.xlabel('loss of different trajectory features')
plt.ylabel("loss")
plt.legend()

plt.show()


