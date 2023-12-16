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
num_list = [17.8, 14.4, 16.7, 13.8, 15.5, 17.4,16.8]
num_list1 = [10.5, 8.8, 10.1, 7.6, 9.4, 10.6,10.8]
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
axes.set_ylim([0, 23])
plt.xlabel('loss of different trajectory features')
plt.ylabel("loss")
plt.legend()

plt.show()


