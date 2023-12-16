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

'''
geolife全部
name_list = [0,"fgsm", "d-fgsm", "0.1", "d-0.1", "0.3","d-0.3","0.5","d-0.5"]
num_list = [0.85301,	0.69141,	0.70343,	0.81842,	0.84197	,0.80754	,0.82605,0.72048,0.81566]
'''


#SHL全部
name_list = [0,1, 2,3,4]
num_list = [0.77778,	0.61728,		0.76543,	0.80754	,0.72048]
num_list1 = [0.77778,0.6111,0.74691, 0.82605,  0.81566]
x = list(range(len(num_list)))
width = 0.3;  # 柱子的宽度
index = np.arange(len(name_list));
plt.bar(index, num_list, width, color='green', tick_label=name_list,label='attack_acc')
plt.bar(index + width, num_list1, width, color='red',  label='defense_acc')
plt.legend(['avg train_loss', 'avg test_loss'], labelspacing=1)

for a, b in zip(index, num_list):  # 柱子上的数字显示
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7);
for a, b in zip(index + width, num_list1):
    plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=7);

plt.xlabel('Approach')
plt.ylabel("acc")
plt.legend()
plt.show()








