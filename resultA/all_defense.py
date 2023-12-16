import numpy as np
#geolife全部特征
import matplotlib.pyplot as plt
#zoo算法
# drop_percentage = list(range(0, 90,10))
drop_percentage = ['normal','I-FGSM','FGSM','SimBA','NAttack']
# acc_geolife = [0.89339, 0.86911, 0.85389, 0.85213, 0.83947, 0.81456, 0.79699, 0.77400, 0.74048]

# acc_SHL = [0.89236, 0.89014, 0.87500, 0.80669, 0.75464, 0.71564, 0.68802, 0.62951, 0.61765]
acc_geolife2 = [0.85301, 0.59245, 0.64342, 0.766048, 0.64208]
acc_shl2 = [0.77074, 0.54241, 0.58471, 0.64309, 0.62208]


axes = plt.gca()
axes.set_ylim([0.2, 1])


plt.scatter(drop_percentage, acc_geolife2)

plt.plot(drop_percentage, acc_geolife2, label='Geolife-defense',marker='o')
plt.scatter(drop_percentage, acc_shl2)
plt.plot(drop_percentage, acc_shl2,'--', label='SHL-defense',marker='o')
plt.xlabel('accuracy of different defense methods')
plt.ylabel('accuracy')
plt.legend()

plt.grid()

plt.show()
