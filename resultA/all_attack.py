import numpy as np
#geolife全部特征
import matplotlib.pyplot as plt
#zoo算法
# drop_percentage = list(range(0, 90,10))
drop_percentage = ['normal','I-FGSM','FGSM','SimBA','NAttack']
# acc_geolife = [0.89339, 0.86911, 0.85389, 0.85213, 0.83947, 0.81456, 0.79699, 0.77400, 0.74048]

# acc_SHL = [0.89236, 0.89014, 0.87500, 0.80669, 0.75464, 0.71564, 0.68802, 0.62951, 0.61765]
acc_geolife1 = [0.85301, 0.55283, 0.61602, 0.72552, 0.60208]

acc_shl1 = [0.77074,0.43294, 0.38272, 0.58605, 0.55605]


axes = plt.gca()
axes.set_ylim([0.2, 1])


plt.scatter(drop_percentage, acc_geolife1)
plt.plot(drop_percentage, acc_geolife1, label='Geolife-attack',marker='o')
plt.scatter(drop_percentage, acc_shl1)
plt.plot(drop_percentage, acc_shl1,'--', label='SHL-attack',marker='o')
plt.xlabel('accuracy of different attack methods')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()
