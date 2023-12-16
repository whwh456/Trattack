import numpy as np
#geolife全部特征
import matplotlib.pyplot as plt

# drop_percentage = list(range(0, 90,10))
drop_percentage = ['normal','I-FGSM','FGSM','aver_per0.5','aver_per0.3','aver_per0.1']
# acc_geolife = [0.89339, 0.86911, 0.85389, 0.85213, 0.83947, 0.81456, 0.79699, 0.77400, 0.74048]

# acc_SHL = [0.89236, 0.89014, 0.87500, 0.80669, 0.75464, 0.71564, 0.68802, 0.62951, 0.61765]
acc_geolife1 = [0.85301, 0.57283, 0.63602, 0.79552, 0.81208, 0.82881]
acc_geolife2 = [0.85301, 0.61245, 0.64342, 0.72048, 0.82605, 0.84197]




axes = plt.gca()
axes.set_ylim([0.4, 1])




plt.scatter(drop_percentage, acc_geolife1)
plt.plot(drop_percentage, acc_geolife1, label='Geolife-attack',marker='o')
plt.scatter(drop_percentage, acc_geolife2)
plt.plot(drop_percentage, acc_geolife2,'--', label='Geolife-defense',marker='o')
plt.xlabel('accuracy of different attack and defense methods')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()
