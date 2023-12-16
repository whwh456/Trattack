import numpy as np
from matplotlib import pyplot as plt
import pylab as pl

'''
name_list = [0, 1, 2, 3, 4]
num_list = [0.77778, 0.61728, 0.76543, 0.80754, 0.72048]
num_list1 = [0.77778, 0.6111, 0.74691, 0.82605, 0.81566]
'''

x1 = [0, 'fgsm', 'Uniform0.1', 'Uniform0.3', 'Uniform0.5']  # Make x, y arrays for each graph
y1 = [0.77778, 0.61728, 0.76543, 0.80754, 0.72048]
x2 = [0, 'fgsm', 'Uniform0.1', 'Uniform0.3', 'Uniform0.5']
y2 = [0.77778, 0.6111, 0.74691, 0.82605, 0.81566]
pl.plot(x1, y1, 'r', label='attack',marker='o')  # use pylab to plot x and y : Give your plots names
pl.plot(x2, y2, 'g', label='defense',linestyle='--',marker='o')


# pl.title('P-T')  # give plot a title
pl.xlabel('Approach')  # make axis labels
pl.ylabel('accuracy')

#pl.xlim(0, 5)  # set axis limits
pl.ylim(0, 1)
pl.legend()
#pl.show()
a=[[1,2]]
np.array(a)

print(a*10)
