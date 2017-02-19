#try to make the dataset in normal distrubution

import matplotlib.pyplot as plt
import csv
import numpy as np

#read data from csv
class AngleNum(dict):
	def __missing__(self, key):
		return 0

angle_num = AngleNum()

#read data from csv,and count the angle num
with open('data/driving_log.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
    	angle = round(float(row[3]),2)*100
    	angle_num[angle] = angle_num[angle]+1


labels=[]
label_num=[]

for (label, num) in angle_num.items():
    labels.append(label)
    label_num.append(num)

x_label = np.arange(-150,150,5)
y_label = np.arange(0,1000,100)
plt.xticks(x_label,x_label,ha='right',rotation=45)
plt.yticks(y_label,y_label)
print(labels)
plt.bar(labels,label_num)
plt.show()