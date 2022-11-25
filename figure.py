# import random
# import matplotlib.pyplot as plt
# import numpy as np

# # num = 10
# # p = []
# # for _ in range(num):
# # 	p.append((random.uniform(0,1),random.uniform(0,1)))
# # 	x = [k[0] for k in p]
# # 	y = [k[1] for k in p]

# # def dominate(a,b):
# # 	for i,j in zip(a,b):
# # 		if i >= j:
# # 			return False
# # 	return True

# # s = []

# # for m in range(num):
# # 	non_d = True
# # 	for n in range(num):
# # 		if m == n:
# # 			continue
# # 		if dominate(p[n],p[m]):
# # 			non_d = False
# # 			break
# # 	if non_d:
# # 		s.append(p[m])
# # # print(p)
# # # print(s)

# # o = [k[0] for k in s]
# # b = [k[1] for k in s]

# g5x = []  #error rate
# g5y = [] # flops

# with open('g5.txt','r') as f:
# 	for line in f.readlines():
# 		if 'Rate' in line.strip():
# 			g5x.append(round(100-float(line.strip().split(' ')[-1][:-1]),2))
# 		if 'GFlOPs' in line.strip(): 
# 			g5y.append(float(line.strip().split(' ')[-2]))
# gg5x = []
# gg5y = []
# for i in range(len(g5x)):
# 	if g5y[i] < 0.3 and g5x[i] < 50:
# 		gg5x.append(g5x[i])
# 		gg5y.append(g5y[i])

# g10x = []
# g10y = []
# with open('g10.txt','r') as f:
# 	for line in f.readlines():
# 		if 'Rate' in line.strip():
# 			g10x.append(round(100-float(line.strip().split(' ')[-1][:-1]),2))
# 		if 'GFlOPs' in line.strip(): 
# 			g10y.append(float(line.strip().split(' ')[-2]))

# gg10x = []
# gg10y = []
# for i in range(len(g10x)):
# 	if g10y[i] < 0.3 and g10x[i] < 50:
# 		gg10x.append(g10x[i])
# 		gg10y.append(g10y[i])

# fig = plt.figure('进化趋势')
# plt.plot()
# plt.scatter(gg5x,gg5y,color='',marker='o',s=15,edgecolors='black',label='Generation 5')
# plt.scatter(gg10x,gg10y,c='black',marker='x',s=15,label='Gernation 10')
# plt.legend(loc='best')
# plt.xlabel('Error rate')
# plt.ylabel('FlOPs')

# plt.title('进化趋势')
# my_x_ticks = np.arange(min(gg5x+gg10x),max(gg5x+gg10x), 10)
# my_y_ticks = np.arange(min(gg5y+gg10y),max(gg5y+gg10y), 0.02)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)

# plt.show()
import time
time.sleep(3)
print('hello world')