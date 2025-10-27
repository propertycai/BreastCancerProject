import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.family"] = "Arial"  # 设置全局字体为Arial
plt.rcParams["font.size"] = 8  # 设置全局字体大小为8pt

# 设置背景为灰色
sns.set_style("darkgrid")

# 设置柱状图的高度
"""heights1 = [0.9868, 0.9757, 0.9875, 0.9813]
heights2 = [0.9897, 0.9800, 0.9917, 0.9855]
heights3 = [0.9883, 0.9798, 0.9875, 0.9834]
heights4 = [0.9956, 0.9920, 0.9958, 0.9938]
heights5 = [0.9927, 0.9880, 0.9917, 0.9896]"""


heights1 = [0.9860, 0.9907, 0.9716, 0.9805]
heights2 = [0.9877, 0.9907, 0.9766, 0.9834]
heights3 = [0.9877, 0.9859, 0.9812, 0.9834]
heights4 = [0.9912, 0.9909, 0.9859, 0.9881]
heights5 = [0.9965, 0.9955, 0.9955, 0.9953]

# 设置柱状图的位置和宽度
x = np.arange(4)
width = 0.15

plt.figure(figsize=(4, 3.1))

# 绘制柱状图
plt.bar(x - 2 * width, heights1, width=width, label='Group 1')
plt.bar(x - width, heights2, width=width, label='Group 2')
plt.bar(x, heights3, width=width, label='Group 3')
plt.bar(x + width, heights4, width=width, label='Group 4')
plt.bar(x + 2 * width, heights5, width=width, label='Group 5')

plt.xticks([])  # 不显示横坐标刻度
plt.ylim(0.90, 1.00)  # 设置纵坐标范围
#plt.title('WBCD',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})
plt.title('WDBC',fontdict={'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'})
# 显示图形
plt.savefig("6-2.svg",dpi=500)
plt.show()
