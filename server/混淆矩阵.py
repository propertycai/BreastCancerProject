import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.family"] = "Airal"  
plt.rcParams["font.size"] = 10  # 设置全局字体大小为8pt
# 设置背景为灰色
sns.set_style("darkgrid")

# 直接定义混淆矩阵的九个数值    1691  1115 20989
cm_values = [
    [1621, 58, 12],
    [55, 1040, 20],
    [31, 431, 20527]
]

# 将数值转化为 NumPy 数组
cm = np.array(cm_values)

# 创建热图
plt.figure(figsize=(4.2, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['', '', ''], yticklabels=['', '', ''])


# 显示图形
plt.savefig('混淆矩阵.svg',dpi=450)
plt.show()
