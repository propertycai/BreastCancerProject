import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.family"] = "Airal"  
plt.rcParams["font.size"] = 8  # 设置全局字体大小为8pt
# 设置背景为灰色
sns.set_style("darkgrid")


plt.figure(figsize=(4, 3))
plt.bar(0,8453/118975,color='#F57E21')
plt.bar(1,5574/118975,color='#ffbb00')
plt.bar(2,104948/118975,color='#69a040')
# 设置 x 轴标签为 '低等级', '中等级', '高等级'
plt.xticks([0, 1, 2], ['', '', ''])

plt.savefig('生存分布1.svg',dpi=450)
plt.show()

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.family"] = "Airal"  
plt.rcParams["font.size"] = 8  # 设置全局字体大小为8pt
# 设置背景为灰色
sns.set_style("darkgrid")


plt.figure(figsize=(3, 3)) 
plt.bar(0,6762/95180,color='#F57E21')  
plt.bar(1,4459/95180,color='#ffbb00')
plt.bar(2,83959/95180,color='#69a040')
# 设置 x 轴标签为 '低等级', '中等级', '高等级'
plt.xticks([0, 1, 2], ['', '', ''])

plt.savefig('生存分布2.svg',dpi=450)
plt.show()

plt.figure(figsize=(3, 3)) 
plt.bar(0,83959/251877,color='#F57E21')
plt.bar(1,83959/251877,color='#ffbb00')
plt.bar(2,83959/251877,color='#69a040')
# 设置 x 轴标签为 '低等级', '中等级', '高等级'
plt.xticks([0, 1, 2], ['', '', ''])

plt.savefig('生存分布3.svg',dpi=450)
plt.show()