import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

dataset = 'gtsrb'
logdir = 'pics'
# colors = ['#DB5A9A', '#E5A46E', '#0A5B54']
colors = ['#253F58', '#AEC33A', '#D92139']

name_list = {'mnist':'MNIST', 'cifar':'CIFAR-10', 'gtsrb':'GTSRB'}
dataset_name = name_list[dataset]

df1 = pd.read_csv(os.path.join(logdir, f'{dataset}/run-td_{dataset}_t_drop_0.5_t_l2_0.001_cm_cnn_3_2_nd_imagenet_api_onehot_sampling_random_lr_0.0_iter_20_k_1000_m_drop_0.5_m_l2_0.001_pat_20-tag-true_data_test_aggrement.csv'))
df1 = df1[df1['Step'] % 2000 == 0]
x1 = np.array(df1['Step'])
y1 = np.array(df1['Value'])
df2 = pd.read_csv(os.path.join(logdir, f'{dataset}/run-td_{dataset}_t_drop_0.5_t_l2_0.001_cm_cnn_3_2_nd_imagenet_api_onehot_sampling_kcenter_lr_0.0_iter_20_k_1000_m_drop_0.5_m_l2_0.001_pat_20-tag-true_data_test_aggrement.csv'))
df2 = df2[df2['Step'] % 2000 == 0]
x2 = np.array(df2['Step'])
y2 = np.array(df2['Value'])
df3 = pd.read_csv(os.path.join(logdir, f'{dataset}/run-td_{dataset}_t_drop_0.5_t_l2_0.001_cm_cnn_3_2_pretrain_nd_imagenet_api_onehot_sampling_random_lr_0.0_iter_20_k_1000_m_drop_0.5_m_l2_0.001_pat_20-tag-true_data_test_aggrement.csv'))
df3 = df3[df3['Step'] % 2000 == 0]
x3 = np.array(df3['Step'])
y3 = np.array(df3['Value'])
print(y3)

# 绘制分组柱状图
mode = ''
if mode == 'bar':
    width = 560
    plt.bar(x1-width,y1,width=width,label='Normal',color=colors[0],edgecolor='k',zorder=1)
    plt.bar(x2,y2,width=width,label='ActiveThief',color=colors[1],edgecolor='k',zorder=2)
    plt.bar(x3+width,y3,width=width,label='TransferThief',color=colors[2],edgecolor='k',zorder=3)
else:
    plt.plot(x1,y1,label='Normal',color=colors[0],zorder=1)
    plt.plot(x2,y2,label='ActiveThief',color=colors[1],zorder=2)
    plt.plot(x3,y3,label='TransferThief',color=colors[2],zorder=3)

# 添加x,y轴名称、图例和网格线
plt.xlabel('Query budget',fontsize=11)
plt.ylabel('Agr',fontsize=11)
plt.legend(frameon=False)
plt.grid(ls='--',alpha=0.8)
plt.title(f'Agr on {dataset_name}')
plt.ylim(0.25,1)

plt.xticks(np.arange(0, 20001, 2000))
# plt.tick_params(axis='x',length=0)

# plt.tight_layout()
# plt.savefig(f'pics/{dataset}.svg',format="svg")
# plt.show()


with PdfPages(f'pics/{dataset}.pdf') as pdf:
    # 将图表添加到PdfPages对象
    pdf.savefig()
    # 关闭图表
    plt.close()