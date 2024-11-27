import argparse
import scipy.io
import torch
import numpy as np
import os
import shutil
from torchvision import datasets
import matplotlib.pyplot as plt

# 解析命令行参数
parser = argparse.ArgumentParser(description='Demo')
#python demo.py --query_index n
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='./data/test',type=str, help='./test_data')
opts = parser.parse_args()

#无人机到卫星
gallery_name = 'gallery_satellite'
query_name = 'query_drone'

# gallery_name = 'gallery_drone'
# query_name = 'query_satellite'

# 加载数据集
data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in [gallery_name, query_name]}

# 定义一个函数，用于显示图像
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated

# 读取保存的特征和标签信息
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

# 定义函数：对图像进行排序
def sort_img(qf, ql, gf, gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl==-1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

# 获取查询图片的索引
i = opts.query_index
index = sort_img(query_feature[i],query_label[i],gallery_feature,gallery_label)


# 显示查询图像路径
query_path, _ = image_datasets[query_name].imgs[i]
query_label = query_label[i]
print(query_path)
print('Top 5 images are as follow:')

# 创建用于保存图像的文件夹
save_folder = 'image_show/%02d'%opts.query_index
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
shutil.copy(query_path, os.path.join(save_folder, 'query.jpg'))

# 尝试显示排名结果
try: # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(5):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets[gallery_name].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        shutil.copy(img_path, os.path.join(save_folder, f"s{i:02d}.jpg"))
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
    #plt.pause(100)  # pause a bit so that plots are updated
except RuntimeError:
    for i in range(5):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png")
plt.show()
