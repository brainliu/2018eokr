#-*-coding:utf8-*-
#user:brian
#created_at:2018/9/15 10:58
# file: pca_code.py
#location: china chengdu 610000
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
###定义一个函数，读取数据并分别加载三个数据

#修改显示图片的颜色
color_dlist=["red","blue","green","black","yellow","pink","purple"]

def load_data(file_place):
    data_pca=open(file_place)
    result=[]
    #读取数据并存储在array中
    for data in data_pca:
        temp=data.split("\t")
        # print(temp)
        temp2=[]
        #转换为float格式并存储到dataframe中
        for j in range(len(temp)-1):
            temp2.append(float(temp[j]))
        temp2.append(temp[-1])
        result.append(temp2)
        # print(temp2)
    data=np.array(result)[:,0:-1].astype(float)
    labels= np.array(result)[:,-1]
    color_name_number = {} #颜色和名字的对应表
    for index, dict_name in enumerate(list(set(labels))):
        # print index, dict_name
        color_name_number[dict_name] = color_dlist[index]


    return data,labels,color_name_number

#定义一个画图的函数，画散点图并标记
def plt_picture(pca_a_xy,label_a,color_dict_a,savename):
    fig,ax=plt.subplots()
    for i in range(len(pca_a_xy)):
        x,y=np.array(pca_a_xy)[i,:]
        label=label_a[i]
        color=color_dict_a[label]
        ax.scatter(x,y,c=color)
    for j in (color_dict_a.keys()):
        ax.scatter(0,0,c=color_dict_a[j],label=j)
    ax.legend(loc= "upper left")
    ax.grid(True)
    plt.savefig(savename)
    plt.show()


##第一种算法--->自己写的pca算法
# PCA计算过程程序代码的逻辑原理:
# 第一步：求均值。求平均值，然后对于所有的样例，都减去对应的均值
# 第二步：求特征协方差矩阵
# 第三步：求协方差的特征值和特征向量
# 第四步：将特征值按照从大到小的顺序排序，选择其中最大的k个，然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵.
# 第五步：将样本点投影到选取的特征向量上。 假设样例数为m，特征数为n，减去均值后的样本矩阵为DataAdjust(m*n)，
# 协方差矩阵是n*n，选取的k个特征向量组成的矩阵为EigenVectors(n*k).那么投影后的数据FinalData为：
#  FinalData(m*k) = DataAdjust(m*n) * EigenVectors(n*k)
def pca_by_me(dataMat, K=65535):  # dataMat是原始数据，一个矩阵，K是要降到的维数
    meanVals = np.mean(dataMat, axis=0)  # 第一步:求均值
    meanRemoved = dataMat - meanVals  # 减去对应的均值
    covMat = np.cov(meanRemoved, rowvar=0)  # 第二步,求特征协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 第三步,求特征值和特征向量
    eigValInd = np.argsort(eigVals)  # 第四步,将特征值按照从小到大的顺序排序
    eigValInd = eigValInd[: -(K + 1): -1]  # 选择其中最大的K个
    redEigVects = eigVects[:, eigValInd]  # 然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵.
    lowDDataMat = meanRemoved * redEigVects  # 第五步,将样本点投影到选取的特征向量上,得到降维后的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 还原数据
    return lowDDataMat, reconMat  #降维后的数据和还原的数据


#分别读取三个数据
data_pca_a,label_a,color_dict_a=load_data("pca_a.txt")
data_pca_b,label_b,color_dict_b=load_data("pca_b.txt")
data_pca_c,label_c,color_dict_c=load_data("pca_c.txt")


#用第一个算法分别对三个数据分别进行降维处理
pca_a_xy,_=pca_by_me(data_pca_a,2)
pca_b_xy,_=pca_by_me(data_pca_b,2)
pca_c_xy,_=pca_by_me(data_pca_c,2)
#画出1-3个图 第一种算法的
plt_picture(pca_a_xy,label_a,color_dict_a,"figure1.png")
plt_picture(pca_b_xy,label_b,color_dict_b,"figure2.png")
plt_picture(pca_c_xy,label_c,color_dict_c,"figure3.png")

# # ======================PCA主成分分析+调用写好了的SVD包来做=================
# 用第二种算法对数据进行降维处理 用的svd算法
from sklearn.decomposition import PCA ,IncrementalPCA  # 主成分分析（PCA）
pca = PCA(n_components=2, whiten=True)  # PCA 使用随机SVD
SVD_pca_a = pca.fit_transform(data_pca_a)
SVD_pca_b = pca.fit_transform(data_pca_b)
SVD_pca_c = pca.fit_transform(data_pca_c)
#画出4-6个图
plt_picture(SVD_pca_a,label_a,color_dict_a,"figure4.png")
plt_picture(SVD_pca_b,label_b,color_dict_b,"figure5.png")
plt_picture(SVD_pca_c,label_c,color_dict_c,"figure6.png")


# # ======================PCA主成分分析+调用TSNE来做=================
#第三种算法TSNE
X_tsne_pca_a = TSNE(n_components=2,learning_rate=100).fit_transform(data_pca_a)
X_tsne_pca_b = TSNE(n_components=2,learning_rate=100).fit_transform(data_pca_b)
X_tsne_pca_c = TSNE(n_components=2,learning_rate=100).fit_transform(data_pca_c)

#画出7-9个图
plt_picture(X_tsne_pca_a,label_a,color_dict_a,"figure7.png")
plt_picture(X_tsne_pca_b,label_b,color_dict_b,"figure8.png")
plt_picture(X_tsne_pca_c,label_c,color_dict_c,"figure9.png")