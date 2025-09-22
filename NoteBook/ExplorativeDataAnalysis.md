---
tags:
  - Statistics
relation:
  - "[[Probability&Statistic]]"
  - "[[Linear Algebra]]"
  - "[[Python学习]]"
teacher: 杨蕾
---

### Cluster Analysis
将样本点写作矩阵, 以行分类为Q聚类分析(样本聚类), 按列则是R聚类分析(指标聚类)
#### Q型聚类
若是我们将每个样本点看作一个p维的向量, 则显然这些样本分布在$\mathbb R^p$空间中, 很容易想到用距离来进行分类
##### Minkowski距离
$$d_q(x,y) = [\sum_{k = 1}^p|x_k - y_k|^q]^{\frac{1}{q}}, \,\, q> 0$$
其中又有常用的绝对值距离, Euclid距离, Chebyshev距离. Euclid距离最常用, 因为他在坐标轴进行正交旋转的时候不变.

值得注意的是在采用 **Minkowski 距离** 时，一定要采用相同量纲的变量。如果变量的量纲不同，测量值变异范围相差悬殊时，建议首先进行数据的标准化处理，然后再计算距离。在采用 **Minkowski 距离** 时，还应尽可能地避免变量的多重相关性（multicollinearity）。多重相关性所造成的信息重叠，会片面强调某些变量的重要性。由于 **Minkowski 距离** 的这些缺点，一种改进的距离就是马氏距离，定义如下
##### 马氏 (Mahalanobis) 距离
$$d(x, y) = \sqrt{(x - y)^T \Sigma^{-1} (x - y)}$$
$\Sigma$为总体样本$Z$的协方差矩阵，实际中$\Sigma$ 往往是不知道的，常常需要用样本协方差来估计。马氏距离对一切线性变换是不变的，故不受量纲的影响.

#### R型聚类
- 使用相关系数矩阵
- 利用夹角余弦
