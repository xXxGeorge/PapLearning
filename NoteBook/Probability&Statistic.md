---
tags:
  - BasicMath
relation:
  - "[[StochasticProcess]]"
---
# Part 1 Probability
---

### 基础知识
##### 随机变量及其分布
随机变量就是一个从样本点指向实数的映射

#### t分布(用于假设检验时)
对于$X \sim N(\mu, \sigma^2), \mu, \sigma^2$未知, 当我们计算得到一个$\mu_0$想知道对不对的时候, 就可以计算检验统计量$$T = \frac{\overline X - \mu_0}{\sqrt{S^2/n}} \sim t(n-1)$$
对于我们抽出的样本, 大致符合Normal Distribution是可能性最大的情况. 在这种情况下, 构造出来的T最有可能落在中心的区域(t分布的图像), 如何定义这个“中间”则由给定的检验水平$\alpha$来确定. 此时如果T不在中间, 则很有可能是$\mu_0$取错了.当然也有可能是样本的取值没有反应原本的趋势(太集中了, 太偏左了之类的), 这就犯了第一类错误, 将正确的$\mu_0$当作错的.
当然也有可能是$\mu_0$取错了(比如使统计量向右偏移), 但是样本取得也有问题(整体偏右), 导致最后的T统计量还是在“中间”的范围内, 这就是第二类错误, 将错误的$\mu_0$当错正确的.
可视化: [[第一类和第二类错误(统计检验)]]

#### $\chi^2$分布
- 如果 $Z_1, Z_2, \dots, Z_k \sim N(0, 1)$，那么：$$\chi^2_k = \sum_{i=1}^k Z_i^2$$
⇒ 称为自由度为k的$\chi^2$分布

$\chi^2$分布与Normal Distribution关联较大, 我们应该注意到几点, 一是组合而成的随机变量应该是符合标准正态分布的, 另一点是符合正态分布的随机变量的线性组合依然是符合正态分布的.
##### 数字特征
| 特征  | 表达式  | 条件或说明     |
| --- | ---- | --------- |
| 均值  | $k$  | 正值        |
| 方差  | $2k$ | 随自由度增大而增大 |

#### t分布
若有$Z \sim N(0, 1), U \sim \chi^2(n)$, 且$U$与$Z$独立；
则我们定义：$$T = \frac{Z}{\sqrt{U/n}}\sim t(n)$$
##### 数字特征
| 特征  | 表达式                   | 条件或说明          |
| --- | --------------------- | -------------- |
| 均值  | $0$                   | 当$\nu > 1$时存在  |
| 方差  | $\frac{\nu}{\nu - 2}$ | 当$\nu > 2$ 时存在 |
#### F分布
而 F 分布来源于两个独立卡方变量的比值. 具体的统计量构建为:
如果：$U \sim \chi^2_m、V \sim \chi^2_n，且 U \perp V$, 那么定义：
$$F = \frac{U/m}{V/n} \sim F(m, n)$$
##### 数字特征
| 特征  | 表达式                                                     | 条件            |
| --- | ------------------------------------------------------- | ------------- |
| 均值  | $\frac{d_2}{d_2 - 2}$                                   | $d_2 > 2$时存在  |
| 方差  | $\frac{2d_2^2(d_1 + d_2 - 2)}{d_1(d_2 - 2)^2(d_2 - 4)}$ | $d_2 > 4$ 时存在 |
#### 假设检验
假设检验分为单尾检验和双尾检验, 不同的情况下选取的分位数不同. 我们通过观察原假设与备择假设来判断. 注意点在于我们一般令原假设取等号.
##### 单尾检验时:
假设 $H_0: \mu \leq \mu_0$，你做的是一个右尾检验：
$H_1: \mu > \mu_0$
你设显著性水平$\alpha = 0.05$，则：
- 你只关心右边 5% 的区域
- 所以临界值是：
$u_{1 - \alpha} = u_{0.95} \approx \boxed{1.645}$
类似地，如果是左尾检验（$H_1: \mu < \mu_0$），用：
$u_\alpha = u_{0.05} \approx \boxed{-1.645}$
##### 双尾检验时：
假设 $H_0: \mu = \mu_0$，你做的是：
 $H_1: \mu \ne \mu_0$
设显著性水平 $\alpha = 0.05$，那么：
- 左右各截去 $\alpha/2 = 0.025$
- 所以你要找的是：
$\text{左右临界值} = \pm u_{1 - \alpha/2} = \pm u_{0.975} \approx \boxed{\pm 1.96}$
而：$u_{0.025} = -1.96$

上面的检验方法在实际应用中常用于样本成正态分布等情况时使用, 可以方便的构造出统计量. 除此以外需要特别注意的一类是对[[回归系数的假设检验]], 构造的方式有很大不同

##### 对回归系数的假设检验


#### 区间估计
[b站讲解](https://www.bilibili.com/video/BV1p1y6Y6E9A/?spm_id_from=333.337.search-card.all.click&vd_source=320e99e9d1061fcda4da2e64ad20113b)
##### Definition
区间估计的两个方向: 可靠性, 精度, 二者在一定程度上成负相关. 我们就是要在保证可靠性的基础上估计未知参数.
此时若
$$P\{\hat \theta_1 \leq \theta \leq \hat\theta_2\} = 1-\alpha$$
则称$[\hat\theta_1, \hat\theta_2]$ 为置信度$1-\alpha$的置信区间.$\alpha$为显著性水平.
对于要求求$\mu$ 的置信区间的题目, 也就是需要求出一个区间, 使得$P\{\mu_L \leq \mu \leq \mu_R \} = 1-\alpha$ .此时问题转化为如何找到这样的关于$\mu$ 的关系, 这时候就可以利用含有$\mu$ 的变量, 结合不同的分布来实现.
##### Example
这时候就出现了几种常见情况(样本来自正态总体):
1. $\mu$ 的区间估计($\sigma^2$ 已知): 此时我们利用标准化正态分布  $\frac{\bar x - \mu}{\frac{\sigma}{\sqrt{n}}} \to N(0,1)$
2. $\mu$ 的区间估计($\sigma^2$ 未知): 此时利用t分布  $\frac{\bar x - \mu}{\frac{S}{\sqrt{n}} }\to t(n-1)$
3.  $\sigma^2$ 的区间估计($\mu$ 已知):  此时利用$\chi^2$分布, $\frac{1}{\sigma^2}\sum_{i= 1}^{n}(x_i - \mu)^2 \to \chi^2(n)$  
4.  $\sigma^2$ 的区间估计($\mu$ 未知):  此时利用$\chi^2$分布, $\frac{(n-1)S^2}{\sigma^2} \to \chi^2(n-1)$ 

结果化简:
![[区间估计总结.png]]
### 参数估计
#### 点估计(moment estimate)
#### 矩估计法(moments estimate)
本质上就是用样本的数据去近似替代某些参数来进行计算, 如计算期望, 方差等等. 计算期望可以直接算出, 但是像方差这样的数据就需要先算出期望的近似再进行计算, 将步骤规整化之后就得到了如下例的步骤:

>写出由样本得到的k阶原点矩$\mu_i$, 与分布所求出的理论原点矩, 如$\mu, \sigma^2$构成等式(此时理论原点矩包含了未知数), 再用等式解出未知数, 再将理论原点矩替换为样本得到的k阶原点矩.

若计算本身简单也可以直接得到:

>如计算协方差或相关矩阵

这里可能会涉及到无偏估计;
##### Definition(无偏估计)
$$\displaylines{若有估计量\mu_1, 若同时有E(u_1) = \mu, 则称其为一个无偏估计量}$$
##### Definition(相合估计)


#### 极大似然估计(maximum likelihood estimate)

>设总体X是离散型随机变量，事件:"$X=x$"的概率为 $p(x,\theta)$，其中$\theta$为待估计的未知参数。假定是样本$x_1, x_2 \dots x_n$的一组观察值$X-1, X_2 \dots X_n$, 即相当于事件
>$X_i = x_i$同时发生了, 由于$X_i$相互独立，且与总体同分布，所以这n个事件同时发生的概率为$$P(X_i = x_i) = {\Pi _{i = 1} }^n P(X_i = x_i)$$
> 此处应该注意, 离散随机变量的时候我们直接写作概率的乘积, 但是面对连续型随机变量的时候我们应该写概率分布函数的乘积而不是概率的乘积
  显然它是$\theta$的函数，称这个函数为似然函数，记作$L(\theta)$，即
$$L(\theta) = {\Pi_{i = 1}}^n p(x_i, \theta)$$

MLE的优良性质: $$if \,\, \theta \to \hat \theta, \Rightarrow g(\theta) \to \hat g(\theta)$$


> [!PDF|note] [[应用统计随堂测试一.pdf#page=1&selection=397,0,416,1&color=note|应用统计随堂测试一, p.1]]
> > 设总体ξ服从均匀分布 (0, θ] ，0.2, 0.4, 0.5, 0.8, 0.6 是一组样本观察值，则θ 的最大似然估计值为 .
> 

## Part 2 Statistic
### 回归分析概述
回归模型一般形式:$$y = g(x_1,x_2,\dots,x_p) + \epsilon \Leftrightarrow E(y|x_1,x_2,\dots,x_p) = g(x_1,x_2,\dots,x_p)$$
这里的 $y = g(x_1,x_2,\dots,x_p) + \epsilon$ 是回归方程,  $E(y|x_1,x_2,\dots,x_p) = g(x_1,x_2,\dots,x_p)$是回归函数.回归函数不含$\epsilon$. 
对于$\epsilon$ 有Gauss-Markov 假设
- $E(\epsilon|x_1,x_2,\dots,x_p) = 0$
- $Var(\epsilon|x_1,x_2,\dots,x_p) = \sigma^2$
- $Cov(\epsilon_i, \epsilon_j) = 0, i \not= j$ 
内生变量: 与$\epsilon$相关
外生变量: 与$\epsilon$无关
存在内生变量使最小二乘估计不成立
若$X_1 \dots X_p$ 之间相关, 则称其存在共线性.此时最小二乘估计无解或有多解.
#### 回归模型梳理
##### 参数回归模型
###### 线性回归模型
$$y = \beta_0 + \sum_{i = 1}^{p}{\beta_ix_i} + \epsilon$$
解释性强, $x_i$变换一个单位会引起$y$平均变化$\beta_i$个单位. 其中的$\beta$是我们可以确定的变量的参数, $\epsilon$则是不可控的部分.
###### 非线性回归模型
$$y= f(x_1,x_2,\dots,x_p;\beta_1,\beta_2,\dots,\beta_p)+ \epsilon$$其中函数形式已知且非线性.
没有显式解, 依赖于迭代估计参数
##### 非参数回归模型
$$y= f(x_1,x_2,\dots,x_p)+ \epsilon$$其中函数形式未知
核心工作即是估计未知函数.取一点$x_0$以及其领域内的点进行拟合分析, 对于不同的点赋予不同的权重.
##### 半参数回归模型
$$y= f(x_1,x_2,\dots,x_p)+ \sum_{i = 1}^{p}{\beta_ix_i} + \epsilon$$
###### 单指标模型
为避免维数灾难, 我们将p维的变量变为其线性组合, 从而变成一元
$$\displaylines{x_1,x_2,\dots,x_p \to \sum_{i = 1}^{p}{k_ix_i}\\ \Rightarrow y = f(\sum_{i = 1}^{p}{k_ix_i}) + \epsilon}$$
###### 可加模型
$$y = \sum_{i= 1}^{p}{f_i(x_i)} + \epsilon$$
###### 变参数回归模型
$$y = \beta_0(u)x_0 + \beta_1(u)x_1 +\dots + \beta_p(u)x_p + \epsilon$$
上述模型均为均值回归模型, 满足$E(y|x_1,x_2,\dots,x_p) = g(x_1,x_2,\dots,x_p)$, 缺点在于鲁棒性差, 对异常点敏感.
##### 参数中位数回归模型
$$Median(y|x_1,x_2\dots,x_p) = g(x_1,x_2\dots,x_p)$$
###### 线性参数中位数回归模型
$$Median(y|x_1,x_2\dots,x_p) = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots +\beta_px_p$$
###### 非线性中位数回归模型
$$Median(y|x_1,x_2,\dots,x_p)= f(x_1,x_2,\dots,x_p;\beta_1,\beta_2,\dots,\beta_p)$$其中函数形式已知且非线性.
##### 非参数中位数回归模型
$$Median(y|x_1,x_2,\dots,x_p)= f(x_1,x_2,\dots,x_p)$$
##### 半参数中位数回归
$$Median(y|x_1,x_2,\dots,x_p)=  f(x_1,x_2,\dots,x_p)+ \sum_{i = 1}^{p}{\beta_ix_i}$$
##### **参数分位数回归模型**
###### 线性参数分位数回归模型
$$Quantile(y|x_1,x_2,\dots,x_p) = \beta_0 + \beta_1x_1 + \dots + \beta_px_p$$
###### 非线性参数分位数回归模型
$$Quantile(y|x_1,x_2,\dots,x_p) = f(x_1,x_2,\dots,x_p;\beta_1,\beta_2,\dots,\beta_p)$$
##### 非参数分位数回归模型
$$Quantile(y|x_1,x_2,\dots,x_p)= f(x_1,x_2,\dots,x_p)$$
##### 半参数分位数回归
$$Quantile(y|x_1,x_2,\dots,x_p)=  f(x_1,x_2,\dots,x_p)+ \sum_{i = 1}^{p}{\beta_ix_i}$$
##### **参数众数回归模型**
###### 线性参数众数回归模型
$$Mode(y|x_1,x_2,\dots,x_p) = \beta_0 + \beta_1x_1 + \dots + \beta_px_p$$
###### 非线性参数众数回归模型
$$Mode(y|x_1,x_2,\dots,x_p) = f(x_1,x_2,\dots,x_p;\beta_1,\beta_2,\dots,\beta_p)$$
##### 非参数众数回归模型
$$Mode(y|x_1,x_2,\dots,x_p)= f(x_1,x_2,\dots,x_p)$$
##### 半参数众数回归
$$Mode(y|x_1,x_2,\dots,x_p)=  f(x_1,x_2,\dots,x_p)+ \sum_{i = 1}^{p}{\beta_ix_i}$$

### 分类模型

分类模型目的是找出分类器，得出将样本分类的方法。
度量分类模型准确率一般看准确率或是错误率：$$\displaylines{accRate = \frac{1}{n}\sum_{i=1}^{n}I(\hat y_i = y_i)\\I(·)为示性函数}$$
### 模型评价

##### 模型估计
我们想要通过样本得到一个未知的回归函数, 就可以极小化下面的**经验风险函数或目标函数**$R_n(g)$，获得$g(\cdot)$的估计，即

$$\widehat{g} = \arg\min_{g} R_n(g) = \arg\min_{g} \frac{1}{n} \sum_{i = 1}^{n} \ell_n(y_i, g(x_i)).$$

$\ell_n(y_i, g(x_i))$为**损失函数**，用于度量$y_i$与$g(x_i)$之间的偏离程度。
常用的损失函数$\ell_n(y, g(x))$有：
- 平方损失函数：$\ell_n(y, g(x))=(y - g(x))^2$；
- 绝对损失函数：$\ell_n(y, g(x))=\vert y - g(x)\vert$；
- $L_q$损失函数：$\ell_n(y, g(x))=\vert y - g(x)\vert^q$，其中$q > 0$；
- $0 - 1$损失函数，负对数似然损失函数和分位数损失函数等。

##### Definition(MSE)

均方差误差：$$MSE = \frac{1}{n}\sum_{n = 1}^{n}(y - \hat y)^2$$
需要理解[[统计推断中y与mu的关系]]才能更好理解公式的推导。
$bias(\hat g(x)) = E(\hat g(x))- g(x)$:表示估计$\hat g(x)$本身的偏差，为零时即满足无偏性
$Var(\epsilon)$: 表示随机误差$\epsilon$的误差
MSE的计算
我们可以使用变形后的结果去计算:$$MSE = Var(\hat \theta) + (E(\hat \theta - \theta))^2$$

##### Example

$$\displaylines{stimulate: y = \beta_1x_1 + \beta_2x_2 + \epsilon \\ estimate \, MSE(MLE)\\ put\,\beta_1 = 1,\beta_2 = 2\\X_1  = N(0,1), X_2 = N(0,1)\\\epsilon =N(0,1)\\calculate \,\, y_1,\dots,y_n(by\,\,stimulate \,\,given)\\calculate\,estimates \, \hat y_1,\dots,\hat y_n\\calculate\,\,MSE, judge\,\, if \,\,the\,\, estimate\,\,is\,\,proper}$$
这是模拟情况，模拟函数已知，真实情况下需要结合数据猜测$\beta_1,\dots,\beta_n$等参数的值。

##### 自由度(degrees of freedom, df)

let $y_i$'s prediction be $\hat y_i = \hat g(x_i)$, then define $\hat g$'s df as:$$df(\hat g) = \frac{1}{\sigma ^2}\sum_{i = 1}^n Cov(\hat y_i, y_i)$$
令$\hat{\boldsymbol{Y}} = (\hat{y}_1, \cdots, \hat{y}_n)^{\mathrm{T}} \in \mathbb{R}^n$和$\boldsymbol{Y} = (y_1, \cdots, y_n)^{\mathrm{T}} \in \mathbb{R}^n$，则自由度可以写成如下矩阵形式

$$df(\hat{g}) = \frac{1}{\sigma^2} \mathrm{tr}(\mathrm{Cov}(\hat{\boldsymbol{Y}}, \boldsymbol{Y}))$$

记$\boldsymbol{S}$为$n \times n$的**投影矩阵或光滑矩阵**，不管对参数或非参数方法，假设回归函数的估计都具有线性光滑，即$\hat{\boldsymbol{g}} = \boldsymbol{S}\boldsymbol{Y} = (\hat{g}(x_1), \cdots, \hat{g}(x_n))^{\mathrm{T}}$。这时，$\boldsymbol{Y}$的预测可以写为$\hat{\boldsymbol{Y}} = \boldsymbol{S}\boldsymbol{Y}$，且由$\mathrm{Cov}(\boldsymbol{Y}) = \sigma^2 \boldsymbol{I}_n$，则自由度为：$df(\hat{g}) = \mathrm{tr}(\boldsymbol{S})$
直观解释，回归函数中所需要估计的有效参数个数，个数越多模型复杂度越大，拟合越灵活。

随自由度增加，MSE呈现U型曲线，意味着存在着两种相反的影响

##### 偏差-方差权衡

模型灵活度高的时候，拟合结果的方差变大，偏差变小，过拟合
模型灵活度低的时候，拟合结果的方差变小，偏差变大，欠拟合

##### 正则化（regularization)-解决过拟合问题

即极小化以下的惩罚经验风险函数：$$\hat g_\lambda = arg\, min\{R_n(g) + p_\lambda (g)\} = arg\, min\{\frac{1}{n}\sum_{i = 1}^{n}l_n(y_i, g(x)) + p_\lambda(g)\}$$
we need to find a proper $\lambda$ to fit the estimate better

#### 非线性回归模型

非参数估计的劣势：容易遭受维数灾难：
E.g.
对于一个二次可微回归函数，均方值误差函数为$$MSE(\hat{g}(x)) = \frac{c}{n^{p/(4-p)}}$$
If$$MSE(\hat g(x)) = \epsilon, \epsilon>0, \to n = (\frac{c}{\epsilon})^{p/4}$$

### 线性回归模型

对于模型我们先猜测是线性模型，画出散点图观察是否有线性关系（写论文需要用到以展示逻辑的完整性）
$y = g(x_1,x_2,\dots,x_p) + \epsilon$，其中$E(\epsilon|x_1,x_2,\dots,x_p) = 0$
这里的 $y = g(x_1,x_2,\dots,x_p) + \epsilon$ 是回归方程,  $E(y|x_1,x_2,\dots,x_p) = g(x_1,x_2,\dots,x_p)$是回归函数.回归函数不含$\epsilon$. 
对于$\epsilon$ 有Gauss-Markov 假设
- $E(\epsilon|x_1,x_2,\dots,x_p) = 0$
- $Var(\epsilon|x_1,x_2,\dots,x_p) = \sigma^2$
- $Cov(\epsilon_i, \epsilon_j) = 0, i \not= j$ 
内生变量: 与$\epsilon$相关
外生变量: 与$\epsilon$无关
存在内生变量使最小二乘估计不成立
若$X_1 \dots X_p$ 之间相关, 则称其存在共线性.此时最小二乘估计无解或有多解.
[[StochasticProcess]]中的基础知识在这里也有用

#### 多元线性回归
对于多元线性回归我们可以取n各数据点构造出矩阵形式, 此时的$X_{n \times (p+1)}$ 被称为设计矩阵.$$Y = \left(\begin{matrix}Y_1\\Y_2\\\vdots\\Y_n\end{matrix}\right), X = \left(\begin{matrix} 1&X_{11}&X_{12}&\dots &X_{1p}\\1&X_{21}&X_{22} &\dots &X_{2p}\\ \vdots & \vdots &\vdots& \ddots & \vdots\\1 & X_{n1} & X_{n2} &\dots &X_{np}\end{matrix}\right)$$

### 最小二乘估计
$$find\,\, a \,\,proper\,\,\beta, s.t. Min \sum_{i = 1}^{n}{(y-\hat y )^2}$$
通过对$\beta$求偏导并令其等于零得到正规方程, 解正规方程, 可得$\beta$的最小二乘估计为$$\hat \beta = (X^TX)^{-1}X^TY$$
由此可以得到经验回归方程$$\hat Y = X\hat \beta = X(X^TX)^{-1}X^TY$$
此处的$H = X(X^TX)^{-1}X^T$成为帽子矩阵或是投影矩阵(几何意义).
最小二乘估计的有效自由度可定义为:
$$df(ols) = tr(H) = tr(X(X^TX)^{-1}X^T) = tr(X^TX(X^TX)^{-1}) = tr(I_{p+1}) = p+1$$
> 在[[误差符合正态分布的情况下,_beta的最小二乘估计等价于最大似然估计]].即$$if\,\,\epsilon \sim N(0,\sigma^2), \,\, \hat \beta_{LS} = \hat \beta_{MLE}$$
> 

特别的对于一元线性回归, 我们有:
$$\displaylines{y = \beta_0 + \beta_1 X + \epsilon\\ \Rightarrow\hat \beta_0 = y - \hat \beta_1 \overline X,\,\,\hat \beta_1 = \frac{\sum_{i = 1}^{n}(x_i - \overline x)(y_i-\overline y)}{\sum_{i = 1}^{n}(x_i - \overline x)^2}}$$

#### $\sigma$的无偏估计
$$\hat {\sigma}^2 = \frac{RSS}{n-p-1}$$


##### Theorem
对于多元线性回归方程, 则OLS满足:
- $E(\hat \beta) = \beta$ 无偏性
- $Cov(\hat \beta) = {\sigma}^2(X^TX)^{-1}$

#### 假设检验
[[为什么我们要进行假设检验]]
$\beta_{OLS} \to$ 
method 1:

method 2(通过惩罚方法):
$$Min\quad\sum_{i = 1}^{n}(y _i - \beta_0x_{i1} - \dots - \beta_ix_{ip})^2 + \lambda\sum_{j= 1}^n\beta_j$$
#### 回归分析的拟合优度
##### SSReg
$$SSReg = \sum_{i = 1}^{n}(\hat y - y_i)^2$$
##### RSS
$$RSS = \sum_{i = 1}^n(\hat y - \overline y)^2$$
##### SST
化简得到
$$SST = SSReg + RSS$$
SST中SSReg的占比越高则模型的拟合效果越好, 也就是$R^2 = \frac{SSReg}{SST}=1-\frac{RSS}{SST}$ .
不过$R^2$只适合一元线性回归, 多远线性回归时应该用修正过的$R^2$.
#### Y的点分布和预测区间
在相同的置信区间$1-\alpha$ 下, $y_0$ 的预测区间要比回归函数$g(x_0)$ 的置信区间要长, 因为$y_0 = \beta_0 + \beta_1x_{01} + \dots + \beta_p x_{0p} + \epsilon$ 比$g(x_0) = \beta_0 + \beta_1x_{01} + \dots + \beta_p x_{0p}$ 多了一个误差项$\epsilon$ .1

