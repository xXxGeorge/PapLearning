### Trace Regression
#### 基本概念
##### Definition: Trace Regression
trace regression 的特点之一是他的输入是一个矩阵, 而输出则是一个标量. 这与常见的回归模型不同(比如多元线性回归, 输入为一个向量, 输出为标量). 他的形式如下:
$$y_i = tr(X^T \beta)  + \epsilon_i$$


我其实不是很理解他的应用领域, 查阅资料的时候可以发现他似乎更偏向于图像处理等领域, 因为矩阵结果可以保留更多的结构信息. 我尝试让 ai 工具提供一些与统计学习关系更密切的例子, 他给我的例子之一是多任务线性回归, 经过处理用 trace regression 的方式去处理这些数据. 但是在我看来这只是一些数学上的变化, 似乎对于估计效果来说并没有太大的意义.
##### Question
1. 对于trace regression的优点, 文献提到它可以共享子空间, 从而一定程度上解决样本数量过少的问题, 但是我们可以发现, 你的x矩阵, 也就是$X_i = x_i \cdot e_t^\top$, 其实也是根据对每个任务单独列式得到的, 也就是说实际上我们还是没有达到共享数据的目的是不是.
也就是说如果你一开始的数据是
$$\hat y_1 = {x_1}^T \beta_1$$
那么经过处理后, 你的形式变成了
$$\hat y_1 = tr([x_1,0,\dots,0]^T[\beta_1, \beta_2, \dots, \beta_t] ) = {x_1}^T\beta_1$$
也就是说, 最后你还是只使用了 $x_1$ 的数据, 并没有使用到别的样本的数据.
2. 回答里经常提到矩阵的低秩可以共享公因子的估计, 这是什么意思, 我不太理解
##### Answer
我觉得可以从低秩的角度来切入: 他论证 trace regression 的优点的时候经常围绕这一点展开, 低秩到底意味着什么?
如果这个时候我们发现 trace regression 的系数矩阵 B 是低秩的, 那当我们以每个任务的视角去观察他, 相当于有类似结构:
$$\displaylines{task1: y_1 = x_1^T \beta_1 \\ task2: y_2 = x_2^T \beta_2\\ task3: y_3 = x_3^T \beta_3 = x_3^T (k_1\beta_2 + k_2\beta_1)}$$
所以如果我们估计出了前两个任务的结果, 我们就可以转而估计组合的数值而不是直接估计 $\beta_3$ .
当我们成功转化为这种形式后, 我们就可以估计两个标量, 而不是向量 $\beta_3$ , 所以只要特征的数量大于二, 我们就得到了优化的效果. 
综上, trace regression 似乎是在任务之间相互关联的时候会有较大的优势, 在多任务之间有关联的时候似乎可以很好的达到去除噪声的效果. 
##### 想法
我们已知可以通过一些变换把多任务线性回归转变为 trace regression, 而trace regression 又有降噪的能力. 那我们可不可以先使用添加噪声的方式对数据和模型进行隐私保护, 然后结合一些公共数据集, 将模型转化为 trace regression, 再使用 trans-learning 将模型迁移到别的相似模型上去? 
不对, 好像不行. 经过隐私保护的数据应该是只会对外部开放最后得到的模型接口, 我们无法得到他的数据, 也就无法通过模型转换进行降噪. 所以我们还是应该先对样本进行 trace regression 和隐私保护, 再拿得到的模型去做迁移学习.

#### 参数估计
我们一般使用最小二乘方法, 结合正则项对其进行约束, 以此来进行参数估计, 如:
$$L(B) = \frac{1}{n} \sum_{i = 1}^{n} (y_i - \text{Tr}(X_i^\top B))^2 + \lambda R(B)$$
这里的 $R(B)$ 是正则项, 可以根据不同的问题选择不同的约束.
对于隐私保护, 首先我们可以先尝试了解为什么需要隐私保护(没有保护时的攻击手法是什么样的)
### 无保护情况下的隐私攻击
我们可以看到这篇经典论文[[隐私攻击Membership_Inference_Attacks_Against_Machine_Learning_Models.pdf]]中提到的攻击方法, 即构造一个表现和目标模型基本相同的影子模型, 然后可以根据影子模型对不同数据点位置的判断(本来他就已知这个点是否在训练集里), 就可以训练出攻击模型, 根据数据点的表现判断这个数据点是训练数据还是测试数据. 然后用在目标模型上, 判断某个成员是否在目标模型的训练数据中.
而如果我们需要做隐私保护, 就需要知道他是如何判断的. 事实上, 他是通过模型对不同的数据点的置信度来判断的. 大致类似于会设立一个标准, 置信度高于这个标准则会被认为数据点在训练集中.
### Differentially Private
#### 基本概念
##### Definition: Differentially Private
给定两个相邻数据集 $D_1, D_2$，随机化函数 $\mathcal{M}: D \to \mathbb{R}^d$，则 $\mathcal{M}$满足$(\varepsilon, \delta)$- 差分隐私当且仅当满足如下条件：$$\text{Pr}[\mathcal{M}(D_1) \in S] \leq \text{Pr}[\mathcal{M}(D_2) \in S] \times e^{\varepsilon} + \delta \quad $$其中$S \subset \text{Range}(\mathcal{M})$，$\varepsilon$ 称为隐私预算，$\delta$ 为松弛项。$\delta$为$0$时，我们又称算法$\mathcal{M}$满足$\varepsilon -$ 差分隐私，$\delta > 0$时算法$\mathcal{M}$以$\delta$的概率不满足$\varepsilon -$ 差分隐私。$(\varepsilon, \delta)$ 的大小体现了隐私保护的程度，$\varepsilon$和$\delta$ 越接近于零，隐私保护的程度越高。
这个定义说明我们希望在任意选出的集合 S 中, 分别以 $D_1, D_2$ 为元数据集处理后, 属于 S 的概率的差异不大. 二者的差异越小, 隐私保护程度越高, 模型拟合的精度越低

我们希望做到隐私保护, 就意味着我们希望这两个数据集的表现尽可能的接近. 我们使用敏感度来判断两个数据集对于算法 f 的差异:
##### Definition: 敏感度
对于任意两个相邻数据集 $D_1, D_2$，算法 $f: D \to \mathbb{R}^d$的敏感度定义为：$$\Delta = \max_{D_1, D_2 \text{ 相邻}} \| f(D_1) - f(D_2) \| \tag{9.1.2}$$其中$\| \cdot \|$取$l_1 -$或$l_2 -$范数分别可以得到$l_1 -$敏感度和$l_2 -$ 敏感度，对应不同的噪声机制。敏感度衡量了单个数据改变对算法输出的影响，为了保证差分隐私，用于扰动的随机噪声大小将正比于算法的敏感度。
##### Notice
这里提到的算法 $f : D \to \mathbb R^d$ 表示不同的映射, 这只是一个指标, 用来判断数据点的存在与否对于模型的咩个特征的影响大不大.

##### Question
为什么敏感度分析的时候会加上一个 max? 是不是指每次取不同的训练集得到的模型之间的而差异? 但是查资料又说这里的 max 体现的是一种最坏情况, 是 f 所固有的是为什么呢

##### Laplace noise
对于算法 $f: D \rightarrow \mathbb{R}^d$，$f$ 的 $l_1 -$ 敏感度为$\Delta_1$，则拉普拉斯机制：

$$\mathcal{M}(D) = f(D) + \eta \tag{9.1.3}$$满足$\varepsilon -$差分隐私，其中$\eta \sim \text{Lap} \left( \frac{\Delta_1}{\varepsilon} \right)^d$。
###### Proof: Laplace noise
一维拉普拉斯分布的密度函数为：
$$f(x \mid b) = \frac{1}{2b} \exp \left( - \frac{|x|}{b} \right)$$因此当$\mathcal{M}(D_1), \mathcal{M}(D_2)$得到同一个输出$y$ 时，其概率之比为：
$$\begin{align*}
\frac{\text{Pr}[\mathcal{M}(D_1) = y]}{\text{Pr}[\mathcal{M}(D_2) = y]}&=\frac{\text{Pr}[\eta_1 = y - f(D_1)]}{\text{Pr}[\eta_2 = y - f(D_2)]}=\prod_{i = 1}^{d}\left(\frac{\exp\left(-\frac{\varepsilon|y_i - f(D_1)_i|}{\Delta}\right)}{\exp\left(-\frac{\varepsilon|y_i - f(D_2)_i|}{\Delta}\right)}\right)\\
&=\prod_{i = 1}^{d}\exp\left(\frac{\varepsilon(|y_i - f(D_1)_i| - |y_i - f(D_2)_i|)}{\Delta}\right)\\
&\leq\prod_{i = 1}^{d}\exp\left(\frac{\varepsilon|f(D_1)_i - f(D_2)_i|}{\Delta}\right)\\
&=\exp\left(\frac{\varepsilon\|f(D_1) - f(D_2)\|_{l_1}}{\Delta}\right)\\
&\leq\exp(\varepsilon),
\end{align*}$$
根据 [[#Definition Differentially Private]] ，拉普拉斯机制满足$\varepsilon -$ 差分隐私。

由此我们可以看出, 差分隐私的工作原理就是给数据添加一些噪声, 而这些噪声可以有不同的分布, 也就带来了不同的隐私保护的方法. 这些证明对我们来说似乎不是很重要, 因为我们只需要使用这样的保护方法就可以了, 不需要知道为什么这样的早上可以符合差分隐私的定义.

##### Properties: 后处理性
令 $\mathcal{M}: D \to \mathcal{Y}$为满足$(\varepsilon, \delta)$- 差分隐私的随机算法, 函数$f: \mathcal{Y} \to \mathcal{Z}$, 则 $\mathcal{M}' \stackrel{\text{def}}{=} f(\mathcal{M}): D \to \mathcal{Z}$也满足$(\varepsilon, \delta)$ - 差分隐私。
这个性质看起来有点像是某种"不可逆性": 只要我们对某组数据进行了 DP 处理, 后续无论对他进行什么操作, 都无法暴露某个样本点是否还在数据集中. *我觉得这有点像是本来有一个瓶子的彩色小球, 攻击者则是想知道某一个红色的小球是否在瓶子里. 我们加入噪声的过程就相当于向瓶子里又加入了一些彩色的小球, 只不过这些小球的颜色符合某种分布(Laplace, Gaussian etc), 只要我向其中加入了这样的噪声, 攻击者就在也无法判断某一个小球原本是否在瓶子中.*

##### 想法
所以我又有一个新的想法: 隐私保护的弊端之一是会增加噪声, 导致模型的精确性下降, 那我们能不能在隐私保护之后再进行降噪? 由于后处理性, 他还是会保持 DP 的性质.
应该不太可行..以刚才的瓶子为例, 如果说他可以保持 DP 性质, 说明我们还是无法分辨出哪些小球原本就在瓶子里. 折旧说明我们降噪过程中丢掉的小球里面其实有一部分是原本就在瓶子里的. 这样一来我们只会令模型的偏差更大, 降噪失去了他的作用. 不过从另一个角度来看, 如果我们降噪算法选的好的话, 我们在保留元数据大致分布的前提下去掉了噪声, 但是这个过程中一定会有部分噪声代替元数据留了下来, 只是他不再令数据的分布产生较大偏差. 这样一来我们也达到了隐私保护的目的: 因为攻击者不知道哪些是被替换过的.

##### Properties: 简单组成原理
令 $\mathcal{M}_i: D \to \mathcal{R}_i$分别为满足$(\varepsilon_i, \delta_i)$- 差分隐私的随机算法，其中$i \in [k]$，则它们的组合 $\mathcal{M}_{[k]}: D \to \prod_{i = 1}^{k} \mathcal{R}_i, \mathcal{M}_{[k]}(D) = (\mathcal{M}_1(D), \ldots, \mathcal{M}_k(D))$满足$(\sum_{i = 1}^{k} \varepsilon_i, \sum_{i = 1}^{k} \delta_i)$ - 差分隐私。

Q: 我不太理解这里的 $\prod$ 是什么意思, 为什么要乘起来..?
A: 这里的 $\prod$ 不是乘法, 是表示集合的笛卡尔积.

如果说后处理性是串行情况下的特性, 那我们就可以发发现简单组成原理就是并行情况下的 DP 特性. 我们这个时候要求不同的差分隐私作用在同一个数据集上, 而这样的 DP 多次叠加就一定会提高隐私泄漏的风险, 就像是对同一个数据集, 从不同的方式进行多次试探. 比如对于一个病人, 我们三次不同的 DP 处理分别处理了病人的身高, 体重, 年龄, 那攻击者就可以综合三次的置信度, 综合判断某一名病人是否在数据集中.

该定理说明了对于同一个数据集重复地查询，其隐私保护程度将线性地下降。
然而该定理所保证的隐私预算上界 $(\sum_{i = 1}^{k} \varepsilon_{i}, \sum_{i = 1}^{k} \delta_{i})$ 是过于宽松的。高级组成原理证明了一个更紧的隐私预算上界。

### 参数估计
从一开始我们就知道, 隐私保护的同时会给我们带来拟合精度下降的问题. 这一点就体现在这里. 

> 令$\mathcal{P}$表示支持于集合$\mathcal{X}$上的分布族，令$\boldsymbol{\theta}:\mathcal{P}\to\boldsymbol{\theta}\subseteq\mathbb{R}^d$为感兴趣的统计量，设从某概率分布$P\in\mathcal{P}$中抽样出$n$个独立同分布的数据集$\boldsymbol{X}=(x_1,\cdots,x_n)\in\mathcal{X}^n$。有了数据，我们可以通过估计量$M(\boldsymbol{X}):\mathcal{X}^n\to\boldsymbol{\theta}$去估计参数$\boldsymbol{\theta}(P)$，其中估计量$M(\boldsymbol{X})$来自于所有满足$(\varepsilon,\delta)-$差分隐私的估计量集合$\mathcal{M}_{\varepsilon,\delta}$。
> 估计量$M(\boldsymbol{X})$的表现由它到真实值$\boldsymbol{\theta}(P)$的距离衡量：令$\rho:\boldsymbol{\theta}\times\boldsymbol{\theta}\to\mathbb{R}^+$为由在$\boldsymbol{\theta}$上的范数$\|\cdot\|$导出的度量，即$\rho(\boldsymbol{\theta}_1,\boldsymbol{\theta}_2)=\|\boldsymbol{\theta}_1 - \boldsymbol{\theta}_2\|$，并令$l:\mathbb{R}^+\to\mathbb{R}^+$为一递增函数，在差分隐私约束下估计$\boldsymbol{\theta}(P)$的极大极小风险定义为
> $$\inf_{M\in\mathcal{M}}\sup_{\varepsilon,\delta}\mathbb{E}[l(\rho(M(\boldsymbol{X}),\boldsymbol{\theta}(P)))] \tag{9.2.1}$$
> 这个量刻画了最优的$(\varepsilon,\delta)-$差分隐私的估计量在$\mathcal{P}$上最差的表现。不同于$(9.2.1)$式，常用的无约束的极大极小风险
> $$\inf_{M}\sup_{P\in\mathcal{P}}\mathbb{E}[l(\rho(M(\boldsymbol{X}),\boldsymbol{\theta}(P)))] \tag{9.2.2}$$
> 为“隐私的成本”。

上述的部分比较绕, 大致上以显式的例子来理解的话类似于: 样本是一群病人, 他们所有人血压取值的集合是 $\mathcal{X}$, 他们的血压分布符合分布 P, 所有可能的P组合成了 $\mathcal{P}$,  $\boldsymbol{\theta}:\mathcal{P}\to\boldsymbol{\Theta}\subseteq\mathbb{R}^d$ 相当于是求这些人的平均血压.现在这些病人的血压分布是 P, 我们从中随机抽样 n 组样本, 通过$M(\boldsymbol{X}):\mathcal{X}^n\to\boldsymbol{\theta}$ 去估计 $\boldsymbol{\theta}(P)$, 也就是通过样本去估计总体. 然而我们同时要满足这个算法 $\mathcal{M}$ 是有隐私保护能力的, 比如说我们不直接计算样本数据的均值, 而是计算经过 Laplace 保护的均值. 