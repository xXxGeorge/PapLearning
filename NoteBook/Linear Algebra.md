---
tags:
  - BasicMath
  - LinearAlgebra
relation:
  - "[[Convex optimization]]"
  - "[[NumericalCalculate]]"
teacher: 左文杰
---
# INTRODUCTION
这个笔记其实是一个梳理, 我尝试同时兼顾梳理自己的思路和将自己的结果整理成讲义一样的形式, 但是最后的实践结果证明了这其实不太可能: 梳理的过程必然会很凌乱很跳脱, 这个时候的梳理在一段时间后重看会显得没有逻辑, 因为当时思考的很多前提都会想不起来了, 就像是失去了引理的定理证明...所以如果有需要重看, 其实是需要搭配教科书一起看的, 也就是说并没有完全符合我预期的要求
我还想在序言中写一些方法论的内容. 看数学书的过程有的时候会面临一种照不清方向的问题. 概念掌握之后面临的第一个挑战是定理. 定理必然与严密的证明相关联, 同时伴随着无数引理的证明. 看书时的一大陷阱就是容易在引理中丢失思路. 我一直认为学习应该复合发展的客观规律, 我们应该理解学科的发展是从提出问题开始. 所以对于定理学习, 我们先要知道我们为什么需要这个定理(一本合格的教材应该会引导你的思路, 发现这个定理的必要性), 在此前提下, 我们才能知道为什么需要这些引理, 构建起知识体系而不会乱.
学习概念的时候还有一点是要结合实例. 脱离实例的学习不会在脑海中形成具像化的理解, 只有结合实例学习(或是经过大量的运用), 才能在需要运用的时候在脑海中出现图像化一般的运用, 尤其是高代的几何特性, 注定他的学习会和具像化的理解息息相关.
现在说回学科本身吧. 高等代数, 最重要的概念就是线性空间, 线性映射, 以及内积空间. 他的所有论证都是为最后构建一个结构化的空间服务, 一整本书都在讲解这样一个空间的性质. 其中有几个点可以将繁琐的知识点串联起来: 
- 特征值. 特征值对应特征向量, 一个等式很直观的写出了线性映射的几何特性(从基底角度), 同时体现了特征空间对线性空间的划分. 特征值本身的正负很好理解的对应了对空间的不同操作, 延展, 旋转, 反转等等.数字上的特性又与行列式, 代数以及几何重数联系起来, 进而影响到代数层面的可对角化, 同步的带着人去理解对角化的根本意义(不同基底视角下的线性映射, 也由此让人懂了相似矩阵之间代表的意义, 有些结论就不证自明, 比如说特征值相同)
- 映射的表示矩阵. 我们知道任意 n 维线性空间会同构于一个 $\mathbb R^n$ 的列向量空间. 这样就将向量这个几何特征联系到了代数角度. 有这样一个同构做桥梁, 就可以将矩阵这样一个代数定义与具体的空间变换行为联系起来, 给矩阵的特征安排上几何的理解, 同样可以令很多性质不证自明(比如非满秩的矩阵为什么不可逆, 因为矩阵划分为列向量的分块矩阵后每个分块可以理解为线性映射之后新空间的基底. 从两空间的维数差异就可以知道有没有丢失信息, 而矩阵求逆的过程就是信息复原的过程). 另一方面几何天生就难以进行细致研究, 通过这个同构可以使用代数工具对空间进行各种变形, 并从数字角度辅助一些抽象概念的理解.
先写这么多吧, 后面有新的结论再写


### *CHAPTER 1 行列式*

##### *Vandermonder行列式*

Vandermonde 矩阵 V 是一个 $n \times n$ 的矩阵，其元素由数列 $x_1, x_2, \dots, x_n$ 组成，具体定义为：
$$
V = \begin{pmatrix}

1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\

1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\

1 & x_3 & x_3^2 & \cdots & x_3^{n-1} \\

\vdots & \vdots & \vdots & \ddots & \vdots \\

1 & x_n & x_n^2 & \cdots & x_n^{n-1}

\end{pmatrix}
$$
**Vandermonde 行列式的计算**

Vandermonde 行列式的值有一个简单的公式，假设给定的数列是$x_1, x_2...x_n$ ，那么其对应的 Vandermonde 行列式 $\det(V)$ 为：
$$
\det(V) = \prod_{1 \leq i < j \leq n} (x_j - x_i)
$$
即，行列式的值是所有$x_j - x_i$（其中 $j > i$ ）的乘积。

**推导过程**

1. **行变换的第一步：减去第一行的倍数**

首先，从每一行（除了第一行）中减去第一行的相应倍数。这是通过使用行列式的线性性质来做的。具体地，对每一行$i （ i = 2, 3, \dots, n ）$，进行如下操作：
$$
\text{新行}_i = \text{原行}_i - \text{原行}_1 \cdot x_i
$$
这样，矩阵的第一列将全是 1，其他列会变成逐行递增的多项式形式。执行这些行变换后，得到的矩阵如下：
$$
V{\prime} = \begin{pmatrix}

1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\

0 & x_2 - x_1 & (x_2 - x_1) x_2 & \cdots & (x_2 - x_1) x_2^{n-2} \\

0 & x_3 - x_1 & (x_3 - x_1) x_3 & \cdots & (x_3 - x_1) x_3^{n-2} \\

\vdots & \vdots & \vdots & \ddots & \vdots \\

0 & x_n - x_1 & (x_n - x_1) x_n & \cdots & (x_n - x_1) x_n^{n-2}

\end{pmatrix}
$$


2. **行变换的第二步：重复上述过程**

继续对第二行到最后一行做类似的变换，减去相应倍数的前一行。这样，我们可以通过一系列的行变换，最终得到一个上三角矩阵。

经过这些行变换后，矩阵的结果是一个上三角矩阵，其中每个对角线元素为$(x_j - x_i)$（即 $x_j$和 $x_i$之间的差值）。具体地，得到的矩阵形状如下：
$$
V{\prime}{\prime} = \begin{pmatrix}

1 & * & * & \cdots & * \\

0 & (x_2 - x_1) & * & \cdots & * \\

0 & 0 & (x_3 - x_2)(x_3 - x_1) & \cdots & * \\

\vdots & \vdots & \vdots & \ddots & \vdots \\

0 & 0 & 0 & \cdots & (x_n - x_1)(x_n - x_2) \cdots (x_n - x_{n-1})

\end{pmatrix}
$$


**4. 行列式的最终计算**



由于现在得到的矩阵是一个上三角矩阵，行列式的值就是对角线上所有元素的乘积。因此，Vandermonde 行列式的值为：
$$
\det(V) = (x_2 - x_1)(x_3 - x_1)(x_3 - x_2)(x_4 - x_1)(x_4 - x_2)(x_4 - x_3) \dots (x_n - x_1)(x_n - x_2) \dots (x_n - x_{n-1})
$$

### *CHAPTER 2 矩阵*

奇异矩阵与非奇异矩阵:针对于方针而言,非方阵不讨论奇异性.

**判断一个矩阵是否奇异**

- 计算矩阵的行列式。如果行列式为零，矩阵就是奇异矩阵。

- 如果矩阵是方阵，并且可以通过初等变换（如高斯消元法）将其简化为一个包含零行的矩阵，那么它也是奇异矩阵。

**性质**:

> 1. **行列式为零**：
>
> 如果矩阵 A 的行列式 $\det(A) = 0$，则矩阵 A 是奇异矩阵。
>
> 2.**不可逆**：
>
> 如果矩阵是奇异矩阵，它没有逆矩阵。这意味着不存在一个矩阵 B，使得 AB = I 或 BA = I，其中 I 是单位矩阵。
>
> 3.**秩（Rank）不足满秩**：
>
> 奇异矩阵的秩 $\text{rank}(A)$ 小于它的阶数（即行数或列数）。如果矩阵的秩等于它的阶数，那么它是可逆矩阵，否则是奇异矩阵。
>
> 4.**线性相关**：
>
> 奇异矩阵的行（或列）是**线性相关**的，换句话说，矩阵的行（或列）不是线性无关的。也就是说，矩阵的某些行或列可以通过其他行或列的线性组合得到。
>
> 5. **零解**：
>
> 对于线性方程组 $A \mathbf{x} = \mathbf{b}$，如果 A 是奇异矩阵，那么方程组要么没有解，要么有无穷多解。没有解的情况通常是因为 A 不是满秩的，导致方程组的系数矩阵无法提供足够的独立信息来唯一确定解。



##### 矩阵的消去律 p67(gb)

关于矩阵不满足消去律的部分

若此时有$AB = AC$, $A \not= O$, 是否能推出$B = C$  ?

- 假设有一个矩阵$ A_{m\times 3}$,    $ B_{3\times x}$,  $ C_{3\times x}$,  则显然可以将A, B, C写为列向量的形式.  不妨将B, C除第一项以外的部分均设为零向量, 而第一部分为

$$
B_1 = \begin{bmatrix}b_1\\b_2 \\b_3\end{bmatrix} , C_1 = \begin{bmatrix}c_1\\c_2\\c_3\end{bmatrix}
$$

则矩阵乘法可以看作是线性方程组, 即是有
$$
A_1b_1 + A_2b_2 + A_3b_3 = M\\A_1c_1 + A_2c_2 + A_3c_3 = M
$$
第一个方程式确定M之后, 我们可以解第二个方程式; 而展开之后显然方程式中有三个未知数, 而方程式的个数由m决定,所以可以知道,至少当$ m < 3$时消去律肯定不成立, 因为此方程组有无数个解

##### 乘以满秩阵对秩的影响 p131(wb)

在p140(gb)中有推论3.6.5:

> 任一矩阵与一非异阵相乘, 其秩不变

然而这个推论有一个局限, 即只有对方阵才可以谈论奇异性. 在处理 p130, 例3.15(wb)时可以发现其实定理可以拓展为

> 任一矩阵与一满秩阵相乘, 其秩不变(行满秩或列满秩均可)

例3.15:

设向量组$\alpha_1, \alpha_2, \alpha_3, ..., \alpha_r$线性无关, 又
$$

$$


### CHAPTER 4 线性映射

#### 概念
![[线性映射以及同构的概念.png]]
特殊一点的映射有零映射或恒等映射, 而这两者也可以用作学习特征值与特征向量的例子.
线性映射的定义中要求V与U同为数域K上的线性空间, 不同数域上线性空间的映射不是线性映射

#### 线性映射的运算

符合正常的运算, 乘法为复合运算.

#### 线性映射与矩阵
我们需要注意到线性映射与普通映射最大的不同点在于, 普通的映射有可能写不出一个通用的公式化的映射来表示, 也就是说不是所有时候都可以用一个$f(x)$ 来表示; 而对于线性映射来说却是一定的: 因为你不可能用任意两个元素相互连线的方式来定义一个线性映射, 两个线性空间内部的元素有固定的对应关系, 也就是在特定基下的唯一表示.

*也就是说, 线性空间 $V->V'$的线性映射完全被它在V上的一个基的作用所决定.*  (从此我们证明两个线性映射相等就可以从任意元素的映射相等和对基底的映射相等两个方向去证明)

如果我们现在有一组基$e_1, e_2 \dots e_n$作为线性空间A的基底, 有另一组基$\beta_1, \beta_2 \dots \beta_m$来作为线性空间B的基底, 若线性映射已经规定了$e_i$ 到 $\beta_i$ 的对应关系, 便可以对任意元素进行以基底的展开, 再运用线性映射的性质加以拆分, 最后再用基底的对应关系加以替换.

对线性映射的理解应该多从两个性质不同的线性空间出发, 比如一个从$\mathbb R^n \to \mathbb M_n(\mathbb R)$, 这样在用哪个空间做基底等问题上就不会犯迷糊.

**相似矩阵有相同的行列式，特征值，极小多项式**。
##### 对不可逆映射的解释
从这个角度就可以看出为什么 $n \not= m$ 时映射不可逆了: 假设$n > m$, 则对于$V'$来说至少有一个$V$的基底 $e_i$ 需要用两个$V'$的基底 $f(\beta_i, \beta_j)$ 表示. 这样在进行逆映射的时候就无法映射出 $e_i$ 来. 有人会有疑问, 为什么不用一开始的表示, 用
$f(\beta_i, \beta_j)$ 表示回去呢? 因为这样当遇到$V'$ 中的元素$\beta = f(\beta_i, \beta_j)$ 时, 就会分别出现两种映射方式, 显然不行.

##### Question1
对数域 K 上的线性映射$\Phi : V \to  V'$, $\Phi$ 是基于V的基底$\alpha$ , 指向V'的基底$\beta$ , 现在显然可能有两个不同的矩阵, A: 表示矩阵, 即$\Phi(\alpha) = A\alpha$ , 以及B: 过渡矩阵, 得到的过程是:
$$
\displaylines{
\Phi(\alpha_1) = b_{11}\beta_1 + b_{12}\beta_2 + \dots + b_{1m}\beta_m   \\  \Phi(\alpha_2) = b_{21}\beta_1 + b_{22}\beta_2 + \dots + b_{2m}\beta_m \\ \dots \\ \Phi(\alpha_n) = b_{n1}\beta_1 + b_{n2}\beta_2 + \dots + b_{nm}\beta_m \tag{1}
}
$$
则A与B之间存在关系吗, 是否任何时候二者都存在呢, 以及如何求两者.
##### Answer1(maybe wrong)
>确实这两者之间有关系而且显然不是互逆. 因为只需简单变形我们就可以得到$$A\alpha =  B\beta \tag{2}$$
 而他表示的意义也很简单, 就是通过两种方式去表示线性映射$\Phi$.

---

与之相关的一节: 高等代数(复旦), p190;
[[高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library).pdf#page=202|高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library), p.202]]
这两者之间似乎有些矛盾并且令我很难受, 不过我现在怀疑这是因为我没有分清系数矩阵以及向量组, 也就是说在A中的等式中, $\alpha, \beta$表示的是向量组, 此处的乘积表示的也是向量本身; 而[[高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library).pdf#page=202|高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library), p.202]]中提到的这个计算等式则是针对于系数, 等式针对的是如何计算另一个基中的系数.

![[线性映射和矩阵的关系.png]]

现在我的想法是, 这里的$\Phi$对应的就是我的$A$ (如果这里的$\Phi$ 可以用矩阵表示?), 这里的$\Phi_A$ 对应的就是$B^T$ , $\eta_1, \eta_2$  则是单纯的将向量映射到基底的对应表示. 
##### Answer2
**我现在的想法是, 我第一步的处理, 也就是将$\Phi(\alpha) \rightarrow A\alpha$** ,是他妈有问题的. 因为我的理论依据是线性映射与数域$K$上的矩阵的同构关系, 但是深入研究就会发现这个对应关系应该表述为, 对线性空间的变换$\Phi \rightarrow$ 对系数的变换$\Phi_A$. 因为$A\alpha$ 这个表达式本身展开似乎并没有意义我突然发现... 比如说我们想要将一个$R^3$ 上的线性空间映射到一个$[K]^3$上, 对列向量的组合怎么都组合不出来一个多项式.
利用$\eta$辅助计算后我终于得到了表示矩阵和过渡矩阵的关系..
$$ A = B^T$$
[对线性映射的表示矩阵的推导](obsidian://open?vault=PapLearning&file=Pictures%2F%E7%BA%BF%E6%80%A7%E6%98%A0%E5%B0%84%E8%A1%A8%E7%A4%BA%E7%9F%A9%E9%98%B5.pdf)

详细总结:
>[[线性映射角度理解过渡矩阵, 计算矩阵(对向量), 以及表示矩阵的关系]]

##### 补丁
线性变换不换基底.所以线性变换前后只有一个基底.

#### 线性映射的 Kernel 和 Image

若数域K上的线性映射$\Phi : V \to V'$, 则$Ker(\Phi)$ 是$V'$中的零向量在$V$中的原像集, 即
$$  Ker (\Phi) : =\{\alpha \in V| \Phi(\alpha) = o'\} \tag{3}$$
显然$Ker(\Phi)$是V的一个子空间
$Ker(\Phi)$的维数称为$\Phi$ 的零度.

线性空间$V$的像(值域)记为 $Im(\Phi)$
显然$Im(\Phi)$也是一个子空间
$Im(\Phi)$的的维数称为$\Phi$ 的秩, 记为$r(\Phi)$.
##### 利用核和像判断单射与满射
命题一: $A$是单射$\Leftrightarrow Ker(A) = 0$
命题二: $A$是满射$\Leftrightarrow Im(A) = V'$

##### Theorem I (线性映射的维数公式)
设$\Phi$ 是$\mathbb{K}$上$n$维线性空间$V$到$\mathbb{K}$上$m$维线性空间$U$的线性映射, 则
$$ dim\,Ker\,\Phi + dim\,Im\,\Phi = dim\,V\tag{4}$$
详细证明: [[高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library).pdf#page=211|高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library), p.211]]

#### 不变子空间

#### 特征值和特征向量(Eigenvalue&Eigenvector)
##### Definition
若$\Phi$ 是一个线性变换，若V中存在一个非零向量$e$, 使得
$$ \Phi(e) = \lambda_0 e$$
则称$\lambda_0, e$为$\Phi$的一个特征值与属于特征值$\lambda_0$ 的一个特征向量, 且$e$构成了一个特征子空间, 记为$V_{\lambda_0}$.
又因为上式可以用$\Phi$在某组基下的表示矩阵写为 
$$A\alpha = \lambda_0 \alpha \to (\lambda_0 I_n - A)\alpha = 0$$所以这个特征子空间也可以写为这个方程的解空间.
白皮书例题的第一题(Example 6.1)可以加深对于此概念以及线性映射的作用的理解, 对于$\Phi$而言, e只是一个向量, 需不需要用坐标表示无关紧要. 可以用分块矩阵连接非坐标的表示与坐标的表示.
[[Example 6.1.pdf]]
**特征值决定了一个特征空间，不同的特征值对应的一定是不相关的特征空间, 因此我们可以发现不同特征值对应的特征向量是正交的(在PCA里面用过)**。这是一个非常重要的点，从这个角度我们也可以理解很多题目。如[[HW0602_WB_6.2_LA.png]],这题我一开始也是从代数变换的角度去理解，但我看了[k重特征根与特征向量关系](https://www.bilibili.com/video/BV1uP411J7Vi/?spm_id_from=333.1391.0.0&vd_source=320e99e9d1061fcda4da2e64ad20113b)后突然发现我们也可以从另一个角度去思考这个问题。将问题改为讨论$\lambda_1, \lambda_2$不同关系下$\alpha_1 + \alpha_2$是特征向量的可能性。若二者不等由于，则他们对应的特征空间一定是无关的（无论这两个特征值是几重的，对应空间维数是多少），而他们所组合出来的向量一定不属于他们二者，除非其中一个特征向量等于零，但是又因为特征空间是去零的，所以不可能
那有没有可能这个和落在第三个特征值张成的特征空间内呢？这也是不可能的，因为另一个特征空间必然也是与这两个无关的。
而如果$\lambda_1 = \lambda_2$，则两个向量都在同一个空间内，组合自然也是特征空间内的元素，这也应征了性质[[Linear Algebra#Inference]].
##### Inference
if $\alpha_1, \alpha_2$ are $\lambda_0$'s eigenvectors, then $\alpha_1 + \alpha_2$ is also $\lambda_0$'s eigenvector.
if $\alpha_1, \alpha_2,\dots,\alpha_n$ are $\lambda_0$'s eigrnvectors, $\sum_{i = 1}^{n}{a_i\alpha_i}$ is also $\lambda_0$'s eigenvector.
##### LA HW0601

这里有息息相关的三道题:
[[HW0601_07:08:09_PracticeSet LA.pdf]]
$\underline{Example 07}$
对于第一题, 一开始我的思考是从题目可知, $\alpha, \beta$正交, 故可以得到对角线元素的乘积为零, 并尝试继续寻找这一结论的用处, 不过最后没有找到
不过从另一个角度来看的话, 我们对于这种情况, 可以尝试在题目中凑出已知的条件. 对于矩阵来说, 虽然不符合交换律但我们仍然可以利用乘法来达到换位的目的, 再结合矩阵的转置不会影响行列式的值就可以得到结论
$\underline{Example 09}$
这一题也是一样. 虽然答案给出的解法是[[利用Example 08解答.jpg]], 但是由于我没做出来08, 所以完全没有想到.. 但是我从07中收获了一点灵感, 用了[[另一种思路.pdf]]尝试找出这个矩阵的特性(这是一种猜测, 即这道题是基于此矩阵某个广泛的特有性质出的, 而不是针对这个矩阵本身数值或组合上的特殊性). 成功找到了特征值和特征向量. 不过这个解法还是有缺陷, 就是我并没有证明这个特征值是A全部的特征值, 后续还需要证明一下唯一性.
#### 对角化
##### Theorem
设$A$为$n$阶方阵, 则$A$相似于对角阵的充分必要条件是$A$有$n$个线性无关的特征向量(此时称$A$为可对角化矩阵)
>应该注意到此时这些特征向量不需要属于不同的特征值. 这一点在学习了[[Linear Algebra#特征值和特征向量(Eigenvalue&Eigenvector)]]之后其实可以从空间角度理解。如果这是一个n重特征根，且这个特征子空间的维数大于一，显然可以找到两个线性无关的向量，但他们都属于同一个特征根。

Proof: 对于线性映射$\mathbb{A}$, 其计算矩阵为A.
$$for\,eigvalue\,\alpha_i, there's\, \mathbb{A}(\alpha_1, \alpha_2 ,\dots, \alpha_n) = (\lambda\alpha_1, \lambda\alpha_2,\dots,\lambda\alpha_n)$$这里的$\alpha_i$均是列向量, 整体构成分块矩阵, 只是将多个特征值的等式写在一起上式可以写为$$\displaylines{\mathbb{A}(\alpha_1, \alpha_2 ,\dots, \alpha_n) =(\lambda\alpha_1, \lambda\alpha_2,\dots,\lambda\alpha_n) \\\Rightarrow A(\alpha_1, \alpha_2 ,\dots, \alpha_n) =(\alpha_1,\dots,\alpha_n) diag(\lambda_1,\dots,\lambda_n)\\= (\alpha_1,\dots,\alpha_n)\begin{pmatrix}\lambda_1&0&\dots&0\\0&\lambda_2&\dots&0\\&&\dots\\0&0&\dots&\lambda_n\end{pmatrix}}$$
对两遍乘以$(\alpha_1, \alpha_2 ,\dots, \alpha_n)^{-1}$后即可得到A与一个对角阵相似.$$ \displaylines{A = (\alpha_1,\dots,\alpha_n) diag(\lambda_1,\dots,\lambda_n)(\alpha_1,\dots,\alpha_n) ^{-1} \\ \Rightarrow A \sim diag(\lambda_1,\dots,\lambda_n)}$$
从先后来说，我们先有对对角化的需求，找出了这个对角矩阵以及使他成立的$P，P^{-1}$后才定义了特征值和特征向量。
由此可以知道这个过程中需要这个分块矩阵满秩. 特征向量线性无关. 又因为这个dim(V) = n, 所以这些特征向量组成基底
而对于$diag(\lambda_1,\dots, \lambda_n)$组成的对角阵而言并没有要求$\lambda_i$是否不相等. 结合同一个特征值可以有不同的特征向量即可得到, 完全可以有对角矩阵中出现同样的特征值的情况, 只要这个特征值提供的两个特征向量可以继续满足分块矩阵的满秩条件, 也就是这些所有特征向量都是线性无关的, 则线性变换$\mathbb{A}$可以正常对角化.
[对角化推导的思考过程](obsidian://open?vault=PapLearning&file=Pictures%2F%E5%AF%B9%E5%AF%B9%E8%A7%92%E5%8C%96%E7%9A%84%E6%80%9D%E8%80%83%E8%BF%87%E7%A8%8B.pdf)
所以我们可以得到对是否能对角化的判断
实对称情况下显然可以直接得到; 若是n个特征值之间没有重根，那这些不同的特征值之间的特征子空间也是无交集的，也可以得到; 当有重根时，因为n重根对应的线性无关特征值不超过n个，所以必须要每个n重根都张成n重特征呢个子空间。

#### 极小多项式（零化多项式）


##### Theorem II (Cayley-Hamilton theorem)


### CHAPTER 5 多项式

[一些例题](https://zhuanlan.zhihu.com/p/4666948586)

#### 多项式组成线性空间(无限维)

多项式存在零元, 负元, 满足数乘, 分配律等等, 满足线性空间的一切法则, 故多项式组成了一个线性空间. 不过是一个无限维的线性空间(如果不限制多项式的次数）。

在证明不同的基之间满足线性无关时, 将等式两边视为两个多项式, 比较零多项式与基之间的系数关系.

#### 一元多项式环

此时我们发现, 包括一元多项式, 矩阵, 整数集等等集合都具有某些共同特性, 所以我们可以将它们抽象的提炼出来得到一个新的代数结构, 环.

> 对环来说, 需要存在两个运算, 加法以及乘法.
>
> 加法:
>
> - 满足结合律, 交换律
> - 存在零元和逆元
>
> 乘法:
>
> - 满足结合律, 分配律(左分配律和右分配律)

此处对乘法的分配律要求是两个都满足, 只满足一个不被定义为环.

#### 多项式的计算

多项式的和....
多项式的乘积，就是用排列组合，将结果按照基底进行排列。简化后可得k次项的系数为$\sum_{i+j = k}{}{a_ib_j}$ .

##### Theorem I
1. $deg(f(x) + g(x)) \leq max(deg(f(x)), deg(g(x)))$
2. $deg(f(x)g(x)) = deg(f(x)) + deg(g(x))$
3. $f(x) \not= 0, g(x) \not= 0 \Rightarrow f(x)g(x) \not=0$
需要注意一点就是我一开始把 degree 和 dimension 弄混了……$deg$是多项式空间里特有的，$deg(0) = -\infty，deg(c) = 0，dim(\mathbb{K}[x]) = \infty$

##### Inference
1. $f(x)g(x) = 0 \Rightarrow f(x) = 0 /(x) = 0$
2. $f(x)g(x) = f(x)h(x), f(x)\not=0 \Rightarrow g(x) = h(x)$

#### 多项式的整除
##### Definition
![[多项式整除定义.png]]
$g(x)|f(x), g(x)\not=0 \rightarrow q(x) = \frac{f(x)}{g(x)}$ 是$f(x)$的商$q(x)$

##### Property
 ![[整除性质.png]]
 都很好证明，略去证明过程

##### 带余除法(Euclidean Division)

存在性以及唯一性：
>[[高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library).pdf#page=226|高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library), p.226]]

其中的数学归纳法可以看一下
#### 最大公因式(g.c.d.)

##### Definition
[[高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library).pdf#page=228|高等代数学 (姚慕生 吴泉水 谢启鸿) (Z-Library), p.228]]

##### Euclid辗转相除法求g.c.d.
为了求$f(x)$与$g(x)$的最大公因式, 不妨设$deg(f(x))>deg(g(x))$, 则由带余除法可得有$f(x) = g(x)h(x) + r(x)$, 移项可得$f(x)- g(x)h(x) = r(x)$, 所以$r(x)$也是$d(x)$的倍式, 由此不断迭代(用$g(x)$除$r(x)$), 直到余数为零, 则可以得到最大公因子.
$$
\displaylines{f(x) = h(x)g(x) + r(x)\\g(x) = h'(x)r(x) + r'(x)\\r(x) = h''(x)r'(x)}
$$
此时可得$r'(x)$ 为最大公因式
上述过程简而言之就是,$g(x)和r(x)$始终以$d(x)$为公因式, 因为$deg(r(x))$一直在减小, 所以最后会出现$g(x) = h(x)r(x)$的情况, 显然此时公因式就是$r(x)$.
Euclid法还有一个特点是不会因为数域的扩大而变化，由过程即可得到
##### Theorem I
$$\displaylines{f(x), g(x) \in \mathbb{K}[x] , d(x) = (f(x), g(x))\to  \\ \exists u(x), v(x) \in \mathbb{K}[x],  f(x)u(x) + g(x)v(x) = d(x)}$$
此时$u{x}, v(x)$不唯一，比如说不妨令$u(x) \to u(x) + g(x), v(x)\to v(x) - f(x)$，此时仍满足等式。
##### Theorem II(Bezout theorem)
$$\displaylines{(f(x), g(x)) = 1 \Leftrightarrow \\ \exists u(x), v(x) \in \mathbb{K}[x]\to f(x)u(x) + g(x)v(x) = 1}$$
Proof:
$\Rightarrow$ : 由Theorem I可得
$\Leftarrow$ : 由RSH可得1为$f(x), g(x)$ 的组合, 所以$d(x)|1$, 所以$d(x) = 1$.
当我们对$deg(u(x)), deg(v(x))$进行限制，则此处的$u(x), v(x)$唯一，证明如下：
Proof： 
$$ \displaylines{if\,\, \exists  \,\,u'(x), v'(x), \rightarrow f(x)u'(x) + g(x)v'(x) = 1 \\ \Rightarrow(u(x) - u'(x))f(x) = -(v(x) - v'(x))g(x) \\  \Rightarrow f(x)| (v(x) - v'(x))(since f(x)  \nmid g(x))\\ \Rightarrow v'(x) = v(x)+ f(x)a(x),\,\, u'(x) = u(x) - g(x)b(x)}$$
也就是说我们得到了$u'(x),v'(x)$的通项公式。

这个式子是目前唯一的对互素的有效翻译

##### Inference
1. 若$(f(x),g(x)) = 1$ ，且$f(x)|g(x)h(x)$，则$f(x)|h(x)$。虽然显然但是证明好玩

##### Theorem III(Chinese Remainder Theorem, CRT)
[详细讲解CRT](https://www.bilibili.com/video/BV1a54y1C7N3/?spm_id_from=333.337.search-card.all.click&vd_source=320e99e9d1061fcda4da2e64ad20113b)
稍后整理.

#### 因式分解
##### Lemma I
>设$f(x)$ 是数域$\mathbb{K}$上的不可约多项式， 则对$\mathbb{K}$上的任一多项式$g(x)$, 或者$f(x)|g(x)$, 或者$(f(x), g(x)) = 1$  

proof:
分类讨论即可。要么整除;不整除时二者的公因子需要整除f(x), f(x)的因子只有1，f(x), 又因为不整除，得证。
##### Theorem

[^1]: 此处应该注意
#### 多项式函数

##### Definition 
设$f(x) \in \mathbb{K[x]}, b\in\mathbb{K}$, 若$f(b) = 0$则称$b$是$f(x)$的一个根或零点。

##### Theorem(余数定理)
if $f(x)\in\mathbb{K[x]}, b\in\mathbb{K}$ ,then $\exists g(x)\in\mathbb{K(x)}$, s.t. $\,f(x)=(x-b)g(x)+f(b)$.
Specially, $b$ is f(x)'s root iff $(x-b)|f(x)$.

#### 复系数多项式

#### 实系数多项式和有理系数多项式
##### Theorem
对多项式$f(x)$, 若$a + bi$是其根，则其共轭复数也是。
所以虚部不为零的根一定成对出现
##### **Theorem**


### 二次型(Quadratic Form)
#### 8.1 二次型的化简与矩阵的合同

在解析几何中，我们曾经学过二次曲线及二次曲面的分类。以平面二次曲线为例，一条二次曲线可以由一个二元二次方程给出：
$$ax^{2} + bxy + cy^{2} + dx + ey + f = 0. \quad (8.1.1)$$要区分 (8.1.1) 式是哪一种曲线（椭圆、双曲线、抛物线或其退化形式），我们通常分两步来做：首先将坐标轴旋转一个角度以消去$xy$项，再作坐标轴的平移以消去一次项。这里的关键是消去$xy$项，通常的坐标变换公式为$$\begin{cases}
x = x'\cos\theta - y'\sin\theta, \\
y = x'\sin\theta + y'\cos\theta.
\end{cases} \quad (8.1.2)$$从线性空间与线性变换的角度来看，(8.1.2) 式表示平面上的一个线性变换。因此二次曲线分类的关键是给出一个线性变换，使 (8.1.1) 式中的二次项只含平方项。这种情形也在空间二次曲面的分类时出现。类似的问题在数学的其他分支、物理、力学中也会遇到。为了讨论问题的方便，只考虑二次齐次多项式。

##### Definition 8.1.1
设$f$是数域$\mathbb{K}$上的$n$元二次齐次多项式：$$\begin{align*}
f(x_1, x_2, \cdots, x_n) &= a_{11}x_1^2 + 2a_{12}x_1x_2 + \cdots + 2a_{1n}x_1x_n \\
&+ a_{22}x_2^2 + \cdots + 2a_{2n}x_2x_n + \cdots + a_{nn}x_n^2, \quad (8.1.3)
\end{align*}$$称$f$为数域$\mathbb{K}$上的$n$元二次型，简称二次型。这里非平方项的系数采用$2a_{ij}$主要是为了以后矩阵表示的方便。
或者采用另一种定义, 即二次型是一种双线性映射(对角化时的特殊情况):$$\displaylines{B(x, y) = x^TAy \\ x = y, \,A  = A^T}$$
A被称为相伴矩阵或系数矩阵.
#### 矩阵的合同关系
二次型理论的基本问题是要寻找一个线性变换把它变成只含平方项. 由上面我们知道, 二次型与对称阵一一对应, 而线性变换可以用矩阵来表示. 自然地, 二次型的变换与矩阵有着密切的关系. 现在我们来探讨这个关系.

设 $V$是$n$维线性空间, 二次型$f(x_1, x_2, \cdots, x_n)$可以看成是$V$上的二次函数. 即若设$V$的一组基为$\{e_1, e_2, \cdots, e_n\}$, 向量 $x$在这组基下的坐标为$x_1, x_2, \cdots, x_n$, 则 $f$便是向量$x$的函数. 现假设$\{f_1, f_2, \cdots, f_n\}$是$V$的另一组基, 向量$x$在$\{f_1, f_2, \cdots, f_n\}$下的坐标为$y_1, y_2, \cdots, y_n$. 记 $C = (c_{ij})$是从基$\{e_1, e_2, \cdots, e_n\}$到基$\{f_1, f_2, \cdots, f_n\}$的过渡矩阵, 则$$\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix}
=
\begin{pmatrix}
c_{11} & c_{12} & \cdots & c_{1n} \\
c_{21} & c_{22} & \cdots & c_{2n} \\
\vdots & \vdots & & \vdots \\
c_{n1} & c_{n2} & \cdots & c_{nn}
\end{pmatrix}
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}$$
或简记为

$$\boldsymbol{x} = \boldsymbol{C}\boldsymbol{y}$$将上式代入$f(x_1, x_2, \cdots, x_n) = \boldsymbol{x}'\boldsymbol{A}\boldsymbol{x}$，得

$$f(x_1, x_2, \cdots, x_n) = \boldsymbol{y}^T\boldsymbol{C}^T\boldsymbol{AC}\boldsymbol{y}$$

显然，$\boldsymbol{C}'\boldsymbol{AC}$仍是一个对称阵，故$\boldsymbol{y}'\boldsymbol{C}'\boldsymbol{AC}\boldsymbol{y}$是以$\boldsymbol{y}$为变元的二次型，记为$g(y_1, y_2, \cdots, y_n)$。由此我们可看出：若二次型 $f(x_1, x_2, \cdots, x_n)$所对应的对称阵为$\boldsymbol{A}$，则经过变量代换之后得到的二次型 $g(y_1, y_2, \cdots, y_n)$所对应的对称阵为$\boldsymbol{C}'\boldsymbol{AC}$。
##### Definition 8.1.2
设 $\boldsymbol{A}, \boldsymbol{B}$是数域$\mathbb{K}$上的$n$阶矩阵，若存在$n$阶 **非异阵** $\boldsymbol{C}$，使
$$\boldsymbol{B} = \boldsymbol{C}'\boldsymbol{AC}$$则称$\boldsymbol{B}$与$\boldsymbol{A}$是合同的，或称$\boldsymbol{B}$与$\boldsymbol{A}$具有合同关系。这里对$\mathbf C$的非异阵要求会在后面的化简中起到检验的作用.
不难证明，合同关系是一个等价关系，即

(1) 任一矩阵$\boldsymbol{A}$与自己合同.
(2) 若 $\boldsymbol{B}$与$\boldsymbol{A}$合同，则$\boldsymbol{A}$与$\boldsymbol{B}$合同.
(3) 若 $\boldsymbol{B}$与$\boldsymbol{A}$ 合同，$\boldsymbol{D}$与$\boldsymbol{B}$合同，则$\boldsymbol{D}$与$\boldsymbol{A}$合同.

因为一个二次型经变量代换后得到的二次型的相伴对称阵与原二次型的相伴对称阵是合同的，又因为只含平方项的二次型其相伴对称阵是一个对角阵，所以，化二次型为只含平方项等价于对对称阵 $\boldsymbol{A}$寻找非异阵$\boldsymbol{C}$，使 $\boldsymbol{C}'\boldsymbol{AC}$是一个对角阵。这一情形与矩阵相似关系颇为类似，在相似关系下我们希望找到一个非异阵$\boldsymbol{P}$，使 $\boldsymbol{P}^{-1}\boldsymbol{AP}$成为简单形式的矩阵（如 Jordan 标准型）。现在我们要找一个非异阵$\boldsymbol{C}$，使 $\boldsymbol{C}'\boldsymbol{AC}$为对角阵。因此二次型化简的问题相当于寻找合同关系下的标准型。我们能找到这样的矩阵$\boldsymbol{C}$ 吗？

首先我们来考察初等变换和矩阵合同的关系。
##### Lemma 8.1.1
对称阵 $A$的下列变换都是合同变换：
1. 对换$A$的第$i$行与第$j$行，再对换第$i$列与第$j$列；
2. 将非零常数$k$乘以$A$的第$i$行，再将$k$乘以第$i$列；
3. 将$A$的第$i$行乘以$k$加到第$j$行上，再将第$i$列乘以$k$加到第$j$列上。

**Proof**：上述变换相当于将一个初等矩阵左乘以$A$ 后再将这个初等矩阵的转置右乘之，因此是合同变换。(但显然不是必要条件)
##### Lemma 8.1.2
设$A$是数域$\mathbb{K}$上的非零对称阵，则必存在非异阵$C$，使 $C'AC$的第$(1,1)$元素不等于零。

**Proof**：若$a_{11} = 0$，而 $a_{ii} \neq 0$，则将 $A$的第一行与第$i$行对换，再将第一列与第$i$列对换，得到的矩阵的第$(1,1)$元素不为零。根据上述引理，这样得到的矩阵和原矩阵合同。

若所有的$a_{ii} = 0 (i = 1, 2, \cdots, n)$。设 $a_{ji} \neq 0 (i \neq j)$，将 $A$的第$j$行加到第$i$行上，再将第$j$列加到第$i$列上。因为$A$ 是对称阵，$a_{ji} = a_{ij} \neq 0$，于是第 $(i,i)$元素是$2a_{ij}$且不为零。再用前面的办法使第$(1,1)$ 元素不等于零。显然我们得到的矩阵和原矩阵仍合同。这就证明了结论。$\square$
##### Theorem 8.1.1
设$A$是数域$\mathbb{K}$上的$n$阶对称阵，则必存在$\mathbb{K}$上的$n$阶非异阵$C$，使 $C'AC$为对角阵。

**证明**：由上述引理(说明我们想用初等变换得到对角阵)，不妨设$A = (a_{ij})$中$a_{11} \neq 0$。若 $a_{i1} \neq 0$，则可将第一行乘以 $-a_{11}^{-1}a_{i1}$加到第$i$行上，再将第一列乘以$-a_{11}^{-1}a_{i1}$加到第$i$列上。由于$a_{i1} = a_{1i}$，故得到的矩阵的第 $(1,i)$元素及第$(i,1)$元素均等于零。由引理 8.1.1 可知，新得到的矩阵与$A$是合同的。这样，可依次把$A$的第一行与第一列除$a_{11}$外的元素都消去。于是$A$合同于下列矩阵：$$\begin{pmatrix}
a_{11} & 0 & 0 & \cdots & 0 \\
0 & b_{22} & b_{23} & \cdots & b_{2n} \\
0 & b_{32} & b_{33} & \cdots & b_{3n} \\
\vdots & \vdots & \vdots & & \vdots \\
0 & b_{n2} & b_{n3} & \cdots & b_{nn}
\end{pmatrix}$$
多次重复上述过程即可, 由此可以知道必定合同.
##### Example 
Proof: 秩等于$r$的对称阵等于$r$个秩为$1$的对称阵之和
对于这种证明矩阵的某个性质或是某种特征的, 可以思考**这些量是哪种变换的不变量**, 从而简化问题. 以这题为例, 秩是合同变化的不变量, 所以可得
$$\displaylines{rank(A) = r, A^T = A\\ A = C^TBC, B \text{为对角阵, 则}\\ B = \sum_{i = 1}^{r}{P_i}, P_i为秩为1的对角阵 \\ \Rightarrow  A = C^TBC = C^T\sum_{i = 1}^{r}{P_i}C}$$
由此我们可以开始探究惯性定理
#### 惯性定理
##### 实二次型
我们不妨设实对称阵已具有下列对角阵的形状：
$$A = \text{diag}\{d_1, d_2, \cdots, d_r, 0, \cdots, 0\}.$$由引理 8.1.1 不难知道，任意调换$A$的主对角线上的元素得到的矩阵仍与$A$合同。因此我们可把零放在一起，把正项与负项放在一起，即可设$d_1 > 0, \cdots, d_p > 0$；$d_{p + 1} < 0, \cdots, d_r < 0$。$A$所代表的二次型为$$f(x_1, x_2, \cdots, x_n) = d_1 x_1^2 + d_2 x_2^2 + \cdots + d_r x_r^2. \quad (8.3.1)$$令$$y_1 = \sqrt{d_1} x_1, \cdots, y_p = \sqrt{d_p} x_p;$$
$$y_{p + 1} = \sqrt{-d_{p + 1}} x_{p + 1}, \cdots, y_r = \sqrt{-d_r} x_r;$$
$$y_j = x_j \quad (j = r + 1, \cdots, n),$$则 (8.3.1) 式变为$$f = y_1^2 + \cdots + y_p^2 - y_{p + 1}^2 - \cdots - y_r^2. \quad (8.3.2)$$这一事实等价于说$A$合同于下列对角阵：$$\text{diag}\{1, \cdots, 1; -1, \cdots, -1; 0, \cdots, 0\}, \quad (8.3.3)$$其中有$p$个$1$，$q$个$-1$，$n - r$ 个零。
###### Definition 8.3.1
设 $f(x_1, x_2, \cdots, x_n)$是一个$n$元实二次型, 且$f$可化为两个标准型:$$\displaylines{c_1 y_1^2 + \cdots + c_p y_p^2 - c_{p + 1} y_{p + 1}^2 - \cdots - c_r y_r^2 \\ d_1 z_1^2 + \cdots + d_k z_k^2 - d_{k + 1} z_{k + 1}^2 - \cdots - d_r z_r^2}$$
其中$c_i > 0, d_i > 0$, 则必有 $p = k$。
###### Definition 8.3.1
设 $f(x_1, x_2, \cdots, x_n)$是一个实二次型，若它能化为形如$(8.3.2)$式的形状，则称$r$ 是该二次型的秩，$p$ 是它的正惯性指数，$q = r - p$ 是它的负惯性指数，$s = p - q$称为$f$的符号差。

显然，若已知秩$r$与符号差$s$，则 $p = \frac{1}{2}(r + s)$，$q = \frac{1}{2}(r - s)$。事实上，在 $p, q, r, s$ 中只需知道其中两个数，其余两个数也就知道了。由于实对称阵与实二次型之间的等价关系，我们将实二次型的秩、惯性指数及符号差也称为相应的实对称阵的秩、惯性指数及符号差。
##### Theorem: Sylvester Law of Inertia
两个矩阵合同当且仅当惯性系数都一样
显然若矩阵符合一组惯性系数则可以合同化为一组标准型, 另一组也可以与这个标准型合同, 又因为等价关系的传递性, 所以两个矩阵也互相合同.

#### 8.2 二次型的化简
##### 配方法
直接进行配方即可
##### 初等变换法
这种方法可总结如下：作$n\times 2n$矩阵$\begin{pmatrix} \boldsymbol{A}, \boldsymbol{I}_n \end{pmatrix}$，对这个矩阵实施初等行变换，同时施以同样的初等列变换，将它左半边化为对角阵，则这个对角阵就是已化简的二次型的相伴矩阵，右半边的转置便是变换矩阵$\boldsymbol{C}$。

如碰到第$(1,1)$元素是零的矩阵，可先设法将第$(1,1)$元素化成非零，再进行上述过程。
##### 谱定理
![[Linear Algebra#Theorem Spectral Theorem]]
##### Example
将二次型 $f(x_1, x_2, x_3) = 2x_1x_2 + 4x_1x_3 - 4x_2x_3$化成对角型.
Solution 
写出与$f$相伴的对称阵$A$, 作 $(A, I_3)$并将它的第二行加到第一行上, 再将第二列加到第一列上:$$(A, I_3) = 
\begin{pmatrix}
0 & 1 & 2 & \mid & 1 & 0 & 0 \\
1 & 0 & -2 & \mid & 0 & 1 & 0 \\
2 & -2 & 0 & \mid & 0 & 0 & 1
\end{pmatrix}
\rightarrow
\begin{pmatrix}
2 & 1 & 0 & \mid & 1 & 1 & 0 \\
1 & 0 & -2 & \mid & 0 & 1 & 0 \\
0 & -2 & 0 & \mid & 0 & 0 & 1
\end{pmatrix}$$同例 8.2.3 一样, 对上述矩阵进行初等变换得到$$\begin{pmatrix}
2 & 0 & 0 & \mid & 1 & 1 & 0 \\
0 & -\frac{1}{2} & 0 & \mid & -\frac{1}{2} & \frac{1}{2} & 0 \\
0 & 0 & 8 & \mid & 2 & -2 & 1
\end{pmatrix}$$因此$f$化简为$$2y_1^2 - \frac{1}{2}y_2^2 + 8y_3^2$$
$$C = 
\begin{pmatrix}
1 & -\frac{1}{2} & 2 \\
1 & \frac{1}{2} & -2 \\
0 & 0 & 1
\end{pmatrix}$$
##### Theorem: Spectral Theorem
对于任意 **实对称矩阵** $A \in \mathbb{R}^{n \times n}$，存在一个 **正交矩阵** Q，使得：
$$Q^T A Q = \Lambda$$
其中 $\Lambda$ 是由 A 的特征值构成的对角矩阵。
它说明了：
- **特征值全为实数**
- **可以找到单位正交特征向量组**
- **可以用正交变换将矩阵对角化**
因为此时Q为正交矩阵, 所以我们的变换不仅是合同变换, 同时也是相似变换, 所以矩阵A与矩阵$\Lambda$具有相同的特征值.
其中用于变换的矩阵 Q 是由特征向量得到的, 因为:
$$Q = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix},\quad A v_i = \lambda_i v_i$$
那么有：
$$AQ = Q \Lambda \quad \Rightarrow \quad Q^{-1} A Q = \Lambda$$
由于 Q 是正交矩阵，$Q^{-1} = Q^T$，所以：
$$\boxed{Q^T A Q = \Lambda}$$
Q.E.D.
内积空间中还会用到, 比如在自伴随算子等.

#### 对合同变换以及相似变换的比较
- 二者都不会对rank产生影响(合同变换更强, 还有惯性系数)
- 相似变换不影响行列式, 合同变换只有是正交变换的时候才不会影响(从代数上考虑是因为正交矩阵乘积为一了, 从特征值来看正交的时候合同变换也是相似变换, 特征值不变, 乘积不变)
- 相似矩阵特征值不变, 特征向量不变
- 合同变换满足正交变换时即可以通过特征值的正负判断惯性系数


### 内积空间
#### 内积空间的概念
在解析几何中，我们已经知道，$\mathbb{R}^3$中任一向量可定义其“长度”，空间中任意两点之间可定义其距离。向量$\boldsymbol{v} = (x_1, x_2, x_3)$的长度为$$\|\boldsymbol{v}\| = \sqrt{x_1^2 + x_2^2 + x_3^2}.$$两点$(x_1, x_2, x_3)$，$(y_1, y_2, y_3)$之间的距离为$$\sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2},$$即这两点所代表的向量之差的长度。我们现在要把“长度”、“距离”的概念推广到一般的实线性空间与复线性空间上去。距离可看成是长度的派生概念，而长度又可看成是内积的派生概念。我们已经知道在$\mathbb{R}^3$中，内积是这样定义的：若$\boldsymbol{u} = (x_1, x_2, x_3)$，$\boldsymbol{v} = (y_1, y_2, y_3)$，则 $\boldsymbol{u}$与$\boldsymbol{v}$的内积（或点积）为$$\boldsymbol{u} \cdot \boldsymbol{v} = x_1y_1 + x_2y_2 + x_3y_3, \tag{9.1.1}$$从而$\boldsymbol{u}$的长度为$$\|\boldsymbol{u}\| = (\boldsymbol{u} \cdot \boldsymbol{u})^{\frac{1}{2}}.$$从 (9.1.1) 式我们可看出$\mathbb{R}^3$中的内积有下列性质。

$\mathbb{R}^3$中向量内积的性质
1.$\boldsymbol{u} \cdot \boldsymbol{v} = \boldsymbol{v} \cdot \boldsymbol{u}$；
2. $(\boldsymbol{u} + \boldsymbol{w}) \cdot \boldsymbol{v} = \boldsymbol{u} \cdot \boldsymbol{v} + \boldsymbol{w} \cdot \boldsymbol{v}$；
3. $(c\boldsymbol{u}) \cdot \boldsymbol{v} = c(\boldsymbol{u} \cdot \boldsymbol{v})$；
4. $\boldsymbol{u} \cdot \boldsymbol{u} \geq 0$，且 $\boldsymbol{u} \cdot \boldsymbol{u} = 0$当且仅当$\boldsymbol{u} = \boldsymbol{0}$，

其中 $\boldsymbol{u}, \boldsymbol{v}, \boldsymbol{w}$是$\mathbb{R}^3$ 中的任意向量，$c$ 是任一实数。

根据上述 4 条性质，我们类似地定义一般线性空间上的内积。
特点: 证明对称性以及共轭对称性时经常使用数的转置等于数本身来进行证明
非标准内积举例:正定实对称阵积的内积(除去单位阵)
##### Definition: Inner Product on $\mathbb R$
设 $V$是实数域上的线性空间, 若存在某种规则, 使对$V$中任意一组有序向量$\{\alpha, \beta\}$, 都唯一地对应一个实数, 记为 $(\alpha, \beta)$, 且适合如下规则:
1. $(\beta, \alpha) = (\alpha, \beta)$;
2. $(\alpha + \beta, \gamma) = (\alpha, \gamma) + (\beta, \gamma)$;
3. $(c\alpha, \beta) = c(\alpha, \beta)$, $c$为任一实数;
4. $(\alpha, \alpha) \geq 0$且等号成立当且仅当$\alpha = 0$,

则称在 $V$上定义了一个内积. 实数$(\alpha, \beta)$称为$\alpha$与$\beta$的内积. 线性空间$V$称为实内积空间. 有限维实内积空间称为 Euclid 空间, 简称为欧氏空间.
##### Definition: Inner Product on $\mathbb C$
设$V$是复数域上的线性空间, 若存在某种规则, 使对$V$中任意一组有序向量$\{\alpha, \beta\}$, 都唯一地对应一个复数, 记为 $(\alpha, \beta)$, 且适合如下规则:
1. $(\beta, \alpha) = \overline{(\alpha, \beta)}$;
2. $(\alpha + \beta, \gamma) = (\alpha, \gamma) + (\beta, \gamma)$;
3. $(c\alpha, \beta) = c(\alpha, \beta)$, $c$为任一复数;
4.$(\alpha, \alpha) \geq 0$且等号成立当且仅当$\alpha = 0$,

则称在 $V$上定义了一个内积. 复数$(\alpha, \beta)$称为$\alpha$与$\beta$的内积. 线性空间$V$称为复内积空间. 有限维复内积空间称为酉空间.
**ps**:
实内积空间的定义与复内积空间的定义是相容的. 事实上, 对一个实数$a$, $\overline{a} = a$. 故实数域定义中的 (1) 与复数域定义中的 (1) 是一致的. 因此, 我们经常将这两种空间统称为内积空间, 在某些定理的叙述及证明中也不区别它们, 而统一作为复内积空间来处理. 但是, 需要注意的是对复内积空间, 定义复数域中的 (1), (3) 意味着:

$$(\alpha, c\beta) = \overline{c}(\alpha, \beta)$$
即
- 对称性(共轭对称性)
- 第一变元线性
- 正定性

##### Hermitian Matrix
若$A \in \mathbb C^{n \times n}$ 满足$A = A^{\dagger}=\overline{A}^{T}$, 则A被称为Hermite阵.

#### 内积的表示和正交积
我们得到了内积在给定基下的表示:
$$(\alpha, \beta) = x'Gy \tag{9.2.1}$$
从这里可以看出内积与二次型有很强的关联性
再来看矩阵 $G$。因为 $(\boldsymbol{v}_i, \boldsymbol{v}_j) = (\boldsymbol{v}_j, \boldsymbol{v}_i)$，所以 $G$是实对称阵。又因为对任意的非零向量$\boldsymbol{\alpha}$，总有 $(\boldsymbol{\alpha}, \boldsymbol{\alpha}) > 0$，所以 $\boldsymbol{x}'G\boldsymbol{x} > 0$对一切$n$维非零实列向量$\boldsymbol{x}$成立。这表明$G$是一个正定阵。反之，给定$n$阶正定实对称阵，利用$(9.2.1)$式，我们也不难定义$V$上的内积（参见例 9.1.5）。由此我们可以看出，若给定了$n$维欧氏空间的一组基，则欧氏空间上的内积和$n$ 阶正定实对称阵之间存在着一个一一对应。

n维内积空间上的内积一定可以表示为Gram阵的形式. 反之亦然. 反之的情况其实很好证明, 因为之前证明过正交矩阵构成的内积形式确实是内积, 也就是说我们只需要满足$g_{ij} = (e_i, e_j)$即可确定这是Gram阵. 按照正交积内积的定义计算(将向量写成基的形式)即可自然得到.(此时相当于时通过这个正交积定义了$(e_i,e_j)$.

事实上这也就是说明n维线性空间上的内积结构和n阶正定实对称阵之间有一个一一对应的关系(Euclid空间上):
$$\displaylines{内积结构\Rightarrow G_{n \times n} = ((e_i, e_j))_{n \times n} \\
(u, v)_G = u^TGv \quad \Leftarrow \quad G_{n \times n}}$$
酉空间上:
$$\displaylines{G^T = \overline G\\
since \quad\overline {g_{ij}} = \overline{(e_i, e_j)} = (e_j, e_i) = g_{ji}\\
namely \quad \overline {G^T} = G}$$
G一定为Hermite阵.
##### Definition
设$\{e_1,e_2,\cdots,e_n\}$是$n$维内积空间$V$的一组基。若$e_i\perp e_j$对一切$i\neq j$成立，则称这组基是$V$的一组正交基。又若$V$的一组正交基中每个向量的长度等于$1$，则称这组正交基为标准正交基。
##### Lemma
设 $V$ 是内积空间，$\boldsymbol{u}_1, \boldsymbol{u}_2, \cdots, \boldsymbol{u}_m$是$V$中$m$个线性无关的向量，则在$V$中存在$m$个两两正交的非零向量$\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_m$，使 $\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_m$张成的$V$的子空间恰好为由$\boldsymbol{u}_1, \boldsymbol{u}_2, \cdots, \boldsymbol{u}_m$张成的$V$的子空间，即$\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_m$ 是该子空间的一组正交基. 
##### Theorem: Gram-Schmidt Method
设$\boldsymbol{v}_{1}=\boldsymbol{u}_{1}$，其余$\boldsymbol{v}_{i}$可用数学归纳法定义如下：假定$\boldsymbol{v}_{k}$已定义好$(1\leq k < m)$，这时$\boldsymbol{v}_{i}(1\leq i\leq k)$两两正交非零且$\boldsymbol{v}_{i}(1\leq i\leq k)$皆属于由$\boldsymbol{u}_{1},\boldsymbol{u}_{2},\cdots,\boldsymbol{u}_{k}$张成的子空间。令
$$\boldsymbol{v}_{k + 1}=\boldsymbol{u}_{k + 1}-\sum_{j = 1}^{k}\frac{(\boldsymbol{u}_{k+1},\boldsymbol{v}_{j})}{\|\boldsymbol{v}_{j}\|^{2}}\boldsymbol{v}_{j}$$
##### Inference 9.2.2
任一有限维内积空间均有标准正交基。

下面我们要讨论子空间的正交问题.

##### Definition 9.2.2
设 $U$是内积空间$V$的子空间. 令$$U^{\perp} = \{ v \in V \mid (v, U) = 0 \},$$这里$(v, U) = 0$表示对一切$u \in U$, 均有 $(v, u) = 0$. 容易验证 $U^{\perp}$是$V$的子空间, 称为$U$的正交补空间.
##### Theorem 9.2.2
设$V$是$n$维内积空间,$U$是$V$的子空间, 则
(1)$V = U \oplus U^{\perp}$;
(2) $U$上的任一组标准正交基均可扩张为$V$上的标准正交基.
**Proof**
(1) 若$x \in U \cap U^{\perp}$, 则 $(x, x) = 0$, 因此 $x = 0$, 即 $U \cap U^{\perp} = 0$. 另一方面, 由推论 9.2.2 可知, 存在 $U$的一组标准正交基$\{ e_1, e_2, \cdots, e_m \}$. 对任意的 $v \in V$, 令
$$u = (v, e_1)e_1 + (v, e_2)e_2 + \cdots + (v, e_m)e_m,$$则$u \in U$. 又令
$$w = v - u,$$则对任一$e_i (i = 1, 2, \cdots, m)$, 有
$$(w, e_i) = (v, e_i) - (u, e_i) = (v, e_i) - (v, e_i) = 0.$$
因此 $w \in U^{\perp}$，而 $v = u + w$，这就证明了 $V = U \oplus U^{\perp}$。

(2) 设 $\{e_1, e_2, \cdots, e_m\}$是$U$ 上任一组标准正交基，$\{e_{m + 1}, \cdots, e_n\}$是$U^{\perp}$上任一组标准正交基，则显然$\{e_1, e_2, \cdots, e_n\}$是$V$的一组标准正交基。$\square$
##### Definition 9.2.3 
设$V$是$n$ 维内积空间，$V_i (i = 1, 2, \cdots, k)$是$V$的子空间。如果对任意的$\alpha \in V_i$和任意的$\beta \in V_j (j \neq i)$均有$(\alpha, \beta) = 0$，则称子空间 $V_i$和$V_j$正交。若$V = V_1 + V_2 + \cdots + V_k$且$V_i$两两正交，则称$V$是$V_i (i = 1, 2, \cdots, k)$的正交和，记为$$V = V_1 \perp V_2 \perp \cdots \perp V_k$$
##### Lemma 9.2.3 
正交和必为直和且任一$V_i$和其余子空间的和正交。
**Proof**
对任意的$v_i \in V_i$和$\sum_{j \neq i} v_j (v_j \in V_j)$，
$$(v_i, \sum_{j \neq i} v_j) = \sum_{j \neq i} (v_i, v_j) = 0$$因此后一个结论成立。任取$v \in V_i \cap (\sum_{j \neq i} V_j)$，则由上述论证可得 $(v, v) = 0$，故 $v = 0$，从而 $V_i \cap (\sum_{j \neq i} V_j) = 0$，即正交和必为直和。因此正交和通常也称为正交直和。 $\square$
##### Definition 9.2.4
设 $V = V_1 \perp V_2 \perp \cdots \perp V_k$，定义 $V$上的线性变换$E_i (i = 1, 2, \cdots, k)$如下：若$v = v_1 + \cdots + v_i + \cdots + v_k (v_i \in V_i)$，令 $E_i (v) = v_i$。容易验证 $E_i$是$V$上的线性变换且$E_i^2 = E_i$，$E_i E_j = 0 (i \neq j)$，$E_1 + E_2 + \cdots + E_k = I_V$。线性变换 $E_i$称为$V$到$V_i$的正交投影（简称投影）。

##### 命题 9.2.1
设$U$是内积空间$V$ 的子空间，$V = U \perp U^\perp$。$E$是$V$到$U$的正交投影，则对任意的$\alpha, \beta \in V$，都有
$$(E(\alpha), \beta) = (\alpha, E(\beta))$$**Proof**：
设$\alpha = u_1 + w_1$，$\beta = u_2 + w_2$，其中 $u_1, u_2 \in U$，$w_1, w_2 \in U^\perp$，则 $E(\alpha) = u_1$，$E(\beta) = u_2$，所以
$$(E(\alpha), \beta) = (u_1, u_2 + w_2) = (u_1, u_2) + (u_1, w_2) = (u_1, u_2)$$
$$(\alpha, E(\beta)) = (u_1 + w_1, u_2) = (u_1, u_2) + (w_1, u_2) = (u_1, u_2)$$
由此即得结论。 □

下面的结论是“斜边大于直角边”这一几何命题在内积空间中的推广。
##### 命题 9.2.2 Bessel不等式
设 $v_1, v_2, \cdots, v_m$是内积空间$V$中的正交非零向量组,$y$是$V$中任一向量, 则$$\sum_{k = 1}^{m} \frac{|(y, v_k)|^2}{\|v_k\|^2} \leq \|y\|^2$$等号成立的充分必要条件是:$y$属于由$\{v_1, v_2, \cdots, v_m\}$张成的子空间.

**Proof**
令$$x = \sum_{k = 1}^{m} \frac{(y, v_k)}{\|v_k\|^2} v_k$$则$x$属于由$\{v_1, v_2, \cdots, v_m\}$张成的子空间$U$. 容易验证
$$(y - x, v_k) = 0$$对一切$k = 1, 2, \cdots, m$成立, 因此$(y - x, x) = 0$. 由勾股定理得:
$$\|y\|^2 = \|y - x\|^2 + \|x\|^2$$故$$\|x\|^2 \leq \|y\|^2$$又由$v_1, v_2, \cdots, v_m$两两正交不难算出$$\|x\|^2 = \sum_{k = 1}^{m} \frac{|(y, v_k)|^2}{\|v_k\|^2}$$若$y$属于由$\{v_1, v_2, \cdots, v_m\}$张成的子空间, 则$y = x$, 故等号成立. 反之, 若等号成立, 则 $\|y - x\|^2 = 0$, 故 $y = x$, 即 $y$属于由$\{v_1, v_2, \cdots, v_m\}$ 张成的子空间. □


#### 内积空间的同构

##### 保积同构
###### Definition
如果对于两个空间U, V有线性映射(更严格情况下为同构)$\Phi(*): U \to V$,且存在性质$(\alpha, \beta)_U = (\Phi(\alpha), \Phi(\beta))_V$ , 则称这个线性映射为保积映射.
可知保积映射一定单射:
$$\displaylines{for \,\,\forall a \in Ker(\Phi) \\
\Rightarrow  (a, a) = (\Phi(a), \Phi(a)) = 0 \\
\Rightarrow a = 0  \\
\Rightarrow 单射}$$
但是不一定满射(e.g.$\mathbb R^2 \to \mathbb R^3$的嵌入映射显然也满足保积性, 但是不满足满射
结合Chapter 4的内容, 前后两个线性空间的维数相同时, $\Phi$是单射 i.i.f.它是满射i.i.f.它是线性同构.
##### remark
保积$\Leftrightarrow$保范$\Leftrightarrow$保距
##### Theorem 1
##### Theorem 2
##### Theorem 3
若有线性映射$\Phi : \mathbb V \to \mathbb U$, 两空间均为实或复内积空间且维度相等, 则下面的陈述相互间等价:
- $\Phi$是保持内积的
- $\Phi$是保积同构
- $\Phi$将任意标准正交基映射为标准正交基
- $\Phi$将某一组标准正交基映射为标准正交基
##### Inference
若有两个有限维内积空间$\mathbb V , \mathbb U$, 同构 $\Leftrightarrow$ 两空间维度相等
正交算子几何的特殊意义就是保持内积的基底变换