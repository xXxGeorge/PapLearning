---
tags:
  - BasicMath
  - Lemma
relation:
  - "[[NumericalCalculate]]"
teacher: Manus
---


## 1. 数值积分概述

### 1.1 数值积分的基本问题

数值积分（Numerical Integration）是计算定积分近似值的方法，当被积函数没有解析原函数或原函数难以求得时尤为重要。数值积分的基本问题是计算以下形式的定积分：

$$I = \int_a^b f(x) dx$$

其中 $f(x)$ 是在区间 $[a, b]$ 上的函数。

### 1.2 数值积分的应用

数值积分在科学和工程领域有广泛的应用：

1. **物理学**：计算能量、功、路径积分等
2. **工程学**：结构分析、热传导、流体力学等
3. **概率统计**：计算概率分布的期望、方差等
4. **金融数学**：期权定价、风险评估等
5. **计算机图形学**：渲染方程、光线追踪等

### 1.3 数值积分方法分类

数值积分方法可以大致分为以下几类：

1. **Newton-Cotes公式**：基于等距节点的插值多项式的积分
2. **高斯求积公式**：基于正交多项式的零点选取节点
3. **自适应方法**：根据函数行为动态调整积分步长
4. **蒙特卡洛方法**：基于随机采样的积分方法
5. **Romberg积分**：利用Richardson外推提高精度

本文将重点讲解Newtonasd-Cotes求积法，这是最基础也是最常用的数值积分方法之一。

## 2. Newton-Cotes求积法的基本原理

### 2.1 基本思想

Newton-Cotes求积法的基本思想是：

1. 将被积函数 $f(x)$ 在积分区间 $[a, b]$ 上用插值多项式 $P_n(x)$ 近似
2. 计算插值多项式的积分作为原函数积分的近似值

即：

$$\int_a^b f(x) dx \approx \int_a^b P_n(x) dx$$

其中 $P_n(x)$ 是在区间 $[a, b]$ 上的 $n$ 次插值多项式，通过 $n+1$ 个点确定。


### 2.2 插值节点选择

Newton-Cotes公式使用等距节点进行插值，即：

$$x_i = a + i \cdot h, \quad i = 0, 1, 2, \ldots, n$$

其中 $h = \frac{b-a}{n}$ 是步长。

### 2.3 积分公式的一般形式

Newton-Cotes积分公式可以表示为：

$$\int_a^b f(x) dx \approx (b-a) \sum_{i=0}^{n} A_i f(x_i)$$

其中 $A_i$ 是权系数，取决于积分公式的阶数和类型。

#### Cotes系数表(至n = 8)

| n   | 节点数 | Cotes 系数（归一化，总和为 1）                                                                                         |
| --- | --- | ----------------------------------------------------------------------------------------------------------- |
| 1   | 2   | 1/2, 1/2                                                                                                    |
| 2   | 3   | 1/6, 4/6, 1/6                                                                                               |
| 3   | 4   | 1/8, 3/8, 3/8, 1/8                                                                                          |
| 4   | 5   | 7/90, 32/90, 12/90, 32/90, 7/90                                                                             |
| 5   | 6   | 19/288, 75/288, 50/288, 50/288, 75/288, 19/288                                                              |
| 6   | 7   | 41/840, 216/840, 27/840, 272/840, 27/840, 216/840, 41/840                                                   |
| 7   | 8   | 751/17280, 3577/17280, 1323/17280, 2989/17280, 2989/17280, 1323/17280, 3577/17280, 751/17280                |
| 8   | 9   | 989/28350, 5888/28350, -928/28350, 10496/28350, -4540/28350, 10496/28350, -928/28350, 5888/28350, 989/28350 |

## 3. 闭型Newton-Cotes公式推导

闭型Newton-Cotes公式使用包含区间端点的节点集，即 $x_0 = a$ 和 $x_n = b$。

### 3.1 推导方法

我们可以通过以下步骤推导Newton-Cotes公式：

1. 在等距节点 $x_0, x_1, \ldots, x_n$ 上构造Lagrange插值多项式 $P_n(x)$
2. 计算 $\int_a^b P_n(x) dx$
3. 整理得到权系数 $A_i$

Lagrange插值多项式可以表示为：

$$P_n(x) = \sum_{i=0}^{n} f(x_i) L_i(x)$$

其中 $L_i(x)$ 是Lagrange基本多项式：

$$L_i(x) = \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j}$$

将 $P_n(x)$ 代入积分：

$$\int_a^b P_n(x) dx = \int_a^b \sum_{i=0}^{n} f(x_i) L_i(x) dx = \sum_{i=0}^{n} f(x_i) \int_a^b L_i(x) dx$$

因此权系数 $A_i$ 可以表示为：

$$A_i = \frac{1}{b-a} \int_a^b L_i(x) dx$$

### 3.2 常用闭型Newton-Cotes公式

#### 3.2.1 梯形法则 (n=1)

当 $n=1$ 时，我们有两个节点 $x_0 = a$ 和 $x_1 = b$。

Lagrange基本多项式为：

$$L_0(x) = \frac{x - x_1}{x_0 - x_1} = \frac{x - b}{a - b} = \frac{b - x}{b - a}$$

$$L_1(x) = \frac{x - x_0}{x_1 - x_0} = \frac{x - a}{b - a}$$

计算权系数：

$$A_0 = \frac{1}{b-a} \int_a^b \frac{b - x}{b - a} dx = \frac{1}{b-a} \cdot \frac{1}{b-a} \int_a^b (b - x) dx = \frac{1}{2}$$

$$A_1 = \frac{1}{b-a} \int_a^b \frac{x - a}{b - a} dx = \frac{1}{b-a} \cdot \frac{1}{b-a} \int_a^b (x - a) dx = \frac{1}{2}$$

因此，梯形法则为：

$$\int_a^b f(x) dx \approx \frac{b-a}{2} [f(a) + f(b)]$$

这相当于用一个梯形的面积近似曲线下的面积。

#### 3.2.2 Simpson法则 (n=2)

当 $n=2$ 时，我们有三个节点 $x_0 = a$, $x_1 = a + \frac{b-a}{2}$, 和 $x_2 = b$。

通过类似的计算，可以得到权系数 $A_0 = \frac{1}{6}$, $A_1 = \frac{4}{6}$, $A_2 = \frac{1}{6}$。

因此，Simpson法则为：

$$\int_a^b f(x) dx \approx \frac{b-a}{6} [f(a) + 4f(\frac{a+b}{2}) + f(b)]$$

#### 3.2.3 Simpson 3/8法则 (n=3)

当 $n=3$ 时，我们有四个节点 $x_0 = a$, $x_1 = a + \frac{b-a}{3}$, $x_2 = a + \frac{2(b-a)}{3}$, 和 $x_3 = b$。

权系数为 $A_0 = \frac{1}{8}$, $A_1 = \frac{3}{8}$, $A_2 = \frac{3}{8}$, $A_3 = \frac{1}{8}$。

因此，Simpson 3/8法则为：

$$\int_a^b f(x) dx \approx \frac{b-a}{8} [f(a) + 3f(a+\frac{b-a}{3}) + 3f(a+\frac{2(b-a)}{3}) + f(b)]$$

#### 3.2.4 Boole法则 (n=4)

当 $n=4$ 时，权系数为 $A_0 = \frac{7}{90}$, $A_1 = \frac{32}{90}$, $A_2 = \frac{12}{90}$, $A_3 = \frac{32}{90}$, $A_4 = \frac{7}{90}$。

Boole法则为：

$$\int_a^b f(x) dx \approx \frac{b-a}{90} [7f(a) + 32f(a+\frac{b-a}{4}) + 12f(a+\frac{2(b-a)}{4}) + 32f(a+\frac{3(b-a)}{4}) + 7f(b)]$$

### 3.3 闭型Newton-Cotes公式的代数精度

Newton-Cotes公式的代数精度是指公式能够精确积分的最高次多项式的次数。

- 梯形法则 (n=1): 代数精度为1，能精确积分线性函数
- Simpson法则 (n=2): 代数精度为3，能精确积分3次多项式
- Simpson 3/8法则 (n=3): 代数精度为3，能精确积分3次多项式
- Boole法则 (n=4): 代数精度为5，能精确积分5次多项式

一般来说，当 $n$ 为偶数时，闭型Newton-Cotes公式的代数精度为 $n+1$；当 $n$ 为奇数时，代数精度为 $n$。

## 4. 开型Newton-Cotes公式

开型Newton-Cotes公式不使用区间端点，即节点 $x_i$ 满足 $a < x_0 < x_1 < \ldots < x_n < b$。

### 4.1 开型公式的节点选择

开型公式的节点通常选为：

$$x_i = a + (i+1) \cdot h, \quad i = 0, 1, 2, \ldots, n$$

其中 $h = \frac{b-a}{n+2}$ 是步长。

### 4.2 常用开型Newton-Cotes公式

#### 4.2.1 中点法则 (n=0)

只使用区间中点的值：

$$\int_a^b f(x) dx \approx (b-a) f(\frac{a+b}{2})$$

#### 4.2.2 开型梯形法则 (n=1)

$$\int_a^b f(x) dx \approx \frac{b-a}{2} [f(a+\frac{b-a}{3}) + f(a+\frac{2(b-a)}{3})]$$

#### 4.2.3 开型Simpson法则 (n=2)

$$\int_a^b f(x) dx \approx \frac{b-a}{3} [2f(a+\frac{b-a}{4}) - f(a+\frac{2(b-a)}{4}) + 2f(a+\frac{3(b-a)}{4})]$$

## 5. 复化Newton-Cotes公式

对于较宽的积分区间，直接应用高阶Newton-Cotes公式可能不够精确，因为高次插值多项式可能出现Runge现象。解决方法是将积分区间分成多个小区间，在每个小区间上应用低阶Newton-Cotes公式，然后将结果相加。这就是复化Newton-Cotes公式。

### 5.1 复化梯形法则

将区间 $[a, b]$ 等分为 $n$ 个子区间，每个子区间上应用梯形法则：

$$\int_a^b f(x) dx \approx \frac{h}{2} [f(a) + 2\sum_{i=1}^{n-1} f(a+ih) + f(b)]$$

其中 $h = \frac{b-a}{n}$ 是步长。

### 5.2 复化Simpson法则

将区间 $[a, b]$ 等分为 $2n$ 个子区间（确保偶数个子区间），每两个相邻子区间上应用Simpson法则：

$$\int_a^b f(x) dx \approx \frac{h}{3} [f(a) + 4\sum_{i=1,3,5,...}^{2n-1} f(a+ih) + 2\sum_{i=2,4,6,...}^{2n-2} f(a+ih) + f(b)]$$

其中 $h = \frac{b-a}{2n}$ 是步长。

### 5.3 复化公式的误差

- 复化梯形法则的误差为 $O(h^2)$
- 复化Simpson法则的误差为 $O(h^4)$

这意味着当步长减半时，复化梯形法则的误差大约减少为原来的1/4，而复化Simpson法则的误差大约减少为原来的1/16。

## 6. Python实现

下面我们使用Python实现各种Newton-Cotes求积法：

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def trapezoidal_rule(f, a, b, n=1):
    """
    使用梯形法则计算积分
    
    参数:
    f: 被积函数
    a, b: 积分区间
    n: 子区间数量
    
    返回:
    积分近似值
    """
    if n <= 0:
        raise ValueError("子区间数量必须为正整数")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    # 应用梯形法则
    result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    return result

def simpson_rule(f, a, b, n=2):
    """
    使用Simpson法则计算积分
    
    参数:
    f: 被积函数
    a, b: 积分区间
    n: 子区间数量，必须为偶数
    
    返回:
    积分近似值
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError("子区间数量必须为正偶数")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    # 应用Simpson法则
    result = h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    
    return result

def simpson38_rule(f, a, b, n=3):
    """
    使用Simpson 3/8法则计算积分
    
    参数:
    f: 被积函数
    a, b: 积分区间
    n: 子区间数量，必须是3的倍数
    
    返回:
    积分近似值
    """
    if n <= 0 or n % 3 != 0:
        raise ValueError("子区间数量必须为3的正整数倍")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    # 应用Simpson 3/8法则
    result = 0
    for i in range(0, n, 3):
        result += 3*h/8 * (y[i] + 3*y[i+1] + 3*y[i+2] + y[i+3])
    
    return result

def boole_rule(f, a, b, n=4):
    """
    使用Boole法则计算积分
    
    参数:
    f: 被积函数
    a, b: 积分区间
    n: 子区间数量，必须是4的倍数
    
    返回:
    积分近似值
    """
    if n <= 0 or n % 4 != 0:
        raise ValueError("子区间数量必须为4的正整数倍")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    # 应用Boole法则
    result = 0
    for i in range(0, n, 4):
        result += 2*h/45 * (7*y[i] + 32*y[i+1] + 12*y[i+2] + 32*y[i+3] + 7*y[i+4])
    
    return result

def midpoint_rule(f, a, b, n=1):
    """
    使用中点法则计算积分
    
    参数:
    f: 被积函数
    a, b: 积分区间
    n: 子区间数量
    
    返回:
    积分近似值
    """
    if n <= 0:
        raise ValueError("子区间数量必须为正整数")
    
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    y = f(x)
    
    # 应用中点法则
    result = h * np.sum(y)
    
    return result

def compare_methods(f, a, b, exact, title=""):
    """
    比较不同Newton-Cotes方法的精度
    
    参数:
    f: 被积函数
    a, b: 积分区间
    exact: 精确积分值
    title: 图表标题
    """
    # 测试不同的子区间数量
    n_values = np.array([4, 8, 16, 32, 64, 128])
    
    # 计算不同方法的误差
    trap_errors = []
    simp_errors = []
    simp38_errors = []
    boole_errors = []
    midp_errors = []
    
    for n in n_values:
        trap_errors.append(abs(trapezoidal_rule(f, a, b, n) - exact))
        simp_errors.append(abs(simpson_rule(f, a, b, n) - exact))
        simp38_errors.append(abs(simpson38_rule(f, a, b, n) - exact))
        boole_errors.append(abs(boole_rule(f, a, b, n) - exact))
        midp_errors.append(abs(midpoint_rule(f, a, b, n) - exact))
    
    # 绘制误差收敛图
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, trap_errors, 'o-', label='梯形法则')
    plt.loglog(n_values, simp_errors, 's-', label='Simpson法则')
    plt.loglog(n_values, simp38_errors, '^-', label='Simpson 3/8法则')
    plt.loglog(n_values, boole_errors, 'd-', label='Boole法则')
    plt.loglog(n_values, midp_errors, 'x-', label='中点法则')
    
    # 添加参考线
    plt.loglog(n_values, 1/n_values**2, 'k--', label='O(h²)')
    plt.loglog(n_values, 1/n_values**4, 'k-.', label='O(h⁴)')
    
    plt.grid(True)
    plt.xlabel('子区间数量')
    plt.ylabel('绝对误差')
    plt.title(f'Newton-Cotes方法误差比较 - {title}')
    plt.legend()
    plt.savefig('newton_cotes_error_comparison.png')
    plt.close()

def visualize_integration(f, a, b, n, title=""):
    """
    可视化不同Newton-Cotes方法的积分过程
    
    参数:
    f: 被积函数
    a, b: 积分区间
    n: 子区间数量
    title: 图表标题
    """
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)
    
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 梯形法则
    axs[0, 0].plot(x_fine, y_fine, 'b-', label='f(x)')
    for i in range(n):
        xs = [x[i], x[i], x[i+1], x[i+1]]
        ys = [0, y[i], y[i+1], 0]
        axs[0, 0].fill(xs, ys, 'r', alpha=0.3)
    axs[0, 0].plot(x, y, 'ro')
    axs[0, 0].grid(True)
    axs[0, 0].set_title('梯形法则')
    
    # Simpson法则
    if n % 2 == 0:
        axs[0, 1].plot(x_fine, y_fine, 'b-', label='f(x)')
        for i in range(0, n, 2):
            # 绘制二次多项式
            x_local = np.linspace(x[i], x[i+2], 100)
            # 使用拉格朗日插值
            y_local = (y[i] * (x_local - x[i+1]) * (x_local - x[i+2]) / ((x[i] - x[i+1]) * (x[i] - x[i+2])) + 
                       y[i+1] * (x_local - x[i]) * (x_local - x[i+2]) / ((x[i+1] - x[i]) * (x[i+1] - x[i+2])) + 
                       y[i+2] * (x_local - x[i]) * (x_local - x[i+1]) / ((x[i+2] - x[i]) * (x[i+2] - x[i+1])))
            axs[0, 1].fill_between(x_local, 0, y_local, alpha=0.3, color='g')
        axs[0, 1].plot(x, y, 'go')
        axs[0, 1].grid(True)
        axs[0, 1].set_title('Simpson法则')
    
    # 中点法则
    x_mid = np.linspace(a + (b-a)/(2*n), b - (b-a)/(2*n), n)
    y_mid = f(x_mid)
    axs[1, 0].plot(x_fine, y_fine, 'b-', label='f(x)')
    for i in range(n):
        xs = [a + i*(b-a)/n, a + i*(b-a)/n, a + (i+1)*(b-a)/n, a + (i+1)*(b-a)/n]
        ys = [0, y_mid[i], y_mid[i], 0]
        axs[1, 0].fill(xs, ys, 'y', alpha=0.3)
    axs[1, 0].plot(x_mid, y_mid, 'yo')
    axs[1, 0].grid(True)
    axs[1, 0].set_title('中点法则')
    
    # Boole法则
    if n % 4 == 0:
        axs[1, 1].plot(x_fine, y_fine, 'b-', label='f(x)')
        for i in range(0, n, 4):
            # 绘制四次多项式
            x_local = np.linspace(x[i], x[i+4], 100)
            # 这里简化处理，实际上应该使用四次拉格朗日插值
            # 但为了可视化目的，我们使用样条插值
            from scipy.interpolate import make_interp_spline
            spl = make_interp_spline(x[i:i+5], y[i:i+5], k=4)
            y_local = spl(x_local)
            axs[1, 1].fill_between(x_local, 0, y_local, alpha=0.3, color='m')
        axs[1, 1].plot(x, y, 'mo')
        axs[1, 1].grid(True)
        axs[1, 1].set_title('Boole法则')
    
    plt.tight_layout()
    plt.savefig('newton_cotes_visualization.png')
    plt.close()

# 示例：计算积分 ∫₀¹ x² dx = 1/3
def f1(x):
    return x**2

# 示例：计算积分 ∫₀π sin(x) dx = 2
def f2(x):
    return np.sin(x)

# 示例：计算积分 ∫₁⁴ 1/x dx = ln(4)
def f3(x):
    return 1/x

def run_examples():
    # 示例1
    a1, b1 = 0, 1
    exact1 = 1/3
    print(f"示例1: ∫₀¹ x² dx")
    print(f"精确值: {exact1}")
    print(f"梯形法则 (n=10): {trapezoidal_rule(f1, a1, b1, 10)}")
    print(f"Simpson法则 (n=10): {simpson_rule(f1, a1, b1, 10)}")
    print(f"Simpson 3/8法则 (n=9): {simpson38_rule(f1, a1, b1, 9)}")
    print(f"Boole法则 (n=8): {boole_rule(f1, a1, b1, 8)}")
    print(f"中点法则 (n=10): {midpoint_rule(f1, a1, b1, 10)}")
    print()
    
    # 可视化
    visualize_integration(f1, a1, b1, 10, "f(x) = x²")
    
    # 比较方法
    compare_methods(f1, a1, b1, exact1, "f(x) = x²")
    
    # 示例2
    a2, b2 = 0, np.pi
    exact2 = 2
    print(f"示例2: ∫₀π sin(x) dx")
    print(f"精确值: {exact2}")
    print(f"梯形法则 (n=10): {trapezoidal_rule(f2, a2, b2, 10)}")
    print(f"Simpson法则 (n=10): {simpson_rule(f2, a2, b2, 10)}")
    print(f"Simpson 3/8法则 (n=9): {simpson38_rule(f2, a2, b2, 9)}")
    print(f"Boole法则 (n=8): {boole_rule(f2, a2, b2, 8)}")
    print(f"中点法则 (n=10): {midpoint_rule(f2, a2, b2, 10)}")
    print()
    
    # 示例3
    a3, b3 = 1, 4
    exact3 = np.log(4)
    print(f"示例3: ∫₁⁴ 1/x dx")
    print(f"精确值: {exact3}")
    print(f"梯形法则 (n=10): {trapezoidal_rule(f3, a3, b3, 10)}")
    print(f"Simpson法则 (n=10): {simpson_rule(f3, a3, b3, 10)}")
    print(f"Simpson 3/8法则 (n=9): {simpson38_rule(f3, a3, b3, 9)}")
    print(f"Boole法则 (n=8): {boole_rule(f3, a3, b3, 8)}")
    print(f"中点法则 (n=10): {midpoint_rule(f3, a3, b3, 10)}")

if __name__ == "__main__":
    run_examples()
```

## 7. Newton-Cotes公式的误差分析

### 7.1 截断误差

Newton-Cotes公式的截断误差主要来自于用多项式近似被积函数的误差。对于闭型公式，误差可以表示为：

- 梯形法则 (n=1): $E = -\frac{(b-a)^3}{12} f''(\xi)$，其中 $\xi \in [a, b]$
- Simpson法则 (n=2): $E = -\frac{(b-a)^5}{2880} f^{(4)}(\xi)$，其中 $\xi \in [a, b]$
- Simpson 3/8法则 (n=3): $E = -\frac{(b-a)^5}{6480} f^{(4)}(\xi)$，其中 $\xi \in [a, b]$
- Boole法则 (n=4): $E = -\frac{(b-a)^7}{1935360} f^{(6)}(\xi)$，其中 $\xi \in [a, b]$

对于复合公式，误差为：

- 复合梯形法则: $E = -\frac{b-a}{12} h^2 f''(\xi)$，其中 $\xi \in [a, b]$
- 复合Simpson法则: $E = -\frac{b-a}{180} h^4 f^{(4)}(\xi)$，其中 $\xi \in [a, b]$

### 7.2 舍入误差

除了截断误差外，还需要考虑计算过程中的舍入误差。当使用非常小的步长时，舍入误差可能变得显著，导致总误差增加。这就是为什么在实际应用中，我们需要在截断误差和舍入误差之间找到平衡。

### 7.3 稳定性分析

Newton-Cotes公式的稳定性与其权系数有关。对于高阶公式（n > 8），某些权系数可能变为负值，这可能导致数值不稳定。因此，在实际应用中，通常使用低阶公式（如梯形法则或Simpson法则）结合复合策略，而不是直接使用高阶公式。

## 8. 实际应用中的考虑

### 8.1 选择合适的公式

在实际应用中，选择合适的Newton-Cotes公式需要考虑以下因素：

1. **精度要求**：如果需要高精度，可以考虑高阶公式或复合公式
2. **函数特性**：如果函数有奇异点或高阶导数变化剧烈，可能需要自适应方法
3. **计算效率**：高阶公式需要更多的函数评估，但可能提供更高的精度
4. **稳定性**：低阶复合公式通常比高阶公式更稳定

### 8.2 自适应策略

为了提高效率，可以使用自适应策略，根据函数在不同区域的行为动态调整步长。例如，在函数变化剧烈的区域使用更小的步长，在函数平滑的区域使用更大的步长。

### 8.3 与其他方法的比较

Newton-Cotes公式是最基本的数值积分方法，但在某些情况下，其他方法可能更合适：

1. **高斯求积法**：对于光滑函数，高斯求积法通常比Newton-Cotes公式更高效
2. **Romberg积分**：结合Richardson外推可以显著提高精度
3. **蒙特卡洛方法**：对于高维积分问题，蒙特卡洛方法可能是唯一可行的选择

## 9. 总结

Newton-Cotes求积法是数值积分的基础方法，通过在等距节点上构造插值多项式来近似定积分。主要特点包括：

1. **简单直观**：基于插值多项式的积分，概念清晰
2. **多种公式**：从简单的梯形法则到高阶的Boole法则，适用于不同精度需求
3. **复合策略**：通过将区间分成小区间，可以提高精度和稳定性
4. **误差控制**：不同公式有不同的误差阶，可以根据需要选择

在实际应用中，Newton-Cotes公式（特别是复合梯形法则和复合Simpson法则）仍然是最常用的数值积分方法之一，因为它们简单、稳定且易于实现。

通过本文的理论推导和Python实现，我们可以看到Newton-Cotes求积法的优雅和实用性。在后续的数值计算方法中，我们将看到更多基于或改进自Newton-Cotes公式的方法，如Romberg积分、自适应求积等。
