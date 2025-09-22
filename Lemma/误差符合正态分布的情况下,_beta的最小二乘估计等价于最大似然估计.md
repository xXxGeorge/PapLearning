---
epoch: 1747016362472
modelKey: deepseek-chat|deepseek
tags:
  - copilot-conversation
---

**user**: 如何证明误差符合正态分布的情况下, beta的最小二乘估计等价于最大似然估计
[Timestamp: 2025/05/12 10:19:22]

**ai**: 在误差项 $\epsilon$ 符合正态分布的假设下，可以证明线性回归模型的最小二乘估计（OLS）与最大似然估计（MLE）等价。以下是证明步骤：

---

### 1. **模型设定**
假设线性回归模型为：
$$ y = X\beta + \epsilon $$
其中：
- $y$ 是 $n \times 1$ 的观测向量，
- $X$ 是 $n \times p$ 的设计矩阵（已知），
- $\beta$ 是 $p \times 1$ 的待估参数向量，
- $\epsilon$ 是 $n \times 1$ 的误差向量，且 $\epsilon \sim N(0, \sigma^2 I)$（独立同分布的正态误差）。

---

### 2. **最小二乘估计（OLS）**
OLS 的目标是最小化残差平方和：
$$ \hat{\beta}_{\text{OLS}} = \arg\min_{\beta} \|y - X\beta\|^2 $$
解为：
$$ \hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T y $$

---

### 3. **最大似然估计（MLE）**
在正态误差假设下，$y$ 的条件分布为：
$$ y \sim N(X\beta, \sigma^2 I) $$
似然函数为：
$$ L(\beta, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - X_i \beta)^2}{2\sigma^2}\right) $$
对数似然函数为：
$$ \ell(\beta, \sigma^2) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \|y - X\beta\|^2 $$

最大化 $\ell(\beta, \sigma^2)$ 等价于最小化 $\|y - X\beta\|^2$，因此：
$$ \hat{\beta}_{\text{MLE}} = \arg\max_{\beta} \ell(\beta, \sigma^2) = \arg\min_{\beta} \|y - X\beta\|^2 $$
这与 OLS 的目标函数一致，故：
$$ \hat{\beta}_{\text{MLE}} = \hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T y $$

---

### 4. **关键点**
- 正态分布假设下，对数似然函数中的二次项 $\|y - X\beta\|^2$ 直接对应最小二乘的目标函数。
- OLS 和 MLE 的优化问题在形式上完全一致，因此解相同。

---

### 5. **补充说明**
- 对于 $\sigma^2$ 的估计，MLE 的解为 $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \|y - X\hat{\beta}\|^2$，而 OLS 通常使用无偏估计 $\hat{\sigma}^2_{\text{OLS}} = \frac{1}{n-p} \|y - X\hat{\beta}\|^2$。
- 若误差非正态，MLE 与 OLS 可能不等价（例如误差为拉普拉斯分布时，MLE 对应最小绝对偏差估计）。

---

### 结论
在误差服从正态分布的假设下，$\beta$ 的最小二乘估计与最大似然估计是等价的。
[Timestamp: 2025/05/12 10:19:58]