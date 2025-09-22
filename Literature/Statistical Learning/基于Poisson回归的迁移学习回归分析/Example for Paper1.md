好的，我们回到第一篇论文“Transfer learning for high-dimensional linear regression: Prediction, estimate and minimax optimality”。这篇论文介绍了两种方法：

1. **Algorithm 1: Oracle Trans-Lasso Algorithm** (理想化或理论版本)
    
2. **Algorithm 2: Trans-Lasso Algorithm** (实际可操作版本，对 Algorithm 1 的扩展和聚合)
    

我们将分别用 Python 思路给这两个算法各举一个例子。

---

**论文标题回顾：** "Transfer learning for high-dimensional linear regression: Prediction, estimate and minimax optimality" (高维线性回归中的迁移学习：预测、估计和最小最大最优性)

**核心思想：** 利用辅助（源）任务的数据来帮助提升主（目标）任务在高维线性回归中的预测和估计性能。

---

### **示例一：Algorithm 1 - Oracle Trans-Lasso Algorithm**

**应用场景：药物研发中的生物标志物预测**

问题背景：

假设一家制药公司正在研究一种新药对某种疾病的疗效。他们有一个主要数据集 (Primary Data)，包含了少量接受新药治疗的患者的基因表达数据 (高维特征 X(0)) 和药物疗效指标 (y(0))。同时，他们也有一些辅助数据集 (Auxiliary Data)，这些数据来自之前对类似药物或相关疾病的研究，包含了大量的基因表达数据 (X(k)) 和治疗结果 (y(k))。

目标：

利用辅助数据中蕴含的通用基因表达模式（即共享的生物标志物），来更准确地识别新药疗效的关键基因（变量选择和参数估计），尤其是在主数据集样本量有限、基因特征维度很高的情况下。

**Algorithm 1 核心回顾：**

- **输入：** 主数据 (X(0),y(0)) 和**有信息**的辅助样本 (X(k),y(k))_kinA。
    
- **输出：** hatbetaA。
    
- **步骤1：** 计算共享组件 hatomegaA（通过**岭回归**或类似方法在**辅助数据**上学习）。这代表了从辅助任务中迁移的通用知识。
    
- **步骤2：** 计算任务特定偏差 hatdeltaA（通过 **Lasso 回归**在**主数据**上学习，目标是 delta=beta−omega）。这代表了针对主任务的特有调整。
    
- **最终：** hatbetaA=hatomegaA+hatdeltaA。
    

---

**Python 实现思路 (Oracle Trans-Lasso Algorithm)**

我们将模拟一个简单的场景：一个共享基因模式 omega，以及主任务特有的基因模式 delta。

**环境准备：** `numpy`, `sklearn.linear_model` (用于 Ridge 和 Lasso)。

Python

```
import numpy as np
from sklearn.linear_model import Ridge, Lasso # 从这个库导入Lasso和Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 用于特征标准化，高维Lasso/Ridge常用

# --- 1. 数据模拟 ---
# 模拟主数据集和辅助数据集
np.random.seed(42)

n0_primary = 50   # 主数据集样本量 (较小)
n_auxiliary_total = 500 # 所有辅助样本总和 (较大)
p = 200           # 特征维度 (高维)

# 真实的共享基因模式 (omega_true)
# 假设有10个基因是普遍相关的
omega_true = np.zeros(p)
omega_true[0:10] = np.random.rand(10) * 2 - 1 # -1 到 1 之间随机值

# 真实的任务特定基因模式 (delta_true)
# 假设主任务额外有5个基因是特有的，且不与omega重叠
delta_true = np.zeros(p)
delta_true[15:20] = np.random.rand(5) * 2 - 1

# 真实的主任务参数 beta_true = omega_true + delta_true
beta_true = omega_true + delta_true

# --- 生成数据 ---
# 协变量 X: 标准正态分布
X_primary = np.random.randn(n0_primary, p)
X_auxiliary = np.random.randn(n_auxiliary_total, p)

# 响应变量 y: 线性模型 + 噪声
y_primary = X_primary @ beta_true + np.random.randn(n0_primary) * 0.5
y_auxiliary = X_auxiliary @ omega_true + np.random.randn(n_auxiliary_total) * 0.8 # 辅助数据只与omega相关

# 特征标准化 (很重要，尤其是L1/L2惩罚模型)
scaler = StandardScaler()
X_primary_scaled = scaler.fit_transform(X_primary)
X_auxiliary_scaled = scaler.fit_transform(X_auxiliary) # 注意：这里简单fit_transform辅助数据，实际中如果希望保持一致性，可能用主数据scaler来transform辅助数据

print(f"数据模拟完成：主数据 {n0_primary} 样本，辅助数据 {n_auxiliary_total} 样本， {p} 个特征。")
print(f"真实omega非零数量：{np.sum(omega_true != 0)}")
print(f"真实delta非零数量：{np.sum(delta_true != 0)}")
print(f"真实beta非零数量：{np.sum(beta_true != 0)}")

# --- 2. Algorithm 1: Oracle Trans-Lasso Algorithm 实现 ---

def oracle_trans_lasso(X0, y0, X_aux, y_aux):
    """
    实现 Oracle Trans-Lasso Algorithm (Algorithm 1)
    X0, y0: 主数据集
    X_aux, y_aux: 辅助数据集 (这里假设A就是所有X_aux, y_aux)
    """
    p = X0.shape[1]

    # Step 1: Compute shared component omega_hat_A (计算共享组件 omega_hat_A)
    # 原文公式(4)为岭回归，惩罚参数 lambda_omega
    # sklearn Ridge的alpha参数是惩罚系数，通常较小的值。
    # 论文中 lambda_omega = c1 * sqrt(log p / n_A)
    n_A = X_aux.shape[0]
    c1 = 1.0 # 假设一个常数
    lambda_omega_param = c1 * np.sqrt(np.log(p) / n_A)
    
    # Lasso和Ridge的alpha参数命名在sklearn中有些混淆
    # Lasso: (1 / (2 * n_samples)) * ||y - Xw||^2 + alpha * ||w||_1
    # Ridge: ||y - Xw||^2 + alpha * ||w||_2^2
    # 为了匹配论文的 ||y - Xw||^2 + lambda_omega * ||w||^2
    # sklearn.Ridge 的 alpha 应该直接是 lambda_omega
    
    ridge_model = Ridge(alpha=lambda_omega_param) 
    ridge_model.fit(X_aux, y_aux)
    omega_hat_A = ridge_model.coef_
    
    print(f"\nStep 1 (Oracle): 估计的omega_hat_A非零数量: {np.sum(omega_hat_A != 0)}")

    # Step 2: Compute task-specific component delta_hat_A (计算任务特定组件 delta_hat_A)
    # 原文公式(6)为Lasso回归，惩罚参数 lambda_delta
    # 目标是最小化 ||y0 - X0(omega_hat_A + delta)||^2 + lambda_delta * ||delta||_1
    # 这等价于最小化 ||(y0 - X0 @ omega_hat_A) - X0 @ delta||^2 + lambda_delta * ||delta||_1
    # 令 pseudo_y = y0 - X0 @ omega_hat_A，这就是一个标准的Lasso问题
    
    pseudo_y = y0 - X0 @ omega_hat_A
    
    c2 = 1.0 # 假设一个常数
    n0 = X0.shape[0]
    lambda_delta_param = c2 * np.sqrt(np.log(p) / n0)
    
    # sklearn.Lasso 的 alpha 惩罚项是 alpha * ||w||_1
    # 为了匹配论文的 lambda_delta * ||delta||_1
    # sklearn.Lasso 的 alpha 应该直接是 lambda_delta
    
    lasso_model = Lasso(alpha=lambda_delta_param, max_iter=2000) # 增加迭代次数防止不收敛
    lasso_model.fit(X0, pseudo_y)
    delta_hat_A = lasso_model.coef_
    
    print(f"Step 2 (Oracle): 估计的delta_hat_A非零数量: {np.sum(delta_hat_A != 0)}")

    # Step 2: Combine to get beta_hat_A
    beta_hat_A = omega_hat_A + delta_hat_A
    
    print(f"最终beta_hat_A非零数量: {np.sum(beta_hat_A != 0)}")
    return beta_hat_A

# --- 3. 运行 Oracle Trans-Lasso Algorithm ---
estimated_beta_oracle = oracle_trans_lasso(X_primary_scaled, y_primary, X_auxiliary_scaled, y_auxiliary)

# --- 4. 评估结果 (与真实beta比较) ---
print("\n--- Oracle Trans-Lasso 结果评估 ---")
print("真实beta非零项索引:", np.where(beta_true != 0)[0])
print("估计beta非零项索引:", np.where(estimated_beta_oracle != 0)[0])

# 简单比较非零系数的准确性 (仅作为示例，实际评估需更严谨)
true_positive = np.sum((beta_true != 0) & (estimated_beta_oracle != 0))
false_positive = np.sum((beta_true == 0) & (estimated_beta_oracle != 0))
false_negative = np.sum((beta_true != 0) & (estimated_beta_oracle == 0))

print(f"真正例 (TP): {true_positive}")
print(f"假正例 (FP): {false_positive}")
print(f"假反例 (FN): {false_negative}")

# 预测性能 (R^2)
from sklearn.metrics import r2_score
y_pred_oracle = X_primary_scaled @ estimated_beta_oracle
r2_oracle = r2_score(y_primary, y_pred_oracle)
print(f"预测R^2 (Oracle Trans-Lasso): {r2_oracle:.4f}")

# 作为对比，单独用Lasso在主数据上训练
lasso_primary = Lasso(alpha=0.1, max_iter=2000) # alpha需调优
lasso_primary.fit(X_primary_scaled, y_primary)
y_pred_lasso_primary = X_primary_scaled @ lasso_primary.coef_
r2_lasso_primary = r2_score(y_primary, y_pred_lasso_primary)
print(f"预测R^2 (仅使用主数据的Lasso): {r2_lasso_primary:.4f}")
print(f"仅使用主数据的Lasso非零数量: {np.sum(lasso_primary.coef_ != 0)}")
```

**Oracle Trans-Lasso 示例讲解：**

1. **数据模拟：** 我们创建了一个小的主数据集（50样本）和大的辅助数据集（500样本），以及高维特征（200个）。`omega_true` 代表了辅助任务和主任务共享的真实基因模式，`delta_true` 代表了主任务特有的基因模式。最终主任务的真实参数 `beta_true` 是两者的和。
    
2. **特征标准化：** 在应用 L1/L2 正则化模型之前，对特征进行标准化是最佳实践，以确保惩罚对所有特征公平。
    
3. **`oracle_trans_lasso` 函数：**
    
    - **Step 1 (计算 hatomegaA)：** 这一步使用辅助数据集 `X_aux`, `y_aux` 和 `sklearn.linear_model.Ridge` 来估计共享参数 `omega_hat_A`。`alpha` 参数 `lambda_omega_param` 是根据论文中的公式 `c1 * np.sqrt(np.log(p) / n_A)` 计算的。
        
    - **Step 2 (计算 hatdeltaA)：** 这一步是关键的迁移学习部分。我们首先计算一个“伪响应变量”`pseudo_y = y0 - X0 @ omega_hat_A`。这个伪响应代表了主任务中**残余的、未被共享知识解释的部分**。然后，我们使用 `sklearn.linear_model.Lasso` 对这个 `pseudo_y` 和主数据 `X0` 进行回归，来估计 `delta_hat_A`。`lambda_delta_param` 同样根据论文公式计算。Lasso 的 L1 惩罚能够使 `delta_hat_A` 稀疏，识别出主任务特有的少数关键基因。
        
    - **组合：** 最终的 `beta_hat_A` 是 `omega_hat_A` 和 `delta_hat_A` 的简单相加。
        
4. **结果评估：** 我们比较了估计的 `beta_hat_A` 的非零系数位置与真实 `beta_true` 的非零系数位置。并计算了模型在主数据上的预测 R^2 值。为了体现迁移学习的优势，我们还计算了仅使用主数据训练的 Lasso 模型的 R^2 和非零系数数量，通常可以看到 Oracle Trans-Lasso 在小样本主数据集上表现更好。
    

---

### **示例二：Algorithm 2 - Trans-Lasso Algorithm**

**应用场景：推荐系统中的用户行为预测**

问题背景：

假设一家电商公司希望预测用户对新上架商品（主任务）的购买意愿。他们有一个主数据集：少量新商品的试用用户数据 (X(0)，用户特征如年龄、性别、地域、浏览历史等；y(0)，购买意愿评分)。同时，他们有大量的辅助数据集：来自过去不同类型商品（例如电子产品、服装、图书等，每个类型对应一个辅助任务 k）的购买数据 (X(k),y(k))。

目标：

通过聚合从不同商品类型（辅助任务）中学习到的用户行为模式，来更准确地预测用户对新商品（主任务）的购买意愿，尤其是在新商品数据不足时。

**Algorithm 2 核心回顾：**

- **输入：** 主数据 (X(0),y(0)) 和所有辅助样本 (X(k),y(k))_kinA。
    
- **输出：** 最终聚合得到的 hatbeta。
    
- **步骤1：** 从每个辅助样本（可能结合部分主数据）生成一系列初步的候选估计量 hatbeta(k)。
    
- **步骤2：** 构建候选集 mathcalG_l（表示不同的辅助信息组合策略）。
    
- **步骤3：** 对于每个 mathcalG_l，运行 **Oracle Trans-Lasso Algorithm (Algorithm 1)**，得到一系列基于不同辅助信息组合的候选估计量 hatbeta_l(G)。
    
- **步骤4：** 计算最终的聚合估计量 hatbeta，通过最小化一个包含预测误差和 Kullback-Leibler 惩罚的优化问题来确定权重，或者直接选择最佳的。
    

---

**Python 实现思路 (Trans-Lasso Algorithm)**

由于 Algorithm 2 涉及多重循环和调用 Algorithm 1，以及复杂的聚合（权重计算和 KL 惩罚），我们将简化其中一些步骤以展示核心概念。

**简化假设：**

- Step 1 的 hatbeta(k) 我们就直接用 Lasso 在每个辅助数据集上训练得到。
    
- Step 2 的 mathcalG_l 我们简化为直接使用不同数量的辅助数据集组合。
    
- Step 3 直接调用我们上面实现的 `oracle_trans_lasso`。
    
- Step 4 的聚合我们简化为：不是计算复杂的 KL 惩罚，而是通过**交叉验证**来选择最佳的 hatbeta_l(G)（即，在主数据的验证集上表现最好的）。或者，更符合论文精神，我们尝试使用一个简单的加权平均，权重根据在主数据验证集上的表现确定。
    

**环境准备：** `numpy`, `sklearn.linear_model`, `sklearn.model_selection`, `scipy.optimize`。

Python

```
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold # 用于交叉验证
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- 1. 数据模拟 (沿用Oracle示例的数据结构，但增加多个辅助任务) ---
np.random.seed(42)

n0_primary = 50   # 主数据集样本量
n_auxiliary_per_task = 100 # 每个辅助任务的样本量
num_aux_tasks = 5 # 5个辅助任务
p = 200           # 特征维度

# 真实的共享组件和任务特定组件
omega_true = np.zeros(p)
omega_true[0:10] = np.random.rand(10) * 2 - 1

delta_true = np.zeros(p)
delta_true[15:20] = np.random.rand(5) * 2 - 1

beta_true = omega_true + delta_true

# 主数据
X_primary = np.random.randn(n0_primary, p)
y_primary = X_primary @ beta_true + np.random.randn(n0_primary) * 0.5

# 辅助数据 (每个任务的omega可能略有不同，以模拟真实世界)
# 简化：这里所有辅助任务都接近omega_true
auxiliary_datasets = []
for k in range(num_aux_tasks):
    X_k = np.random.randn(n_auxiliary_per_task, p)
    # 辅助任务的真实beta可能略有偏差
    beta_k_true = omega_true + np.random.randn(p) * 0.05 # 给omega_true加点小噪声
    y_k = X_k @ beta_k_true + np.random.randn(n_auxiliary_per_task) * 0.8
    auxiliary_datasets.append({'X': X_k, 'y': y_k})

# 特征标准化
scaler = StandardScaler()
X_primary_scaled = scaler.fit_transform(X_primary)
# 对每个辅助数据集独立进行标准化
auxiliary_datasets_scaled = []
for aux_data in auxiliary_datasets:
    auxiliary_datasets_scaled.append({
        'X': scaler.fit_transform(aux_data['X']), # 实际中可能用主数据的scaler来transform
        'y': aux_data['y']
    })

print(f"数据模拟完成：主数据 {n0_primary} 样本， {num_aux_tasks} 个辅助任务，每个 {n_auxiliary_per_task} 样本， {p} 个特征。")

# 假设我们之前定义的 oracle_trans_lasso 函数可用
# （为避免重复，这里不再次定义，请确保您运行示例一时已运行过该定义）

# --- 2. Algorithm 2: Trans-Lasso Algorithm 实现 ---

def trans_lasso_algorithm(X0, y0, aux_data_list, alpha_range_lasso=np.logspace(-2, 0, 5), alpha_range_ridge=np.logspace(-2, 0, 5)):
    """
    实现 Trans-Lasso Algorithm (Algorithm 2) 简化版。
    X0, y0: 主数据集
    aux_data_list: 辅助数据集列表，每个元素是 {'X': X_k, 'y': y_k}
    alpha_range_lasso: Lasso惩罚参数的范围 (用于Step 1生成候选估计量 和 Step 3的Oracle)
    alpha_range_ridge: Ridge惩罚参数的范围 (用于Step 3的Oracle)
    """
    n0 = X0.shape[0]
    p = X0.shape[1]
    
    # 将主数据划分为训练集和验证集 (用于聚合时的权重确定)
    X0_train, X0_val, y0_train, y0_val = train_test_split(X0, y0, test_size=0.3, random_state=42)

    all_candidate_betas_G = [] # 存储 Step 3 生成的所有 beta_l^(G)

    # Step 1 & 2 (简化)：生成不同的辅助数据组合 (G)
    # 我们可以通过不同地选择辅助任务来形成不同的G
    # 示例：只考虑单个辅助任务，或所有辅助任务的组合
    
    # 策略 1：单独使用每个辅助任务 + 主数据
    for k_idx, aux_data in enumerate(aux_data_list):
        print(f"\nStep 3: 调用 Oracle Trans-Lasso with auxiliary task {k_idx+1}")
        # 这里模拟G只包含一个辅助任务的情况
        beta_G_k = oracle_trans_lasso(X0_train, y0_train, aux_data['X'], aux_data['y'])
        all_candidate_betas_G.append(beta_G_k)

    # 策略 2：使用所有辅助任务的集合 + 主数据 (模拟一个大的G)
    print("\nStep 3: 调用 Oracle Trans-Lasso with ALL auxiliary tasks")
    X_all_aux = np.vstack([aux['X'] for aux in aux_data_list])
    y_all_aux = np.concatenate([aux['y'] for aux in aux_data_list])
    beta_G_all = oracle_trans_lasso(X0_train, y0_train, X_all_aux, y_all_aux)
    all_candidate_betas_G.append(beta_G_all)

    # Step 4: Compute final aggregated estimator beta_hat (计算最终聚合估计量 beta_hat)
    # 论文原文是使用KL惩罚的复杂优化，这里我们简化为：
    # 1. 计算每个候选beta在主数据验证集上的表现 (MSE)
    # 2. 根据表现赋予权重 (例如，MSE越低权重越高)
    # 3. 加权平均这些候选beta

    # 计算每个候选beta在验证集上的MSE
    mses = []
    for candidate_beta in all_candidate_betas_G:
        y_pred_val = X0_val @ candidate_beta
        mse = mean_squared_error(y0_val, y_pred_val)
        mses.append(mse)
    
    # 转换为权重：MSE越小，权重越大。例如，使用 exp(-mse) 或 1/mse
    # 为了数值稳定性，对MSE进行归一化或者取倒数再归一化
    inverse_mses = 1 / (np.array(mses) + 1e-6) # 加一个小的常数避免除以零
    weights = inverse_mses / np.sum(inverse_mses) # 归一化，使得和为1

    print(f"\nStep 4: 候选模型的验证集MSE: {np.round(mses, 4)}")
    print(f"对应的聚合权重: {np.round(weights, 4)}")

    # 加权平均得到最终的beta_hat
    final_beta_hat = np.zeros(p)
    for i, candidate_beta in enumerate(all_candidate_betas_G):
        final_beta_hat += weights[i] * candidate_beta
        
    print(f"最终聚合beta的非零数量: {np.sum(final_beta_hat != 0)}")
    return final_beta_hat

# --- 3. 运行 Trans-Lasso Algorithm ---
estimated_beta_translasso = trans_lasso_algorithm(X_primary_scaled, y_primary, auxiliary_datasets_scaled)

# --- 4. 评估结果 ---
print("\n--- Trans-Lasso (聚合版) 结果评估 ---")
print("真实beta非零项索引:", np.where(beta_true != 0)[0])
print("估计beta非零项索引:", np.where(estimated_beta_translasso != 0)[0])

y_pred_translasso = X_primary_scaled @ estimated_beta_translasso
r2_translasso = r2_score(y_primary, y_pred_translasso)
print(f"预测R^2 (Trans-Lasso 聚合版): {r2_translasso:.4f}")

# 再次与仅使用主数据的Lasso对比
# (Lasso_primary 已在Oracle示例中计算过，这里直接引用或重新计算)
lasso_primary = Lasso(alpha=0.1, max_iter=2000) 
lasso_primary.fit(X_primary_scaled, y_primary)
y_pred_lasso_primary = X_primary_scaled @ lasso_primary.coef_
r2_lasso_primary = r2_score(y_primary, y_pred_lasso_primary)
print(f"预测R^2 (仅使用主数据的Lasso): {r2_lasso_primary:.4f}")
print(f"仅使用主数据的Lasso非零数量: {np.sum(lasso_primary.coef_ != 0)}")
```

**Trans-Lasso (Algorithm 2) 示例讲解：**

1. **数据模拟：** 增加了多个辅助任务，每个任务对应一个独立的数据集。每个辅助任务的真实参数 `beta_k_true` 都是 `omega_true` 加上一点噪声，模拟辅助任务与主任务既相关又不完全相同的情况。
    
2. **主数据划分：** 为了在 Step 4 中进行模型聚合时评估各个候选模型的性能，我们将主数据集划分为训练集和验证集。
    
3. **`trans_lasso_algorithm` 函数：**
    
    - **Step 1 & 2 简化：** 论文中的 Step 1 和 Step 2 涉及生成多种候选估计量和构建候选集 mathcalG。为了简化，我们在这里直接考虑两种 `G` 的策略：
        
        - 分别使用**每个单独的辅助任务**作为 `G`。
            
        - 使用**所有辅助任务的组合**作为一个大的 `G`。
            
        - 对于每个这样的 `G`，我们都调用了前面实现的 `oracle_trans_lasso` 函数（对应论文中的 Step 3），得到一个 `beta_G` 估计量。这些构成了 `all_candidate_betas_G` 列表。
            
    - **Step 4 (简化聚合)：** 这是与论文原文差异最大的简化点。原文使用了基于 Kullback-Leibler 惩罚的复杂优化来确定权重。在这里，我们采取了一个更直观的简化方法：
        
        - 计算每个 `all_candidate_betas_G` 中的候选模型在主数据的**验证集**上的均方误差 (MSE)。
            
        - 根据 MSE 赋予权重：MSE 越小（模型表现越好），其权重越大。我们使用了 `1 / (MSE + epsilon)` 的形式，然后归一化。
            
        - 最终的 `final_beta_hat` 是所有候选 `beta_G` 的**加权平均**。
            

**关键的简化和注意事项：**

- **惩罚参数的选择：** 在两个算法中，我为 Lasso 和 Ridge 的 `alpha` 参数（即惩罚强度）设定了简单的值或根据论文公式计算，但这些值通常需要通过交叉验证或更精细的网格搜索在实际应用中进行调优。
    
- **X∗ 和 beta∗ 的模糊性：** 尤其是在 Algorithm 1 中，X∗ 和 beta∗ 的精确定义在原文中非常浓缩。我采用了标准统计建模中将 GLM 转化为加权最小二乘/Lasso 问题的常见方法。如果需要严格复现论文，可能需要更深入的数学推导。
    
- **Algorithm 2 的聚合：** 论文中 Algorithm 2 的 Step 4 是一个复杂的优化问题，涉及 KL 惩罚。我的简化版使用了基于验证集 MSE 的加权平均，这在实践中也是常见的模型聚合策略，但与论文原文的方法不同。实现原文的 KL 聚合需要解决一个带有特定约束的优化问题，通常会使用 `scipy.optimize` 等更高级的优化工具，并自定义损失函数。
    
- **计算效率：** Algorithm 2 在每个迭代中会多次调用 Algorithm 1（Lasso 和 Ridge），这在高维数据和大量辅助任务的情况下计算成本较高。
    

这两个例子展示了如何将论文中的理论算法概念性地转化为 Python 代码。请记住，实际应用中可能需要更复杂的细节处理和更严格的数学推导，尤其是在处理正则化参数、稀疏性以及算法收敛性方面。