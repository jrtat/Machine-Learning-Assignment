import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
solvers.options['show_progress'] = False  # 隐藏进度信息
solvers.options['abstol'] = 1e-6    # 绝对精度
solvers.options['reltol'] = 1e-6    # 相对精度
solvers.options['feastol'] = 1e-6   # 可行性容差

class CPTWSVM:
    def __init__(self, C1=1.0, C2=1.0, kernel='linear', mu=0.1, tol=1e-4, max_iter=1000):
        self.C1 = C1
        self.C2 = C2
        self.kernel = kernel
        self.mu = mu # RBF核函数用
        self.tol = tol
        self.max_iter = max_iter
        self.models = []  # 多分类One-vs-All模型
        self.classes = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_train_tilde = None  # 训练扩展样本
        self.y_train = None  # 训练标签

    def _kernel(self, X1, X2): # X1和X2为向量
        """核函数计算（线性核、RBF核）"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            dist_sq = cdist(X1, X2, 'sqeuclidean')
            return np.exp(-self.mu * dist_sq)
        else:
            raise ValueError("仅支持'linear'/'rbf'核")

    def _solve_dual(self, Z, y):
        """
        用cvxopt.solvers.qp实现CPTWSVM对偶问题
        :param Z: 原始特征矩阵（m×d，不含偏置项，z_i）
        :param y: 标签向量（m×1，正类=1，负类=-1，y^k）
        :param C1/C2: 惩罚参数
        :param kernel_type/mu: 核函数类型/RBF带宽
        :return: 优化后的乘子与参数w, b
        """

        m, d = Z.shape  # m=样本数，d=特征数
        I1 = np.where(y == 1)[0].tolist()
        m1 = len(I1)

        # 预计算核矩阵K（仅用原始特征，论文1-40节，shape=(m,m)）
        K = self._kernel(Z, Z)
        # 转换为cvxopt矩阵格式（必须为float64）
        K_cvx = matrix(K.astype(np.float64))

        # 初始化拉格朗日乘子
        alpha = np.zeros(m1)  # 仅正样本有alpha（维度m1×1）
        beta = np.zeros(m)  # 全样本有beta（维度m×1）
        gamma = np.zeros(m)  # 全样本有gamma（维度m×1）

        term1 = np.empty(m1)  # 用于alpha更新计算
        term2 = np.empty(m1)  # 用于alpha更新计算
        q_alpha = np.empty(m1)
        q_beta = np.empty(m)
        q_gamma = np.empty(m)

        for iter in range(self.max_iter):
            alpha_old = alpha.copy()
            beta_old = beta.copy()
            gamma_old = gamma.copy()

            # -------------------------- 1. 更新alpha--------------------------
            # 公式28：min 0.5α^T K_{I1I1}α + α^T [K_{I1}(β-γ) + C2 K_{I1}y] - 0.5∑α_i
            # 转化为QP标准形式：P=K_{I1I1}, q=K_{I1}(β-γ) + C2 K_{I1}y - 0.5*1
            # 约束：0 ≤ α ≤ C1
            P_alpha = K_cvx[I1, I1]  # K_{I1I1}（m1×m1）
            # 计算q_alpha（一次项系数）
            term1 = np.dot(K[I1], (beta - gamma))  # K_{I1}(β-γ)
            term2 = self.C2 * np.dot(K[I1], y)  # C2 K_{I1}y
            q_alpha = term1 + term2 - 0.5 * np.ones(m1)
            q_alpha_cvx = matrix(q_alpha.astype(np.float64))

            # 不等式约束：0 ≤ α ≤ C1 → G=[-I; I], h=[0; C1*1]
            G_alpha = matrix(np.vstack([-np.eye(m1), np.eye(m1)]).astype(np.float64))
            h_alpha = matrix(np.hstack([np.zeros(m1), self.C1 * np.ones(m1)]).astype(np.float64))

            # 求解QP（无等式约束，A=None, b=None）
            res_alpha = solvers.qp(P_alpha, q_alpha_cvx, G_alpha, h_alpha)
            alpha = np.array(res_alpha['x']).squeeze()  # 转换为numpy数组

            # -------------------------- 2. 更新beta --------------------------
            # 公式29：min 0.5β^T Kβ + β^T [K(α_full - γ) + C2 K y]
            # 其中α_full：扩展alpha到全样本（负样本处为0）
            # 转化为QP标准形式：P=K, q=K(α_full - γ) + C2 K y
            # 约束：β ≥ 0
            alpha_full = np.zeros(m)
            alpha_full[I1] = alpha  # 扩展alpha到全样本维度
            # 计算q_beta（一次项系数）
            term1 = np.dot(K, (alpha_full - gamma))  # K(α_full - γ)
            term2 = self.C2 * np.dot(K, y)  # C2 K y
            q_beta = term1 + term2
            q_beta_cvx = matrix(q_beta.astype(np.float64))

            # 不等式约束：β ≥ 0 → G=-I, h=0
            G_beta = matrix(-np.eye(m).astype(np.float64))
            h_beta = matrix(np.zeros(m).astype(np.float64))

            # 求解QP
            res_beta = solvers.qp(K_cvx, q_beta_cvx, G_beta, h_beta)
            beta = np.array(res_beta['x']).squeeze()

            # -------------------------- 3. 更新gamma --------------------------
            # 公式30：min 0.5γ^T Kγ + γ^T [-K(α_full + β) - C2 K y + 1]
            # 转化为QP标准形式：P=K, q=-K(α_full + β) - C2 K y + 1
            # 约束：γ ≥ 0
            # 计算q_gamma（一次项系数）
            term1 = -np.dot(K, (alpha_full + beta))  # -K(α_full + β)
            term2 = -self.C2 * np.dot(K, y)  # -C2 K y
            q_gamma = term1 + term2 + np.ones(m)  # +1（常数项）
            q_gamma_cvx = matrix(q_gamma.astype(np.float64))

            # 不等式约束：γ ≥ 0 → G=-I, h=0
            G_gamma = matrix(-np.eye(m).astype(np.float64))
            h_gamma = matrix(np.zeros(m).astype(np.float64))

            # 求解QP
            res_gamma = solvers.qp(K_cvx, q_gamma_cvx, G_gamma, h_gamma)
            gamma = np.array(res_gamma['x']).squeeze()

            # -------------------------- 收敛判断 --------------------------
            delta = np.sqrt(
                np.sum((alpha - alpha_old) ** 2) +
                np.sum((beta - beta_old) ** 2) +
                np.sum((gamma - gamma_old) ** 2)
            )
            delta_old = np.sqrt(
                np.sum(alpha_old ** 2) +
                np.sum(beta_old ** 2) +
                np.sum(gamma_old ** 2)
            )
            if delta / (delta_old + 1e-8) < self.tol:
                print(f"迭代{iter + 1}次收敛", end=" | ")
                break

        return {
            'alpha': alpha, 'beta': beta, 'gamma': gamma, 'I1': I1,
        }

    def fit(self, X, y):
        """训练模型（多分类One-vs-All）"""
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = X_scaled
        self.X_train_tilde = np.hstack([X_scaled, np.ones((X_scaled.shape[0], 1))])
        self.y_train = y
        self.classes = np.unique(y, axis=0)

        # # 普通训练
        # for cls in self.classes:
        #     y_k = np.where(y == cls, 1, -1).astype(float)
        #     model_k = self._solve_dual(self.X_train_tilde, y_k)
        #     self.models.append(model_k)

        # 辅助函数
        def _train_binary_model(cls):
            y_k = np.where(np.all(y == cls, axis=1), 1, -1).astype(float)
            return self._solve_dual(self.X_train_tilde, y_k)

        # 并行训练
        self.models = Parallel(n_jobs=-1)(
            delayed(_train_binary_model)(cls)
            for cls in self.classes
        )

        # 计算训练准确率（Acc）
        y_pred = self.predict(X)
        train_acc = np.mean(y_pred == y) * 100
        print(f"训练集准确率: {train_acc:.2f}%", end=" | ")
        return train_acc

    def predict_proba(self, X):
        """预测概率（概率约束[0,1]）"""
        X_scaled = self.scaler.transform(X)
        m = X_scaled.shape[0]
        prob = np.empty((m, len(self.classes)))

        for k in range(len(self.classes)):
            model_k = self.models[k]

            # 计算核矩阵
            K_test = self._kernel(X_scaled, self.X_train)

            # 构造组合乘子
            m_train = self.X_train.shape[0]
            alpha_full = np.zeros(m_train)
            alpha_full[model_k["I1"]] = model_k['alpha']
            y = np.array([1 if np.all(y_k == self.classes[k]) else -1 for y_k in self.y_train])
            comb_multiplier = alpha_full + model_k['beta'] - model_k['gamma'] + self.C2 * y

            p = np.dot(K_test, comb_multiplier)

            prob[:, k] = p # 对比时不特意进行约束

        # prob = np.exp(prob - np.max(prob, axis=1, keepdims=True))  # 防止溢出
        # prob = prob / np.sum(prob, axis=1, keepdims=True)

        return prob

    def predict(self, X):
        """预测类别"""
        prob = self.predict_proba(X)
        return self.classes[np.argmax(prob, axis=1)]
        # y_pred = np.argmax(prob, axis=1)
        # # 返回one-hot向量
        # return np.array([self.classes[i] for i in y_pred])

# class PSVM:
#     def __init__(self, C=1.0, epsilon=0.1, kernel='linear', mu=0.1, tol=1e-4, max_iter=1000):
#         self.C = C  # 惩罚参数，对应原问题中C/ε的结构
#         self.epsilon = epsilon  # 原问题中的ε
#         self.kernel = kernel
#         self.mu = mu  # RBF核带宽
#         self.tol = tol
#         self.max_iter = max_iter
#         self.alpha = None
#         self.beta = None
#         self.gamma = None
#         self.scaler = None
#         self.X_train = None
#         self.y_train = None  # 标签：正类1，负类-1
#
#     def _kernel(self, X1, X2):
#         """核函数计算（线性核、RBF核）"""
#         if self.kernel == 'linear':
#             return np.dot(X1, X2.T)
#         elif self.kernel == 'rbf':
#             dist_sq = cdist(X1, X2, 'sqeuclidean')
#             return np.exp(-self.mu * dist_sq)
#         else:
#             raise ValueError("仅支持'linear'/'rbf'核")
#
#     def _solve_dual(self, Z, y):
#         """
#         求解PSVM对偶问题（基于cvxopt.qp）
#         :param Z: 特征矩阵（m×d，不含偏置）
#         :param y: 标签向量（m×1，正类1，负类-1）
#         :return: 优化后的乘子alpha, beta, gamma
#         """
#         m, d = Z.shape
#         I = np.arange(m)  # 所有样本的索引
#
#         # 预计算核矩阵K
#         K = self._kernel(Z, Z)
#         K_cvx = matrix(K.astype(np.float64))
#
#         # -------------------------- 构造QP的P, q, G, h, A, b --------------------------
#         # 目标函数：0.5∑(y_iα_i + β_i - γ_i)^T(y_jα_j + β_j - γ_j)<z_i,z_j> - ∑(0.5α_i(y_i+ε) - γ_i)
#         # 转化为QP标准形式：min 0.5 x^T P x + q^T x
#         # 其中x = [α; β; γ]，维度3m
#         P = np.zeros((3*m, 3*m))
#         # 填充P的块：<z_i,z_j> = K[i,j]，因此(y_iα_i + β_i - γ_i)(y_jα_j + β_j - γ_j)K[i,j]
#         for i in range(m):
#             for j in range(m):
#                 coeff = y[i] * y[j] * K[i,j]
#                 P[i, j] += coeff  # α_iα_j项
#                 P[i, m+j] += K[i,j]  # α_iβ_j项
#                 P[i, 2*m+j] -= K[i,j]  # α_iγ_j项
#                 P[m+i, j] += K[i,j]  # β_iα_j项
#                 P[m+i, m+j] += K[i,j]  # β_iβ_j项
#                 P[m+i, 2*m+j] -= K[i,j]  # β_iγ_j项
#                 P[2*m+i, j] -= K[i,j]  # γ_iα_j项
#                 P[2*m+i, m+j] -= K[i,j]  # γ_iβ_j项
#                 P[2*m+i, 2*m+j] += K[i,j]  # γ_iγ_j项
#         P_cvx = matrix(P.astype(np.float64))
#
#         # 一次项q：对α的一次项是 -0.5(y_i+ε)，对γ的一次项是 +1，其余（β）为0
#         q = np.zeros(3*m)
#         for i in range(m):
#             q[i] = -0.5 * (y[i] + self.epsilon)  # α的一次项
#             q[2*m + i] = 1.0  # γ的一次项
#         q_cvx = matrix(q.astype(np.float64))
#
#         # 等式约束：∑(y_iα_i + β_i - γ_i) = 0 → A x = b，其中A是1×3m矩阵，b=0
#         A = np.zeros((1, 3*m))
#         for i in range(m):
#             A[0, i] = y[i]
#             A[0, m+i] = 1.0
#             A[0, 2*m+i] = -1.0
#         A_cvx = matrix(A.astype(np.float64))
#         b_cvx = matrix(np.array([0.0]).astype(np.float64))
#
#         # 不等式约束：
#         # 0 ≤ α_i ≤ C/ε
#         # β_i ≥ 0
#         # γ_i ≥ 0
#         # 构造G和h：G x ≤ h
#         G_rows = []
#         h_rows = []
#         # 约束1：α_i ≥ 0 → -α_i ≤ 0
#         for i in range(m):
#             row = np.zeros(3*m)
#             row[i] = -1.0
#             G_rows.append(row)
#             h_rows.append(0.0)
#         # 约束2：α_i ≤ C/ε → α_i ≤ C/ε
#         for i in range(m):
#             row = np.zeros(3*m)
#             row[i] = 1.0
#             G_rows.append(row)
#             h_rows.append(self.C / self.epsilon)
#         # 约束3：β_i ≥ 0 → -β_i ≤ 0
#         for i in range(m):
#             row = np.zeros(3*m)
#             row[m+i] = -1.0
#             G_rows.append(row)
#             h_rows.append(0.0)
#         # 约束4：γ_i ≥ 0 → -γ_i ≤ 0
#         for i in range(m):
#             row = np.zeros(3*m)
#             row[2*m+i] = -1.0
#             G_rows.append(row)
#             h_rows.append(0.0)
#         G_cvx = matrix(np.array(G_rows).astype(np.float64))
#         h_cvx = matrix(np.array(h_rows).astype(np.float64))
#
#         # 求解QP
#         res = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
#         x = np.array(res['x']).squeeze()
#
#         # 拆分alpha, beta, gamma
#         alpha = x[:m]
#         beta = x[m:2*m]
#         gamma = x[2*m:]
#
#         return alpha, beta, gamma
#
#     def fit(self, X, y):
#         """训练模型"""
#         from sklearn.preprocessing import StandardScaler
#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)
#         self.X_train = X_scaled
#         self.y_train = y
#
#         # 求解对偶问题
#         self.alpha, self.beta, self.gamma = self._solve_dual(X_scaled, y)
#
#         # 计算决策函数参数（以线性核为例，若用RBF核需调整）
#         # 决策函数：f(z) = ∑(y_iα_i + β_i - γ_i)<z, z_i>
#         self.comb_multiplier = y * self.alpha + self.beta - self.gamma
#         return self
#
#     def predict(self, X):
#         """预测类别"""
#         X_scaled = self.scaler.transform(X)
#         K_test = self._kernel(X_scaled, self.X_train)
#         scores = np.dot(K_test, self.comb_multiplier)
#         return np.where(scores >= 0, 1, -1)
#
#     def score(self, X, y):
#         """计算准确率"""
#         y_pred = self.predict(X)
#         return np.mean(y_pred == y)
#
#     def predict_score(self, X):
#         """输出预测得分（决策函数值），用于计算AUC"""
#         X_scaled = self.scaler.transform(X)
#         K_test = self._kernel(X_scaled, self.X_train)
#         scores = np.dot(K_test, self.comb_multiplier)  # PSVM的核心得分计算
#         return scores

class PSVM:
    def __init__(self, C=1.0, epsilon=0.1, kernel='linear', mu=0.1):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.mu = mu
        self.scaler = StandardScaler()
        self.X_train = None  # 标准化后的训练特征
        self.y_train = None  # 编码后的标签（1/-1）
        self.unique_y = None  # 原始标签的唯一值（用于反向映射）
        self.x_opt = None  # 最优变量 [α; β; γ]（长度3m）
        self.w = None  # 线性核的权重
        self.b = None  # 偏置项

    def _kernel(self, X1, X2):
        """核函数计算（线性核、RBF核）"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            dist_sq = np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2)
            return np.exp(-self.mu * dist_sq)
        else:
            raise ValueError("仅支持'linear'/'rbf'核")

    def _construct_qp_params(self, Z, y):
        """
        构造完整QP问题的参数：P, q, G, h, A, b
        :param Z: 标准化后的特征矩阵（m×d）
        :param y: 编码后的标签（m×1，1/-1）
        :return: QP参数（cvxopt矩阵格式）
        """
        m, d = Z.shape
        K = self._kernel(Z, Z) + 1e-6 * np.eye(m)  # 核矩阵+正则化（避免奇异）

        # -------------------------- 1. 二次项矩阵 P（3m × 3m）--------------------------
        # P结构：
        # [K   K  -K]
        # [K   K   0]
        # [-K  0   K]
        P = np.zeros((3 * m, 3 * m))
        P[:m, :m] = K  # α-α交叉项
        P[:m, m:2 * m] = K  # α-β交叉项
        P[:m, 2 * m:] = -K  # α-γ交叉项
        P[m:2 * m, :m] = K  # β-α交叉项
        P[m:2 * m, m:2 * m] = K  # β-β交叉项
        P[2 * m:, :m] = -K  # γ-α交叉项
        P[2 * m:, 2 * m:] = K  # γ-γ交叉项
        P += 2 * np.eye(3 * m) # 严格正则（aus）
        P_cvx = matrix(P.astype(np.float64))

        # -------------------------- 2. 一次项向量 q（3m × 1）--------------------------
        q_alpha = -0.5 * (y + self.epsilon)  # α的一次项
        q_beta = np.zeros(m)  # β的一次项
        q_gamma = np.ones(m)  # γ的一次项
        q = np.hstack([q_alpha, q_beta, q_gamma])
        q_cvx = matrix(q.astype(np.float64))

        # -------------------------- 3. 不等式约束 G x ≤ h --------------------------
        # 约束1：0 ≤ α_i ≤ C/epsilon（2m个约束）
        G_alpha_lower = np.hstack([-np.eye(m), np.zeros((m, 2 * m))])  # -α ≤ 0
        G_alpha_upper = np.hstack([np.eye(m), np.zeros((m, 2 * m))])  # α ≤ C/epsilon
        h_alpha_lower = np.zeros(m)
        h_alpha_upper = (self.C / self.epsilon) * np.ones(m)

        # 约束2：β_i ≥ 0 → -β ≤ 0（m个约束）
        G_beta = np.hstack([np.zeros((m, m)), -np.eye(m), np.zeros((m, m))])
        h_beta = np.zeros(m)

        # 约束3：γ_i ≥ 0 → -γ ≤ 0（m个约束）
        G_gamma = np.hstack([np.zeros((m, 2 * m)), -np.eye(m)])
        h_gamma = np.zeros(m)

        # 合并所有不等式约束
        G = np.vstack([G_alpha_lower, G_alpha_upper, G_beta, G_gamma])
        h = np.hstack([h_alpha_lower, h_alpha_upper, h_beta, h_gamma])
        G_cvx = matrix(G.astype(np.float64))
        h_cvx = matrix(h.astype(np.float64))

        # -------------------------- 4. 等式约束 A x = b --------------------------
        # 约束：sum(y_i α_i + β_i - γ_i) = 0（1个约束）
        A = np.hstack([y.reshape(1, m), np.ones((1, m)), -np.ones((1, m))])
        b = np.zeros(1)
        A_cvx = matrix(A.astype(np.float64))
        b_cvx = matrix(b.astype(np.float64))
        # # 验证关键矩阵的秩（调试用）
        # print(f"K_reg 秩: {np.linalg.matrix_rank(K)}")
        # print(f"A 秩: {np.linalg.matrix_rank(A)}")
        # print(f"P 最小特征值: {np.min(np.real(np.linalg.eigvals(P)))}")  # 应>0

        return P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx

    def fit(self, X, y):
        """训练模型（直接求解完整QP问题）"""
        # 1. 标签处理：二分类标签编码为1/-1
        self.unique_y = np.unique(y)
        if len(self.unique_y) != 2:
            raise ValueError(f"PSVM仅支持二分类，输入数据包含{len(self.unique_y)}个类别")
        self.y_train = np.where(y == self.unique_y[0], -1, 1).astype(np.float64)

        # 2. 特征标准化
        self.X_train = self.scaler.fit_transform(X)
        m, d = self.X_train.shape

        # 3. 构造QP参数
        P, q, G, h, A, b = self._construct_qp_params(self.X_train, self.y_train)

        # 4. 求解QP问题
        res = solvers.qp(P, q, G, h, A, b)
        self.x_opt = np.array(res['x']).squeeze()  # 最优变量 [α; β; γ]

        # 5. 拆分最优乘子
        self.alpha = self.x_opt[:m]
        self.beta = self.x_opt[m:2 * m]
        self.gamma = self.x_opt[2 * m:]

        # 6. 计算线性核的w和b（RBF核无需显式计算w）
        if self.kernel == 'linear':
            comb = self.alpha * self.y_train + self.beta - self.gamma
            self.w = np.dot(self.X_train.T, comb)
            # 支持向量：α/β/γ非零的样本
            sv_idx = np.where((np.abs(self.alpha) > 1e-4) |
                              (np.abs(self.beta) > 1e-4) |
                              (np.abs(self.gamma) > 1e-4))[0]
            if len(sv_idx) > 0:
                self.b = 1 - np.dot(self.w, self.X_train[sv_idx[0]])
            else:
                self.b = 0

        # 7. 计算训练准确率
        y_pred = self.predict(X)
        train_acc = np.mean(y_pred == y) * 100
        print(f"训练完成 | 训练集准确率: {train_acc:.2f}%")
        return train_acc

    def predict(self, X, return_original_label=True):
        """预测类别"""
        if self.x_opt is None:
            raise RuntimeError("模型未训练，请先调用fit方法")

        X_scaled = self.scaler.transform(X)
        m_test = X_scaled.shape[0]

        # 计算决策函数值 f(x)
        if self.kernel == 'linear' and self.w is not None:
            scores = np.dot(X_scaled, self.w) + self.b
        else:
            K_test = self._kernel(X_scaled, self.X_train)
            comb = self.alpha * self.y_train + self.beta - self.gamma
            scores = np.dot(K_test, comb)

        # 编码标签（1/-1）
        y_pred_encoded = np.where(scores >= 0, 1, -1)

        # 可选：反向映射到原始标签
        if return_original_label:
            y_pred = np.where(y_pred_encoded == -1, self.unique_y[0], self.unique_y[1])
            return y_pred
        else:
            return y_pred_encoded

    def predict_proba(self, X):
        """预测概率"""
        X_scaled = self.scaler.transform(X)
        if self.kernel == 'linear' and self.w is not None:
            scores = np.dot(X_scaled, self.w) + self.b
        else:
            K_test = self._kernel(X_scaled, self.X_train)
            comb = self.alpha * self.y_train + self.beta - self.gamma
            scores = np.dot(K_test, comb)

        # Sigmoid转换为概率（约束在[0,1]）
        prob_pos = 1 / (1 + np.exp(-scores))
        prob_neg = 1 - prob_pos
        return np.vstack([prob_neg, prob_pos]).T  # 列顺序：负类概率、正类概率

        # X_scaled = self.scaler.transform(X)
        # K_test = self._kernel(X_scaled, self.X_train)
        # scores = np.dot(K_test, self.comb_multiplier)  # PSVM的核心得分计算
        # return scores