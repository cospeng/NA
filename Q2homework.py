import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
import time
import scienceplots
plt.style.use(['science', 'no-latex'])
#plt.rcParams.update({
#    "font.family": "serif",   # specify font family here
    #"font.serif": ["Source Han Serif SC VF"],  # specify font here
    #})          # specify font size here
#def huashangsanjiao(Ab):
    #n = Ab.shape[0]
    #Ab = Ab.astype(float)
    #for k in range(1, n):
        #index = np.argmax(abs(Ab[k - 1 :, k - 1]))
        #Ab[[index + k - 1, k - 1], :] = Ab[[k - 1, index + k - 1], :]
        #for i in range(k + 1, n + 1):
            #Ab[i - 1, k - 1] = Ab[i - 1, k - 1] / Ab[k - 1, k - 1]
            #Ab[i - 1, k:] = Ab[i - 1, k:] - Ab[i - 1, k - 1] * Ab[k - 1, k:]
    #return Ab


#def qiujie(Ab):
    #""" """
    #n = Ab.shape[0]
    #Ab = Ab.astype(float)
    #b = Ab[:, -1]
    #X = np.zeros([n])
    #X[n - 1] = b[n - 1] / Ab[n - 1, n - 1]
    #for i in range(n - 2, -1, -1):
        #X[i] = (b[i] - Ab[i, i + 1 : n].dot(X[i + 1 : n])) / Ab[i, i]
    #return X


#def duichenzhengdingxin(A):
    #tzz = np.linalg.eig(A)[0]
    #return np.all(tzz > 0) and np.all(A == A.T)


#def gs(A, b):
    #n = b.shape[0]
    #x = np.zeros([1001, n])
    #for k in range(1000):
        #e = 0.0
        #for i in range(n):
            #x[k + 1, i] = (
                #b[i] - A[i, :i].dot(x[k, :i]) - A[i, i + 1 :].dot(x[k, i + 1 :])
            #) / A[i, i]
        #e = e + np.max(np.abs(x[k + 1] - x[k]))
        #if e < 0.0001:
            #break
    #return x[k + 1]


def zg(a, b, c, d):
    n = len(b) - 1  # Adjust for 0-based indexing
    u = np.zeros(n + 1)
    l = np.zeros(n + 1)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Initialize the first element
    if b[0] == 0:
        raise ValueError("Matrix is singular.")

    u[0] = b[0]

    # Forward elimination
    for k in range(1, n + 1):
        if u[k - 1] == 0:
            raise ValueError("Matrix is singular.")
        l[k] = a[k] / u[k - 1]
        u[k] = b[k] - l[k] * c[k - 1]  # Update u[k] based on l[k]

    # Back substitution
    y[0] = d[0]

    for k in range(1, n + 1):
        y[k] = d[k] - l[k] * y[k - 1]

    if u[n] == 0:
        raise ValueError("Matrix is singular.")

    x[n] = y[n] / u[n]
    for k in range(n - 1, -1, -1):
        x[k] = (y[k] - c[k] * x[k + 1]) / u[k]

    return x


def Q2j(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([1001, n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    for k in range(max_iter):
        e = 0.0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                u[k + 1, i, j] = (
                    u[k, i - 1, j]
                    + u[k, i + 1, j]
                    + u[k, i, j - 1]
                    + u[k, i, j + 1]
                    + f[i, j]
                ) / 4
                e = e + np.abs(u[k + 1, i, j] - u[k, i, j])
        if e / n**2 < tol:
            break
    print(f"三维数组jacobi迭代次数为：{k+1}")
    return u[k + 1], k+1


def Q2SOR2(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    w = 1.4
    for k in range(max_iter):
        e = 0.0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uo = u[i, j].copy()
                u_new = (
                    u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] + f[i, j]
                ) / 4
                u[i, j] = (1 - w) * uo + w * u_new
                e = e + np.abs(u[i, j] - uo)
        if e / n**2 < tol:
            break
    print(f"SOR2迭代次数为：{k+1}")
    return u, k+1


def Q2gs(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    for k in range(max_iter):
        e = 0.0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uo = u[i, j].copy()
                u[i, j] = (
                    u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] + f[i, j]
                ) / 4
                e = e + np.abs(u[i, j] - uo)
        if e / n**2 < tol:
            break
    print(f"G-S迭代次数为：{k+1}")
    return u, k+1


def Q2J(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    for k in range(max_iter):
        e = 0.0
        uo = u.copy() # 存储上一个迭代步数的解
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uol = u[i, j].copy()
                u[i, j] = (
                    uo[i - 1, j] + u[i + 1, j] + uo[i, j - 1] + u[i, j + 1] + f[i, j]
                ) / 4
                e = e + np.abs(u[i, j] - uol)
        if e / n**2 < tol:
            break
    print(f"Jacobi迭代次数为：{k+1}")
    return u, k+1


def Q2Bgs(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.zeros([n + 2, n + 2])
    f[1 : n + 1, 1 : n + 1] = h**2 * 2  # 仅内部节点赋值
    a = -1 * np.ones(n)
    b = 4 * np.ones(n)
    c = -1 * np.ones(n)
    d = np.zeros(n)
    for k in range(max_iter):
        e = 0.0
        for j in range(1, n + 1):
            u_old = u[:, j].copy()
            d = f[1 : n + 1, j] + u[1 : n + 1, j - 1] + u[1 : n + 1, j + 1]
            x = zg(a, b, c, d)
            u[1 : n + 1, j] = x
            e = e + np.linalg.norm(u_old - u[:, j], 1)
        if e / n**2 < tol:
            break
    print(f"块Gauss-Seider方法迭代次数：{k+1}")
    return u, k+1


def Q2BSOR(n=9, max_iter=1000, tol=1e-5, w=1.4):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.zeros([n + 2, n + 2])
    f[1 : n + 1, 1 : n + 1] = h**2 * 2  # 仅内部节点赋值
    a = -1 * np.ones(n)
    b = 4 * np.ones(n)
    c = -1 * np.ones(n)
    d = np.zeros(n)
    for k in range(max_iter):
        e = 0.0
        for j in range(1, n + 1):
            u_old = u[:, j].copy()
            d = f[1 : n + 1, j] + u[1 : n + 1, j - 1] + u[1 : n + 1, j + 1]
            x = zg(a, b, c, d)
            u[1 : n + 1, j] = w * x + (1 - w) * u_old[1 : n + 1]
            e = e + np.linalg.norm(u_old - u[:, j], 1)
        if e / n**2 < tol:
            break
    print(f"块SOR方法迭代次数：{k+1}")
    return u, k+1


def Q2BSSOR(n=9, max_iter=1000, tol=1e-5, w=1.4):
    u = np.zeros([n + 2, n + 2])
    um = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.zeros([n + 2, n + 2])
    f[1 : n + 1, 1 : n + 1] = h**2 * 2  # 仅内部节点赋值
    a = -1 * np.ones(n)
    b = 4 * np.ones(n)
    c = -1 * np.ones(n)
    for k in range(max_iter):
        e1 = 0.0
        e2 = 0.0
        for j in range(1, n + 1):
            u_old = um[:, j].copy()
            d1 = f[1 : n + 1, j] + um[1 : n + 1, j - 1] + um[1 : n + 1, j + 1]
            x = zg(a, b, c, d1)
            um[1 : n + 1, j] = w * x + (1 - w) * u_old[1 : n + 1]
            e1 = e1 + np.linalg.norm(u_old - um[:, j], 1)
        for j in range(n, 0, -1):
            um_old = um[:, j].copy()
            d2 = f[1 : n + 1, j] + um[1 : n + 1, j - 1] + u[1 : n + 1, j + 1]
            x2 = zg(a, b, c, d2)
            u[1 : n + 1, j] = w * x2 + (1 - w) * um_old[1 : n + 1]
            e2 = e2 + np.linalg.norm(um_old - u[:, j], 1)

        if (e1 + e2) / n**2 < tol:
            break
    print(f"块SSOR方法迭代次数：{k+1}")
    return u, k+1


def Q2Bj(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    a = -1 * np.ones(n)
    b = 4 * np.ones(n)
    c = -1 * np.ones(n)
    d = np.zeros(n)
    for k in range(max_iter):
        e = 0.0
        u_old = u.copy()
        for j in range(1, n + 1):
            u_o_n = u[:, j].copy()
            d = f[1 : n + 1, j] + u_old[1 : n + 1, j - 1] + u[1 : n + 1, j + 1]
            x = zg(a, b, c, d)
            u[1 : n + 1, j] = x
            e = e + np.linalg.norm(u_o_n - u[:, j], 1)
        if e / n**2 < tol:
            break
    print(f"块Jacobi方法迭代次数：{k+1}")
    return u, k+1


def Q2SOR(n=9, max_iter=1000, tol=1e-5, w=1.4):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    for k in range(max_iter):
        e = 0.0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uo = u[i, j].copy()
                u[i, j] = (
                    (
                        (4 / w - 4) * u[i, j]
                        + u[i - 1, j]
                        + u[i + 1, j]
                        + u[i, j - 1]
                        + u[i, j + 1]
                        + f[i, j]
                    )
                    * w
                    / 4
                )
                e = e + np.abs(u[i, j] - uo)
        if e / n**2 < tol:
            break
    print(f"SOR迭代次数为：{k+1}")
    return u, k+1


def Q2SSOR(n=9, max_iter=1000, tol=1e-5, w=1.4):
    u = np.zeros([n + 2, n + 2])
    um = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    w = 1.4
    for k in range(max_iter):
        e1 = 0.0
        e2 = 0.0
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uo = um[i, j].copy()
                um[i, j] = (
                    (
                        (4 / w - 4) * um[i, j]
                        + um[i - 1, j]
                        + um[i + 1, j]
                        + um[i, j - 1]
                        + um[i, j + 1]
                        + f[i, j]
                    )
                    * w
                    / 4
                )
                e1 = e1 + np.abs(um[i, j] - uo)
        for j in range(n, 0, -1):
            for i in range(n, 0, -1):
                uo1 = um[i, j].copy()
                u[i, j] = (
                    (
                        (4 / w - 4) * um[i, j]
                        + um[i - 1, j]
                        + u[i + 1, j]
                        + um[i, j - 1]
                        + u[i, j + 1]
                        + f[i, j]
                    )
                    * w
                    / 4
                )
                e2 = e2 + np.abs(u[i, j] - uo1)
        if (e1 + e2) / n**2 < tol:
            break
    print(f"SSOR迭代次数为：{k+1}")
    return u, k+1


def zsxj(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros(n * n)
    r = np.zeros(n * n)
    h = 1 / (n + 1)
    f = np.full(n * n, h**2 * 2)
    # 中心对角线，所有元素为 4
    main_diag = np.full(n * n, 4)

    # 相邻元素（上下左右）的对角线，所有元素为 -1
    off_diag_1 = np.full(n * n - 1, -1)
    off_diag_1[np.arange(1, n * n) % n == 0] = 0  # 每一行的右端与下一行不相邻

    off_diag_n = np.full(n * n - n, -1)  # 间隔 n 的对角线（行与行之间的相邻关系）

    # 组装对角线数据
    diagonals = [main_diag, off_diag_1, off_diag_1, off_diag_n, off_diag_n]
    offsets = [0, 1, -1, n, -n]

    # 使用scipy的diags函数创建稀疏矩阵
    A = diags(diagonals, offsets, shape=(n * n, n * n), format="csr")

    for k in range(max_iter):
        uo = u.copy()
        r = f - A @ u
        alphak = np.dot(r, r) / np.dot(A @ r, r)
        u = u + alphak * r
        if np.linalg.norm((u - uo)) / n**2 < tol:
            break
    print(f"最速下降迭代次数：{k+1}")
    return u.reshape([n, n]), k+1


def GD(n=9, max_iter=1000, tol=1e-5):
    u = np.zeros([n + 2, n + 2])
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)

    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    for k in range(max_iter):
        r = np.zeros([n + 2, n + 2])

        # 计算残差
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                r[i, j] = f[i, j] - (
                    4 * u[i, j] - u[i - 1, j] - u[i + 1, j] - u[i, j - 1] - u[i, j + 1]
                )

        # 计算步长 alpha
        alpha_k = np.sum(r**2) / np.sum(
            4 * r[1:-1, 1:-1] ** 2
            - r[1:-1, :-2] * r[1:-1, 1:-1]
            - r[1:-1, 2:] * r[1:-1, 1:-1]
            - r[:-2, 1:-1] * r[1:-1, 1:-1]
            - r[2:, 1:-1] * r[1:-1, 1:-1]
        )

        e = 0.0
        # 更新解 u
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uo = u[i, j].copy()
                u[i, j] = u[i, j] + alpha_k * r[i, j]
                e = e + np.abs(u[i, j] - uo)

        # 检查收敛性
        # if np.linalg.norm(u - u_old) / n**2 < tol:
        if e / n**2 < tol:
            break

    print(f"最速下降法迭代次数为：{k+1}")
    return u, k+1


def CG(n=9, max_iter=1000, tol=1e-5):
    # 初始化
    u = np.zeros([n + 2, n + 2])  # 解的网格 (包括边界)
    h = 1 / (n + 1)
    f = np.full([n + 2, n + 2], h**2 * 2)
    # 设置边界条件 (假设为零)
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

    r = np.zeros([n + 2, n + 2])
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            r[i, j] = f[i, j] - (
                4 * u[i, j] - u[i - 1, j] - u[i + 1, j] - u[i, j - 1] - u[i, j + 1]
            )
    p = r.copy()

    for k in range(max_iter):

        # 计算步长 alpha_k
        alpha_k = np.sum(r**2) / np.sum(
            4 * p[1:-1, 1:-1] * p[1:-1, 1:-1]
            - p[1:-1, :-2] * p[1:-1, 1:-1]
            - p[1:-1, 2:] * p[1:-1, 1:-1]
            - p[:-2, 1:-1] * p[1:-1, 1:-1]
            - p[2:, 1:-1] * p[1:-1, 1:-1]
        )

        e = 0.0
        # 更新解 u
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                uo = u[i, j].copy()
                u[i, j] = u[i, j] + alpha_k * p[i, j]
                e = e + np.abs(u[i, j] - uo)

        # 检查收敛性
        if e / n**2 < tol:
            break

        # 更新r
        r_old = r.copy()
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                r[i, j] = r[i, j] - alpha_k * (
                    4 * p[i, j] - p[i - 1, j] - p[i + 1, j] - p[i, j - 1] - p[i, j + 1]
                )

        # 计算beta_k
        beta_k = np.sum(r**2) / np.sum(r_old**2)

        # 计算p(k+1)
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                p[i, j] = r[i, j] + beta_k * p[i, j]

    print(f"CG法迭代次数为：{k+1}")
    return u, k+1


def Q2huatu(data):
    # 创建3D图形
    fig = plt.figure(figsize=(10, 7))

    # 创建3D坐标轴
    ax = fig.add_subplot(111, projection="3d")

    # 定义x, y的值（网格坐标）
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)

    # 绘制3D曲面
    surf = ax.plot_surface(x, y, data, cmap="viridis")

    # 添加颜色条
    fig.colorbar(surf)

    # 设置标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 显示图形
    plt.show()


def main():
    algorithms = {
        '1': Q2J,
        '2': Q2gs,
        '3': Q2SOR,
        '4': Q2Bj,
        '5': Q2Bgs,
        '6': Q2BSOR,
        '7': Q2SSOR,
        '8': Q2BSSOR,
        '9': GD,
        '10': CG
    }
    
    print("请选择要模拟的算法 (用逗号分隔多个编号):")
    for key, value in algorithms.items():
        print(f"{key}: {value.__name__}")
    
    choices = input("输入算法编号: ").split(',')
    plot = input("是否绘图? (y/n): ").strip().lower() == 'y'
    
    metrics = []  # 用于存储每个算法的性能指标
    
    for choice in choices:
        choice = choice.strip()  # 去掉多余的空格
        if choice in algorithms:
            algorithm_function = algorithms[choice]
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行算法并获取结果和迭代次数
            u, k = algorithm_function()
            
            # 记录结束时间和总时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 存储性能指标
            metrics.append({
                'algorithm': algorithm_function.__name__,
                'iterations': k,
                'time': elapsed_time
            })
        else:
            print(f"无效的选择: {choice}")

    # 如果需要绘制对比图
    if plot:
        # 提取数据用于绘图
        algorithms = [m['algorithm'] for m in metrics]
        iterations = [m['iterations'] for m in metrics]
        times = [m['time'] for m in metrics]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 绘制迭代次数柱状图
        ax1.bar(algorithms, iterations, alpha=0.6, label='Iterations')
        ax1.set_ylabel('Iterations')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 创建第二个y轴共享x轴，用于绘制时间数据
        ax2 = ax1.twinx()
        ax2.plot(algorithms, times, marker='o', label='Time (s)')
        ax2.set_ylabel('Time (s)',)
        ax2.tick_params(axis='y', labelcolor='r')

        # 显示图例和标题
        fig.suptitle("Algorithm Performance Comparison")
        fig.tight_layout()  # 调整布局避免重叠
        plt.show()
    else:
        print("算法性能指标:", metrics)


if __name__ == "__main__":
    main()
