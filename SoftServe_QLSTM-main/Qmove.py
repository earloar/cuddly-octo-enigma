import numpy as np

def Qmove(Qbit, N, dim, delta_theta):
    """
    更新代理的速度和位置
    :param Qbit: 量子比特矩阵
    :param N: 代理数量
    :param dim: 维度
    :param delta_theta: 每个代理的角度变化矩阵
    :return: 更新后的 Qbit
    """
    for i in range(1, N):
        for j in range(dim):
            alpha = Qbit[i, j, 0]
            beta = Qbit[i, j, 1]

            TETA = delta_theta[i, j]
            # 构建旋转矩阵并应用旋转
            rotation_matrix = np.array([[np.cos(TETA), -np.sin(TETA)], 
                                        [np.sin(TETA), np.cos(TETA)]])
            
            new_angle = np.dot(rotation_matrix, np.array([alpha, beta]))
            Qbit[i, j, 0] = new_angle[0]
            Qbit[i, j, 1] = new_angle[1]
    
    return Qbit
