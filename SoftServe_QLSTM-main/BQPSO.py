import numpy as np
from Qinitialization import Qinitialization
from Qobservation import Qobservation
from Qmove import Qmove
from MyCost import Mycost
import random
from scipy.stats import levy


def gliding_distance():
        lift = 0.9783723933835806 / random.uniform(0.675, 1.5)
        drag = 1.630620655639301
        return 8.0 / (18 * drag / lift)
    
def BQPSO(N, dim, max_it):
    Qbit = Qinitialization(N, dim)
    V = np.zeros((N, dim))
    theta_min = 0.001 * np.pi
    theta_max = 0.05 * np.pi
    
    best_fitness_values = []
    for it in range(1, max_it + 1):
        delta_theta = np.zeros((N, dim))
        if it == 1:
            X = Qobservation(Qbit, N, dim)
            fitness = np.array([Mycost(X[aa]) for aa in range(N)])
            sorted_indices = np.argsort(fitness)
            X = X[sorted_indices]
            Qbit = Qbit[sorted_indices]
            delta_theta = delta_theta[sorted_indices]
        # 声明松鼠在不同树上的分布
        n1 = 1  # 山核桃树上的松鼠数量
        n2 = 2  # 橡树上的松鼠数量
        n3 = 7   # 普通树上的松鼠数量
        g_c = 1.9
        theta = theta_max - ((theta_max - theta_min) * (it / max_it))
        for i in range(n1, N):
            R1 = random.random()  # 更新橡树和普通树上的松鼠
            for j in range(dim):
                if R1 >= 0.2:
                    if i < n1 + n2:  # 橡树上的松鼠
                        delta_theta[i][j] = theta * (gliding_distance() * g_c * (X[0][j] - X[i][j]))
                    elif i < n1 + n2 + n3 / 2:  # 普通树上的松鼠
                        delta_theta[i][j] = theta * (gliding_distance() * g_c * (X[random.randint(0, n2-1)][j] - X[i][j]))
                    else:
                        delta_theta[i][j] = theta * (gliding_distance() * g_c * (X[0][j] - X[i][j]))
                else:
                    delta_theta[i][j] = np.random.uniform(theta_min, theta_max)
        # 计算季节常数
        sum = 0
        for i in range(1, n1 + n2):
            for j in range(0, 2):
                sum = sum + (X[i][j] - X[0][j]) ** 2

        Sc = np.sqrt(sum)
        Smin = 1e-5 / (365 ** (2.5 * it / max_it))
        
        # 季节性监测条件
        if Sc < Smin:
            print(1)
            # 随机重新定位松鼠
            for i in range(N - n3, N):
                for j in range(dim):
                    delta_theta[i][j] = theta_min + levy.rvs(size=1) * (theta_max - theta_min)
                    
        Qbit = Qmove(Qbit, N, dim, delta_theta)
        X = Qobservation(Qbit, N, dim)
        fitness = np.array([Mycost(X[aa]) for aa in range(N)])
        sorted_indices = np.argsort(fitness)
        fitness = fitness[sorted_indices]
        Qbit = Qbit[sorted_indices]
        delta_theta = delta_theta[sorted_indices]
        X = X[sorted_indices]
        best_fitness_values.append(fitness[0])
        print(X[0], fitness[0], best_fitness_values)

    return X[0], best_fitness_values[0], best_fitness_values