import matplotlib.pyplot as plt
import numpy as np
import random
import math

N = 10
n = 20
averages_lst = []
for i in range(n):
    tot_sum = 0
    for j in range(N):
        x = random.choice([0, 1])
        tot_sum += x
    averages_lst.append(tot_sum / N)

ls = np.linspace(0, 1, 50)
Y = []
H = []
for e in ls:
    y = 0
    for average in averages_lst:
        if math.fabs(average - 0.5) > e:
            y += 1
    h = 2 * math.pow(math.e, -2 * e**2 * N)
    H.append(h)
    Y.append(y / n)

plt.plot(ls, Y)
plt.plot(ls, H)
plt.show()
