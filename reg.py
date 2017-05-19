from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

#xs = np.array([1,2,3,4,5,6],dtype=float)
#ys = np.array([5,4,6,5,6,7], dtype=float)

def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step
        xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=float) , np.array(ys, dtype=float)

def best_fit_line_and_intercept(xs,ys):
    m = ((mean(xs)*mean(ys)) - (mean(xs * ys)))/ ((mean(xs)*mean(xs)) - mean(xs * xs))
    b  = mean(ys) - m * mean(xs)
    return m , b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    mean_line = [mean(ys_orig) for y in ys_orig]
    reg_line_err = squared_error(ys_orig, ys_line)
    mean_line_err = squared_error(ys_orig, mean_line)
    return 1 - (reg_line_err/mean_line_err)

xs, ys = create_dataset(100, 60, 2, correlation="neg")
m, b = best_fit_line_and_intercept(xs, ys)
print(m , b)

regression_line = [(m*x)+b for x in xs]

r = coefficient_of_determination(ys, regression_line)
print(r)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)

plt.show()