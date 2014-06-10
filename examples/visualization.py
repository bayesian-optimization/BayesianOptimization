# Python 2.7 users.
from __future__ import print_function
from __future__ import division

import numpy
import matplotlib.pyplot as plt
from math import log, fabs, sqrt, exp

from bayes_opt.bo.bayes_opt import bayes_opt
from bayes_opt.bo.GP import GP
from bayes_opt.support.objects import acquisition

# ------------------------------ // ------------------------------ // ------------------------------ #
# ------------------------------ // ------------------------------ // ------------------------------ #
def my_function1(x):
    return 2.5*exp(-(x - 2)**2) + 5*exp(-(x - 5)**2)+\
           3*exp(-(x/2 - 4)**2)+ 8*exp(-2*(x - 11)**2)

def my_function2(x):
    return exp(-(x - 2)**2) + 5*exp(-(x - 5)**2) +\
           3*exp(-(x/2 - 4)**2) + \
           8*exp(-10*(x - 0.1)**2) - exp(-2*(x - 9)**2)

# ------------------------------ // ------------------------------ // ------------------------------ #
# ------------------------------ // ------------------------------ // ------------------------------ #
def show_functions(grid, log_grid):
    data1 = numpy.asarray([my_function1(x) for x in grid])
    data2 = numpy.asarray([my_function2(x) for x in numpy.arange(0.01,13,0.01)])

    ax1 = plt.subplot(1, 1, 1)
    ax1.grid(True, color='k', linestyle='--', linewidth=1, alpha = 0.5)
    p1, = ax1.plot(grid, data1)
    plt.show()

    ax2 = plt.subplot(2, 1, 1)
    ax2.grid(True, color='k', linestyle='--', linewidth=1, alpha = 0.5)
    p2, = ax2.plot(grid, data2)
    
    ax3 = plt.subplot(2, 1, 2)
    ax3.grid(True, color='k', linestyle='--', linewidth=1, alpha = 0.5)
    p3, = ax3.plot(log_grid, data2)

    plt.show()

# ------------------------------ // ------------------------------ // ------------------------------ #
def gp1(grid):
    x = numpy.asarray([3,6,8,10]).reshape((4, 1))
    y = numpy.asarray([my_function1(x[0]) for x in x])

    gp = GP(kernel = 'squared_exp', theta = 1, l = 1)
    gp.fit(x, y)

    mean, var = gp.fast_predict(grid.reshape((len(grid), 1)))

    # ------------------------------ // ------------------------------ #     
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, color='k', linestyle='--', linewidth= 0.8, alpha = 0.4)

    
    p1, = ax1.plot(x, y, 'b-', marker='o', color = 'k')
    p2, = ax1.plot(grid, [(mean[i] + 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p3, = ax1.plot(grid, [(mean[i] - 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p4, = ax1.plot(grid, numpy.asarray([my_function1(x) for x in grid]), 'm--')
    p5, = ax1.plot(grid, [mean[i] for i in range(len(mean))], 'y')

    p1.set_linestyle(' ')
    ax1.legend([p1, p2, p3, p4, p5],\
               ['Data','Upper 95% bound','Lower 95% bound', 'True function', 'Predicted mean'], loc = 2)

    # ------------------------------ // ------------------------------ # 
    x = numpy.arange(0, 14, 1).reshape((14, 1))
    y = numpy.asarray([my_function1(x[0]) for x in x])

    gp = GP(kernel = 'squared_exp', theta = 0.5, l = .9)
    gp.fit(x, y)

    mean, var = gp.fast_predict(grid.reshape((len(grid), 1)))
    
    # ------------------------------ // ------------------------------ # 
    ax2 = plt.subplot(2, 1, 2)
    ax2.grid(True, color='k', linestyle='--', linewidth=.8, alpha = 0.4)
    
    p12, = ax2.plot(x, y, 'b-', marker='o', color = 'k')
    p22, = ax2.plot(grid, [(mean[i] + 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p32, = ax2.plot(grid, [(mean[i] - 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p42, = ax2.plot(grid, numpy.asarray([my_function1(x) for x in grid]), 'm--')
    p52, = ax2.plot(grid, [mean[i] for i in range(len(mean))], 'y')

    p12.set_linestyle(' ')
    ax2.legend([p12, p22, p32, p42, p52],\
               ['Data','Upper 95% bound','Lower 95% bound', 'True function', 'Predicted mean'], loc = 2)

    plt.show()


# ------------------------------ // ------------------------------ // ------------------------------ #
def find_max(grid):


    bo = bayes_opt(my_function1, {'x' : (0, 13)})
    ymax, xmax, y, x = bo.maximize(init_points = 5, full_out = True)

    ax = plt.subplot(1,1,1)
    ax.grid(True, color='k', linestyle='--', linewidth=.8, alpha = 0.4)

    p1, = ax.plot(x, y, 'b-', marker='o', color = 'k')
    p1.set_linestyle(' ')
    p2, = ax.plot(grid, numpy.asarray([my_function1(x) for x in grid]), 'm-')

    ax.legend([p1, p2],\
               ['Sampled points','Target function'], loc = 2)

    plt.show()

    return x

# ------------------------------ // ------------------------------ // ------------------------------ #
def gp2(grid, sampled_x):
    x = sampled_x
    y = numpy.asarray([my_function1(x) for x in x])

    gp = GP(kernel = 'squared_exp')
    gp.best_fit(x, y)

    mean, var = gp.fast_predict(grid.reshape((len(grid), 1)))
    ymax = y.max()

    ac = acquisition()
    ucb = ac.full_UCB(mean, var)
    ei = ac.full_EI(ymax, mean, var)
    poi = ac.full_PoI(ymax, mean, var)

    # ------------------------------ // ------------------------------ #     
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, color='k', linestyle='--', linewidth= 0.8, alpha = 0.4)
    
    p1, = ax1.plot(x, y, 'b-', marker='o', color = 'k')
    p2, = ax1.plot(grid, [(mean[i] + 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p3, = ax1.plot(grid, [(mean[i] - 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p4, = ax1.plot(grid, numpy.asarray([my_function1(x) for x in grid]), 'm--')
    p5, = ax1.plot(grid, [mean[i] for i in range(len(mean))], 'y')

    p1.set_linestyle(' ')
    ax1.legend([p1, p2, p3, p4, p5],\
               ['Data','Upper 95% bound','Lower 95% bound', 'True function', 'Predicted mean'], loc = 3)


    ax2 = plt.subplot(2,1,2)
    ax2.grid(True, color='k', linestyle='--', linewidth= 0.8, alpha = 0.4)
    p21, = ax2.plot(grid, ucb/ucb.max(), 'r')
    p22, = ax2.plot(grid, ei/(ei.max() + 1e-6), 'orange')
    p23, = ax2.plot(grid, poi, 'green')
    ax2.legend([p21, p22, p23], ['Upper Confidence Bound', 'Expected Improvement', 'Probability of Improvement'], loc = 3)

    plt.show()


# ------------------------------ // ------------------------------ // ------------------------------ #
def find_max_log(grid, log_grid):

    bo = bayes_opt(my_function2, {'x' : (0.01, 13)})
    ymax, xmax, y, x = bo.log_maximize(init_points = 5, full_out = True)

    ax = plt.subplot(1,1,1)
    ax.grid(True, color='k', linestyle='--', linewidth=.8, alpha = 0.4)

    p1, = ax.plot(numpy.log10(x/0.01) / log(13/0.01, 10), y, 'b-', marker='o', color = 'k')
    p1.set_linestyle(' ')
    p2, = ax.plot(log_grid, numpy.asarray([my_function2(x) for x in grid]), 'm-')

    ax.legend([p1, p2],\
               ['Sampled points','Target function'], loc = 2)

    plt.show()

    return x

# ------------------------------ // ------------------------------ // ------------------------------ #
def gp3_log(grid, log_grid, sampled_x):
    '''This is broken, something wrong with the GP and plots, fix it!'''
    x = sampled_x
    y = numpy.asarray([my_function2(x) for x in x])#numpy.asarray([my_function2(0.01 * (10 ** (x * log(13/0.01, 10)))) for x in x])

    gp = GP(kernel = 'squared_exp')
    gp.best_fit(x, y)

    mean, var = gp.fast_predict(grid.reshape((len(log_grid), 1)))
    ymax = y.max()

    ac = acquisition()
    ucb = ac.full_UCB(mean, var)
    ei = ac.full_EI(ymax, mean, var)
    poi = ac.full_PoI(ymax, mean, var)

    # ------------------------------ // ------------------------------ #     
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, color='k', linestyle='--', linewidth= 0.8, alpha = 0.4)
    
    p1, = ax1.plot(numpy.log10(x/0.01) / log(13/0.01, 10), y, 'b-', marker='o', color = 'k')
    p2, = ax1.plot(log_grid, [(mean[i] + 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p3, = ax1.plot(log_grid, [(mean[i] - 2*sqrt(fabs(var[i]))) for i in range(len(mean))])
    p4, = ax1.plot(log_grid, numpy.asarray([my_function2(x) for x in grid]), 'm--')
    p5, = ax1.plot(log_grid, [mean[i] for i in range(len(mean))], 'y')

    p1.set_linestyle(' ')
    ax1.legend([p1, p2, p3, p4, p5],\
               ['Data','Upper 95% bound','Lower 95% bound', 'True function', 'Predicted mean'], loc = 3)


    ax2 = plt.subplot(2,1,2)
    ax2.grid(True, color='k', linestyle='--', linewidth= 0.8, alpha = 0.4)
    p21, = ax2.plot(log_grid, ucb/ucb.max(), 'r')
    p22, = ax2.plot(log_grid, ei/(ei.max() + 1e-6), 'orange')
    p23, = ax2.plot(log_grid, poi, 'green')
    ax2.legend([p21, p22, p23], ['Upper Confidence Bound', 'Expected Improvement', 'Probability of Improvement'], loc = 3)

    plt.show()


# ------------------------------ // ------------------------------ // ------------------------------ #
# ------------------------------ // ------------------------------ // ------------------------------ #
if __name__ == "__main__":

    grid = numpy.arange(0.01,13,0.01)
    log_grid = numpy.log10(numpy.arange(0.01,13,0.01)/0.01)/log(13/0.01, 10)

    # ------------------------------ // ------------------------------ # 
    show_functions(grid, log_grid)
    gp1(grid)
    
    sampled_x = find_max(grid)
    gp2(grid, sampled_x)
    
    sampled_x_log = find_max_log(grid, log_grid)
    gp3_log(grid, log_grid, sampled_x_log)
