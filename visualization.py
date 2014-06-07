import numpy
import matplotlib.pyplot as plt
from math import log, fabs, sqrt
from bayes_opt import bayes_opt, GP
from help_functions import acquisition

# ------------------------------ // ------------------------------ // ------------------------------ #
# ------------------------------ // ------------------------------ // ------------------------------ #
def my_function1(array):
    return 2.5*numpy.exp(-numpy.square(array - 2)) + 5*numpy.exp(-numpy.square(array - 5)) +\
           3*numpy.exp(-numpy.square(array/2 - 4))+ 8*numpy.exp(-numpy.square((array - 11)*2))

def my_function2(array):
    return numpy.exp(-numpy.square(array - 2)) + 5*numpy.exp(-numpy.square(array - 5)) +\
           3*numpy.exp(-numpy.square(array/2 - 4))+\
           8*numpy.exp(-numpy.square((array - 0.1)*10)) - numpy.exp(-numpy.square((array - 9)*2))

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
    y = numpy.asarray([my_function1(x) for x in x])

    gp = GP(kernel = 'squared_exp', theta = 1, l = 1)
    gp.fit(x, y, verbose = True)

    mean, var = gp.fast_predict(grid.reshape((len(grid), 1)))

    # ------------------------------ // ------------------------------ #     
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid(True, color='k', linestyle='--', linewidth= 0.8, alpha = 0.4)
    
    p1, = ax1.plot(x, y, 'b-', marker='o', color = 'k')
    p2, = ax1.plot(grid, [(mean[i] + 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
    p3, = ax1.plot(grid, [(mean[i] - 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
    p4, = ax1.plot(grid, numpy.asarray([my_function1(x) for x in grid]), 'm--')
    p5, = ax1.plot(grid, [mean[i] for i in range(len(mean))], 'y')

    p1.set_linestyle(' ')
    ax1.legend([p1, p2, p3, p4, p5],\
               ['Data','Upper 95% bound','Lower 95% bound', 'True function', 'Predicted mean'], loc = 2)

    # ------------------------------ // ------------------------------ # 
    x = numpy.arange(0, 14, 1).reshape((14, 1))
    y = numpy.asarray([my_function1(x) for x in x])

    gp = GP(kernel = 'squared_exp', theta = 0.5, l = .9)
    gp.fit(x, y, verbose = True)

    mean, var = gp.fast_predict(grid.reshape((len(grid), 1)))
    
    # ------------------------------ // ------------------------------ # 
    ax2 = plt.subplot(2, 1, 2)
    ax2.grid(True, color='k', linestyle='--', linewidth=.8, alpha = 0.4)
    
    p12, = ax2.plot(x, y, 'b-', marker='o', color = 'k')
    p22, = ax2.plot(grid, [(mean[i] + 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
    p32, = ax2.plot(grid, [(mean[i] - 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
    p42, = ax2.plot(grid, numpy.asarray([my_function1(x) for x in grid]), 'm--')
    p52, = ax2.plot(grid, [mean[i] for i in range(len(mean))], 'y')

    p12.set_linestyle(' ')
    ax2.legend([p12, p22, p32, p42, p52],\
               ['Data','Upper 95% bound','Lower 95% bound', 'True function', 'Predicted mean'], loc = 2)

    plt.show()


# ------------------------------ // ------------------------------ // ------------------------------ #
def find_max(grid):

    # The target function has to take arrays as entries
    def local_fun(para):
        return my_function1(para[0])

    bo = bayes_opt(local_fun, [(0, 13)])
    ymax, xmax, y, x = bo.maximize(init_points = 5, ei_threshold = 0.001, full_out = True)

    ax = plt.subplot(1,1,1)
    #Read on how to creat these objects properly!
    #ax.suptitle('Finding the maximum', fontsize=20)
    #ax.xlabel('x')
    #ax.ylabel('f(x)')
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
    p2, = ax1.plot(grid, [(mean[i] + 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
    p3, = ax1.plot(grid, [(mean[i] - 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
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
<<<<<<< HEAD
def find_max_log(grid, log_grid):

    # The target function has to take arrays as entries
    def local_fun(para):
        return my_function2(para[0])

    bo = bayes_opt(local_fun, [(0.01, 13)])
    ymax, xmax, y, x = bo.log_maximize(init_points = 5, ei_threshold = 0.001, full_out = True)

    ax = plt.subplot(1,1,1)
    #Read on how to creat these objects properly!
    #ax.suptitle('Finding the maximum', fontsize=20)
    #ax.xlabel('x')
    #ax.ylabel('f(x)')
    ax.grid(True, color='k', linestyle='--', linewidth=.8, alpha = 0.4)

    p1, = ax.plot(x, y, 'b-', marker='o', color = 'k')
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
    y = numpy.asarray([my_function2(0.01 * (10 ** (x * log(13/0.01, 10)))) for x in x])

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
    
    p1, = ax1.plot(x, y, 'b-', marker='o', color = 'k')
    p2, = ax1.plot(log_grid, [(mean[i] + 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
    p3, = ax1.plot(log_grid, [(mean[i] - 2*sqrt(fabs(var[i])))[0] for i in range(len(mean))])
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
    #show_functions(grid, log_grid)
    #gp1(grid)
    #sampled_x = find_max(grid)
    #gp2(grid, sampled_x)
    sampled_x_log = find_max_log(grid, log_grid)
    gp3_log(grid, log_grid, sampled_x_log)

=======
# ------------------------------ // ------------------------------ // ------------------------------ #
if __name__ == "__main__":

    grid = numpy.arange(0.01,13,0.01)
    log_grid = numpy.log10(numpy.arange(0.01,13,0.01)/0.01)/log(13/0.01, 10)

    # ------------------------------ // ------------------------------ # 
    #show_functions(grid, log_grid)
    #gp1(grid)
    sampled_x = find_max(grid)
    gp2(grid, sampled_x)


    '''
    xtrain = numpy.asarray([[1],[2],[3],[3.5],[8],[4.25],[6],[6.5],[9]])
    ytrain = my_function(xtrain) 

    xtest = numpy.linspace(0, 10, 100).reshape(-1, 1)    
    ytest = my_function(xtest)

    mean, var, log_like = GP(xtrain, ytrain, xtest, verbose = True)
    #print('Log likelihood: ', log_like)

    visualisation(xtrain, ytrain, xtest, mean, var, my_function)
    

    #xtr, ytr, xte, me, sig = optimizer(my_function, .1, 14, te_grid = 1000, log_grid = True)
    #visualisation(xtr, ytr, xte, me, sig, my_function)

    
    xtrain = numpy.asarray([[1],[2],[4.25],[6],[6.5],[9]])
    ytrain = my_function(xtrain) 

    xtest = numpy.linspace(0, 10, 100).reshape(-1, 1)    
    ytest = my_function(xtest)

    gp = GP()
    gp.fit(xtrain, ytrain)

    xtrain2d = numpy.asarray([[1,1],[2,3],[4.25,2],[6,5],[6.5,1.5],[9,7]])
    ytrain2d = numpy.asarray([my_2dfunction(xtrain2d) for x in xtrain])

    bo = bayes_opt(my_2dfunction, [(1,10),(1,10)])
'''
>>>>>>> upstream/master
