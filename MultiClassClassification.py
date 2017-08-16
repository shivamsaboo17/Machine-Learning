import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import scipy.misc
import matplotlib.cm as cm
import random
import scipy.special
from scipy.special import expit
from scipy import optimize


datafile = "ex3data1.mat"
mat = scipy.io.loadmat(datafile)
x, y = mat['X'], mat['y']
x = np.insert(x,0,1,axis=1)

def getDatumImg(row):
    width, height = 20,20
    square = row[1:].reshape(width, height)
    return square.T

def displayData(indices_to_display = None):
    width, height = 20,20
    nrow, ncol = 70,70
    if not indices_to_display:
        indices_to_display = random.sample(range(x.shape[0]), nrow*ncol)
    big_picture = np.zeros((height*nrow, width*ncol))
    irow, icol = 0,0
    for idx in indices_to_display:
        if icol == ncol:
            icol = 0
            irow += 1
        iimg = getDatumImg(x[idx])
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(10, 10))
    img = scipy.misc.toimage(big_picture)
    plt.imshow(img, cmap=cm.Greys_r)
    plt.show()

displayData()

def h(mytheta,myx): #Logistic hypothesis function
    return expit(np.dot(myx,mytheta))

#A more simply written cost function than last week, inspired by subokita:
def computeCost(mytheta,myx,myy,mylambda = 0.):
    m = myx.shape[0] #5000
    myh = h(mytheta,myx) #shape: (5000,1)
    term1 = np.log( myh ).dot( -myy.T ) #shape: (5000,5000)
    term2 = np.log( 1.0 - myh ).dot( 1 - myy.T ) #shape: (5000,5000)
    left_hand = (term1 - term2) / m #shape: (5000,5000)
    right_hand = mytheta.T.dot( mytheta ) * mylambda / (2*m) #shape: (1,1)
    return left_hand + right_hand #shape: (5000,5000)

def costGradient(mytheta,myx,myy,mylambda = 0.):
    m = myx.shape[0]
    #Tranpose y here because it makes the units work out in dot products later
    #(with the way I've written them, anyway)
    beta = h(mytheta,myx)-myy.T #shape: (5000,5000)

    #regularization skips the first element in theta
    regterm = mytheta[1:]*(mylambda/m) #shape: (400,1)

    grad = (1./m)*np.dot(myx.T,beta) #shape: (401, 5000)
    #regularization skips the first element in theta
    grad[1:] = grad[1:] + regterm
    return grad #shape: (401, 5000)

def optimizeTheta(mytheta,myx,myy,mylambda=0.):
    result = optimize.fmin_cg(computeCost, fprime=costGradient, x0=mytheta, \
                              args=(myx, myy, mylambda), maxiter=50, disp=False,\
                              full_output=True)
    return result[0], result[1]

def buildTheta():
    """
    Function that determines an optimized theta for each class
    and returns a Theta function where each row corresponds
    to the learned logistic regression params for one class
    """
    mylambda = 0.
    initial_theta = np.zeros((x.shape[1],1)).reshape(-1)
    Theta = np.zeros((10,x.shape[1]))
    for i in range(10):
        iclass = i if i else 10 #class "10" corresponds to handwritten zero
        print ("Optimizing for handwritten number %d..."%i)
        logic_Y = np.array([1 if x1 == iclass else 0 for x1 in y])#.reshape((x.shape[0],1))
        itheta, imincost = optimizeTheta(initial_theta,x,logic_Y,mylambda)
        Theta[i,:] = itheta
    print ("Done!")
    return Theta

Theta = buildTheta()