import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy import optimize

datafile = "ex2data2"
cols = np.loadtxt(datafile,delimiter=",",usecols=(0,1,2),unpack=True)

x = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
x = np.insert(x,0,1,axis=1)

pos = np.array([x[i] for i in range(x.shape[0]) if y[i] == 1])
neg = np.array([x[i] for i in range(x.shape[0]) if y[i] == 0])

def plotData():
    plt.plot(pos[:,1],pos[:,2],'k+',label='y=1')
    plt.plot(neg[:,1],neg[:,2],'yo',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)




def mapFeature( x1col, x2col ):
    """ 
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 6
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 )
            out   = np.hstack(( out, term ))
    return out

mappedx = mapFeature(x[:,1],x[:,2])

def h(mytheta, myx):
    return expit(np.dot(myx,mytheta))

def computeCost(mytheta,myx,myy, mylambda = 0):
    term1 = np.dot(-np.array(myy).T,np.log(h(mytheta,myx)))
    term2 = np.dot((1-np.array(myy)).T,np.log(1-h(mytheta,myx)))
    regterm = (mylambda / 2) * np.sum(np.dot(mytheta[1:].T, mytheta[1:]))
    return float ((1./m)* (np.sum(term1-term2)+regterm))

initialtheta = np.zeros((mappedx.shape[1],1))

def optimizeusingregularization(mytheta,myx,myy,mylambda = 0):
    result = optimize.minimize(computeCost,mytheta,args=(myx,myy,mylambda),method="BFGS",options={"maxiter":500,"disp":False})
    return np.array([result.x]), result.fun

theta, mincost = optimizeusingregularization(initialtheta,mappedx,y)

def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta, mincost = optimizeusingregularization(mytheta,myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")
    plt.show()

plt.figure(figsize=(12,10))
plt.subplot(221)
plotData()
plotBoundary(theta,mappedx,y,1)






