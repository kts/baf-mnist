import numpy as np
import baf

def read(f,selectors):
    """
    """
    print "TEMP: reading whole file, then taking selectors. todo: just read selectors."
    dat = baf.read(f)
    return [dat[s] for s in selectors]

def test():
    #X,Y = read(baffile,     selectors = ["Xs","Ys"])
    #X,Y = read(baffile,     selectors = ["Xt","Yt"])
    X,Y = read(baffile,     selectors = ["Xv","Yv"])
    
    W,b = read("model.baf", selectors = ["W", "b"])

    Yhat = (np.dot(X, W) + b).argmax(axis=1)
    nerr = (Yhat != Y).sum()
    
    print "Error: %.3f%% (%d/%d)" % (100 * (float(nerr)/len(Y)), nerr,len(Y))


if __name__ == "__main__":
    #tobaf()
    test()
