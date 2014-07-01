"""
Read the MNIST files, convert and store as BAF.

Stores as a dict with keys,
  Xt,Yt
  Xv,Yv
  Xs,Ys

where,
  X,Y: data and labels
  t: train set
  v: validation set
  s: test set

Need to have the MNIST files,
 wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
 wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
 wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
 wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

"""
import os
import gzip
import numpy as np
import struct
import baf

def read_X_header(fp):
    """
    """
    # 'uchar':
    a = struct.unpack('BBBB',fp.read(4))

    if a != (0,0,8,3):
        raise ValueError("error reading file (first 4 bytes)...")

    (N,d1,d2) = struct.unpack('>iii',fp.read(12))

    if d1 != 28 or d2 != 28:
        raise ValueError("expecting dimensions 28x28")

    return N,d1,d2

def np_fromfile(fp,dtype):
    """
    ugly hack because np.fromfile will not work
    from gzip file object.
    """
    import tempfile
    tf = tempfile.mktemp()
    fp2 = open(tf,'w')
    fp2.write(fp.read())
    fp2.close()

    return np.fromfile(open(tf), dtype, -1)

    
def loadX(f):
    """
    Return array (N,784)   (784=28x28)
    dtype numpy.uint8    
    """
    #fp = open(f,'rb')
    fp = gzip.GzipFile(f, 'rb')

    N,d1,d2 = read_X_header(fp)
    d = d1*d2

    X = np_fromfile(fp, np.uint8)
    
    # make (N,d1*d2)
    return X.reshape((N,d))

def loadY(f):
    """
    Return array (N,)  
    """
    #fp = open(f,'rb')
    fp = gzip.GzipFile(f, 'rb')
    

    # 'uchar':
    a = struct.unpack('BBBB',fp.read(4))

    if a != (0,0,8,1):
        raise ValueError("error reading file (first 4 bytes)...")

    N = struct.unpack('>i',fp.read(4))[0]

    #Y = np.fromfile(fp,np.uint8,-1)
    Y = np_fromfile(fp,np.uint8)

    if len(Y) != N:
        raise ValueError("wrong size Y (%d vs %d).  corrupt file?" % (len(Y),N))

    return Y


def go(datadir, outfile):
    """
    """
    files = {
        #train:
        'Xt': os.path.join(datadir,"train-images-idx3-ubyte.gz"),
        'Yt': os.path.join(datadir,"train-labels-idx1-ubyte.gz"),
        #test:
        'Xs': os.path.join(datadir,"t10k-images-idx3-ubyte.gz"),
        'Ys': os.path.join(datadir,"t10k-labels-idx1-ubyte.gz"),
        }

    if sum(map(os.path.isfile, files.values())) != len(files):
        print r"""

You need all these files:

%s

""" % "\n".join(files.values())
        sys.exit(1)
    
    X = loadX(files['Xt'])
    Y = loadY(files['Yt'])
    print "train data:", X.shape, Y.shape

    ##
    nt = 50000
    Xt = X[:nt,:]
    Yt = Y[:nt]

    Xv = X[nt:,:]
    Yv = Y[nt:]
    
    Xs = loadX(files['Xs'])
    Ys = loadY(files['Ys'])
    print "test data: ", Xs.shape, Ys.shape

    ### I'll store
# convert: uint8 -> float32
#c = lambda x : np.array(x / 255.0, dtype='float32')

    # convert: uint8 -> float32
    ## note: using 256.0 to match deeplearning.net tutorial
    ##       usually would use 255.0 to map values to [0,1], inclusive.
    c = lambda x : np.array(x / 256.0, dtype='float32')
    
    baf.write({
        'Xt': c(Xt),
        'Yt':   Yt,
        'Xv': c(Xv),
        'Yv':   Yv,
        'Xs': c(Xs),
        'Ys':   Ys,
        },
        outfile
    )
    print "Wrote:", outfile
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print r"""

Give two args: datadir outfile

  datadir : directory of the four MNIST .gz files],
  outfile : path of output .baf file
  
"""
        sys.exit(1)

    go(datadir = sys.argv[1],
       outfile = sys.argv[2])
