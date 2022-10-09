import numpy as np

def gradient2D(X):
    """return XDh and DvX as defined above"""
    if(len(X.shape)>2):
      raise AssertionError
    else:
        XDh= np.concatenate((np.diff(X),np.zeros((X.shape[0],1))),axis=1) #on fait la différenciation en colonnes et on ajoute une colonne de zéros
        DvX= np.concatenate((np.diff(X.T),np.zeros((X.shape[1],1))),axis=1).T #on fait la différenciation en lignes (d'où le recours à la transposée) et on ajoute une colonne de zéros avant de retransposer
        
        return(XDh,DvX)
      
def tv(X):
    """ compute TV(X) as defined above"""
    XDh,DvX = gradient2D(X)
    sum=0
    for m in range (len(XDh)):
        for n in range (len(XDh[0])):
            sum +=((XDh[m][n]**2 + DvX[m][n]**2)**(1/2))
            
    return(sum)

def gradient2D_adjoint(Y):
    """return D* as defined above"""
    if(len(Y.shape)!=3):
        raise AssertionError
    else:
        Yh = Y[0]

        Yh_1=Yh[:,0].reshape(-1,1)
        Yh_N_1 = Yh[:,-2].reshape(-1,1)

        Yv = Y[1]
        Yv_1_tilde = Yv.T[:,0].reshape(-1,1)
        Yv_N_1_tilde = Yv.T[:,-2].reshape(-1,1)

        print( Yh.shape,Yv.shape)
        YhDh= np.concatenate((-Yh_1,-np.diff(Yh[:,:-1]),Yh_N_1),axis=1)
        DvYv= np.concatenate((-Yv_1_tilde,-np.diff(Yv.T[:,:-1]),Yv_N_1_tilde),axis=1).T

        print( YhDh.shape,DvYv.shape)
        return(YhDh + DvYv)