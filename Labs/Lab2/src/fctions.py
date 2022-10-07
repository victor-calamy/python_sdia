import numpy as np

def gradient2D(X):
    """return XDh and DvX as defined above"""
    if(len(X.shape)>2):
      raise AssertionError
    else:
        XDh= np.concatenate((np.diff(X),np.zeros((X.shape[0],1))),axis=1) #on fait la différenciation en colonnes et on ajoute une colonne de zéros
        DvX= np.concatenate((np.diff(X.T),np.zeros((X.shape[1],1))),axis=1).T #on fait la différenciation en lignes (d'où le recours à la transposée) et on ajoute une colonne de zéros avant de retransposer
        
        return((XDh,DvX))
      
def tv(X):
    """ compute TV(X) as defined above"""
    XDh,DvX = gradient2D(X)
    sum=0
    for m in range (len(XDh)):
        for n in range (len(XDh[0])):
            sum +=((XDh[m][n]**2 + DvX[m][n]**2)**(1/2))
            
    return(sum)