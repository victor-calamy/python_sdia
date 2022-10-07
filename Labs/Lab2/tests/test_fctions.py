import pytest
from src.fctions import gradient2D, tv
import numpy as np


X_dim_wrong = np.array([[[1,2,3],[4,5,6],[1,3,5],[2,4,6]]]) # de dim(1,4,3) donc doit lever une exception

X1=np.array([[1,2,3],[4,5,6],[1,3,5],[2,4,6]])
X1_res =(np.array([[1., 1., 0.],[1., 1., 0.],[2., 2., 0.],[2., 2., 0.]]),
         np.array([[ 3.,  3.,  3.],[-3., -2., -1.],[ 1.,  1.,  1.],[ 0.,  0.,  0.]]))

X2=np.array([[1,2],[3,4]])
X2_res = (np.array([[1,0],[1,0]]),np.array([[2,2],[0,0]]))

def test_gradient2D():
    assert (np.array_equal(gradient2D(X1),X1_res) and np.array_equal(gradient2D(X2),X2_res))
    
def test_bug():
    with pytest.raises(AssertionError):
        gradient2D(X_dim_wrong)


tv_X1 = (1.+9.)**(1/2) + (1.+9.)**(1/2) + (0.+9.)**(1/2) + (1.+9.)**(1/2) +(1.+4.)**(1/2) + (1.)**(1/2) + (4.+1.)**(1/2) + (4.+1.)**(1/2) + (1.)**(1/2) + (4.)**(1/2) + (4.)**(1/2) 
tv_X2 = (1.+4.)**(1/2) +(4.)**(1/2) +(1.)**(1/2) 
def test_tv():
    assert(tv(X1) == tv_X1 and tv(X2)== tv_X2)
