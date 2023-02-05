# En este archivo se definen las funciones para los problemas de la tarea 2
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
'''
Ejercicio 1
'''
# Funcion de Rosenbrock para graficar
def f_Rosenbrock_graph(x1,x2):
    return 100.0*(x2-x1**2)**2+(1.0-x1)**2
# Funcion de Rosenbrock
def f_Rosenbrock(x):
    x1,x2=x.T
    return 100.0*(x2-x1**2)**2+(1.0-x1)**2
# Gradiente de Rosenbrock
def grad_Rosenbrock(x):
    x1,x2=x.T
    g1=400.0*(x1**2-x2)*x1+2.0*(x1-1.0)
    g2=200.0*(x2-x1**2)
    return np.array([g1,g2]).reshape((-1,1)) # vector columna
# Hessiana de Rosenbrock
def hess_Rosenbrock(x):
    x1,x2=x.T
    g11=1200.0*x1**2+2-400.0*x2
    g22=200.0
    g12=-400.0*x1
    return np.array([[g11,g12],[g12,g22]]) # matrix de 2x2
'''
Ejercicio 2
'''
def diff_f(func,x,h):
    n=len(x)
    I=np.eye(n)
    grad=np.array([(func(x+h*I[:,i])-func(x))/h for i in range(n)]).reshape(-1,1)
    return grad    
'''
Ejercicio 3
'''
def hess_f(func,x,h):
    n=len(x)
    I=np.eye(n)
    hess=np.zeros((n,n))
    for i in range(n):
        ei=I[:,i].T
        aux1=func(x)
        aux2=func(x+h*ei)
        hess[i,i:]=np.array([(func(x+h*ei+h*I[:,j].T)-aux2-func(x+h*I[:,j].T)+aux1)/h**2 for j in range(i,n)]).reshape(1,-1)
    hess=np.tril(hess.T,-1)+hess
    return hess
