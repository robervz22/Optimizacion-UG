import numpy as np
import matplotlib.pyplot as plt
from pytest import param
import scipy.linalg as linalg

N_BACKTRACKING=1000

'''
Ejercicio 1
'''
# Función cuadrática
def quad_fun(x):
    n=len(x)
    x_col=x.reshape(-1,1)
    A=n*np.eye(n)+np.ones((n,n))
    b=np.ones(n).reshape(-1,1)
    value=x_col.T@A@x_col/2.0-b.T@x_col
    return np.squeeze(value).item()

# Gradiente función cuadrática
def grad_quad_fun(x):
    n=len(x)
    x_col=x.reshape(-1,1)
    A=n*np.eye(n)+np.ones((n,n))
    b=np.ones(n).reshape(-1,1)
    grad=A@x_col-b
    return np.squeeze(grad.reshape(1,-1))

# Rosenbrock general
def gen_rosenbrock(x):
    n=len(x)
    sum1=100.0*np.sum((x[1:]-x[:-1]**2)**2)
    sum2=np.sum((1.0-x[:-1])**2)
    return sum1+sum2

# Gradiente Rosenbrock general
def grad_gen_rosenbrock(x):
    # Primer entrada
    aux1=-400.0*(x[1]-x[0]**2)*x[0]-2.0*(1.0-x[0])
    # De la segunda a la n-2
    xi=x[1:-1]
    xi_minus_1=x[:-2]
    xi_plus_1=x[2:]
    aux2=200.0*(xi-xi_minus_1**2)-400.0*(xi_plus_1-xi**2)*xi-2.0*(1.0-xi)
    # Ultima entrada
    aux3=200.0*(x[-1]-x[-2]**2)

    return np.concatenate(([aux1],aux2,[aux3]))

'''
Ejercicio 2
'''
# Algoritmo backtracking para tamaño de paso  
def backtracking(f,fk,gk,xk,pk,a0,rho,c):
    a=a0
    k=0
    while f(xk+a*pk)>fk+c*a*np.squeeze(gk.T@pk) and k<N_BACKTRACKING:
        a=rho*a
        k+=1
    return a

# Algoritmo general Fletcher Reeves
def fletcher_reeves(f,g,x0,N,tol,rho):
    a0,c=2.0,1e-4 # Tamaño inicial y factor de proporción fijo para las pruebas
    xk,res,k=x0,0,0
    pk=-g(xk)
    while k<N:
        gk=g(xk)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        fk=f(xk)
        ak=backtracking(f,fk,gk,xk,pk,a0,rho,c)
        xk=xk+ak*pk
        new_gk=g(xk)
        beta_k=np.squeeze(new_gk.T@new_gk/(gk.T@gk))
        pk=-new_gk+beta_k*pk
        k+=1
        if k>=N:
            res=0
            break
    fk=f(xk)
    gk=g(xk)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}

# Prueba Fletcher-Reeves
def proof_fletcher_reeves(f,g,x0,N,tol,rho):
    n=len(x0)
    dic_results=fletcher_reeves(f,g,x0,N,tol,rho)
    if dic_results['res']==1:
        print('El algoritmo de Fletcher-Reeves CONVERGE')
        print('n = ',n)
        print('f(x0) = ',f(x0))
        if n>8:
            print('Primer y últimas 4 entradas de xk = ',np.squeeze(dic_results['xk'][:4]),"...",np.squeeze(dic_results['xk'][-4:]))
        else:
            print('xk = ',np.squeeze(dic_results['xk']))
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
    if dic_results['res']==0:
        print('El algoritmo de Fletcher-Reeves NO CONVERGE')
        print('n = ',n)
        print('f(x0) = ',f(x0))
        if n>=8:
            print('Primer y últimas 4 entradas de xk = ',np.squeeze(dic_results['xk'][:4]),"...",np.squeeze(dic_results['xk'][-4:]))
        else:
            print('xk = ',np.squeeze(dic_results['xk']))
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))    

'''
Ejercicio 3
'''
# Algoritmo general Fletcher Reeves
def hestenes_stiefel(f,g,x0,N,tol,rho):
    a0,c=2.0,1e-4 # Tamaño inicial y factor de proporción fijo para las pruebas
    xk,res,k=x0,0,0
    pk=-g(xk)
    while k<N:
        gk=g(xk)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        fk=f(xk)
        ak=backtracking(f,fk,gk,xk,pk,a0,rho,c)
        xk=xk+ak*pk
        new_gk=g(xk)
        yk=new_gk-gk
        try:
            beta_k=np.squeeze(new_gk.T@yk/(pk.T@yk))
            pk=-new_gk+beta_k*pk
        except RuntimeWarning:
            pk=-new_gk 

        k+=1
        if k>=N:
            res=0
            break
    fk=f(xk)
    gk=g(xk)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}

# Prueba Fletcher-Reeves
def proof_hestenes_stiefel(f,g,x0,N,tol,rho):
    n=len(x0)
    dic_results=hestenes_stiefel(f,g,x0,N,tol,rho)
    if dic_results['res']==1:
        print('El algoritmo de Hestenes-Stiefel CONVERGE')
        print('n = ',n)
        print('f(x0) = ',f(x0))
        if n>8:
            print('Primer y últimas 4 entradas de xk = ',np.squeeze(dic_results['xk'][:4]),"...",np.squeeze(dic_results['xk'][-4:]))
        else:
            print('xk = ',np.squeeze(dic_results['xk']))
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
    if dic_results['res']==0:
        print('El algoritmo de Hestenes-Stiefel NO CONVERGE')
        print('n = ',n)
        print('f(x0) = ',f(x0))
        if n>=8:
            print('Primer y últimas 4 entradas de xk = ',np.squeeze(dic_results['xk'][:4]),"...",np.squeeze(dic_results['xk'][-4:]))
        else:
            print('xk = ',np.squeeze(dic_results['xk']))
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))    


