import numpy as np
import matplotlib.pyplot as plt
from pytest import param
import scipy.linalg as linalg

N_BACKTRACKING=1000

'''
Ejercicio 1
'''
# Función tridiagonal generalizada
def tri_diag_gen(x):
    x_squeezed=np.squeeze(x)
    arr1=(x_squeezed[:-1]+x_squeezed[1:]-3.0)**2
    arr2=(x_squeezed[:-1]-x_squeezed[1:]+1.0)**4
    sum1=np.sum(arr1)
    sum2=np.sum(arr2)
    return sum1+sum2

# Gradiente función cuadrática
def grad_tri_diag_gen(x):
    x_squeezed=np.squeeze(x)
    # Primer entrada
    aux1=2.0*(x_squeezed[0]+x_squeezed[1]-3.0)+4.0*(x_squeezed[0]-x_squeezed[1]+1.0)**3
    # De la segunda a la n-2
    xi=x_squeezed[1:-1]
    xi_minus_1=x_squeezed[:-2]
    xi_plus_1=x_squeezed[2:]
    tmp1=2.0*(xi_minus_1+xi-3.0)-4.0*(xi_minus_1-xi+1.0)**3
    tmp2=2.0*(xi+xi_plus_1-3.0)+4.0*(xi-xi_plus_1+1.0)**3
    aux2=tmp1+tmp2
    # Ultima entrada
    aux3=2.0*(x_squeezed[-2]+x_squeezed[-1]-3.0)-4.0*(x_squeezed[-2]-x_squeezed[-1]+1.0)**3

    return np.concatenate(([aux1],aux2,[aux3])).reshape(-1,1)

# Rosenbrock general
def gen_rosenbrock(x):
    x_squeezed=np.squeeze(x)
    arr1=(x_squeezed[1:]-x_squeezed[:-1]**2)**2
    arr2=(1.0-x_squeezed[:-1])**2
    sum1=100.0*np.sum(arr1)
    sum2=np.sum(arr2)
    return sum1+sum2

# Gradiente Rosenbrock general
def grad_gen_rosenbrock(x):
    x_squeezed=np.squeeze(x)
    # Primer entrada
    aux1=-400.0*(x_squeezed[1]-x_squeezed[0]**2)*x_squeezed[0]-2.0*(1.0-x_squeezed[0])
    # De la segunda a la n-2
    xi=x_squeezed[1:-1]
    xi_minus_1=x_squeezed[:-2]
    xi_plus_1=x_squeezed[2:]
    aux2=200.0*(xi-xi_minus_1**2)-400.0*(xi_plus_1-xi**2)*xi-2.0*(1.0-xi)
    # Ultima entrada
    aux3=200.0*(x_squeezed[-1]-x_squeezed[-2]**2)

    return np.concatenate(([aux1],aux2,[aux3])).reshape(-1,1)

'''
Ejercicio 2
'''
# Algoritmo backtracking para tamaño de paso  
def backtracking(f,fk,gk,xk,pk,a0,rho,c):
    a=a0
    k=0
    while f(xk+a*pk)>fk+c*a*(gk.T@pk).item() and k<N_BACKTRACKING:
        a=rho*a
        k+=1
    return a
# Método BFGS modificado
def mod_BFGS(f,g,x0,H0,N,tol,rho):
    n=len(x0)
    Id=np.eye(n)
    a0,c=2.0,1e-4 # Para backtracking
    xk,res,k=x0.reshape(-1,1),0,0
    Hk=H0
    while k<N:
        gk=g(xk)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        pk=-Hk@gk
        tmp1=(pk.T@gk).item()
        if tmp1>0.0:
            lamb1=1e-5+tmp1/(gk.T@gk).item()
            Hk=Hk+lamb1*Id
            pk=pk-lamb1*gk
        fk=f(xk)
        ak=backtracking(f,fk,gk,xk,pk,a0,rho,c)
        new_xk=xk+ak*pk
        new_gk=g(new_xk)
        # Actualizacion Hk
        sk=new_xk-xk
        yk=new_gk-gk
        xk=new_xk # Actualizacion del paso
        tmp2=(yk.T@sk).item()
        if tmp2<=0:
            lamb2=1e-5-tmp2/((sk.T@sk).item()+1e-5)
            Hk=Hk+lamb2*Id
        else:
            rho_k=1.0/tmp2
            '''
            # Valor sugerido para H0
            if k==0:
                Hk=(tmp2/np.asscalar(yk.T@yk))*Id
            '''
            aux1=Id-rho_k*(sk@yk.T)
            aux2=rho_k*(sk@sk.T)
            Hk=(aux1@Hk)@aux1+aux2
        k+=1
        if k>=N:
            res=0
            break
    fk=f(xk)
    gk=g(xk)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}
# Prueba BFGS modificado
def proof_mod_BFGS(f,g,x0,H0,N,tol,rho):
    n=len(x0)
    dic_results=mod_BFGS(f,g,x0,H0,N,tol,rho)
    if dic_results['res']==1:
        print('El algoritmo BFGS modificado CONVERGE')
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
        print('El algoritmo BFGS modificado NO CONVERGE')
        print('n = ',n)
        print('f(x0) = ',f(x0))
        if n>=8:
            print('Primer y últimas 4 entradas de xk = ',np.squeeze(dic_results['xk'][:4]),"...",np.squeeze(dic_results['xk'][-4:]))
        else:
            print('xk = ',np.squeeze(dic_results['xk']))
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))    


