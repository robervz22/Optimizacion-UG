from math import dist
import numpy as np
import os

'''
Ejercicio 1
'''
# Descenso Máximo para funciones cuadráticas
def grad_max_quadratic(A,b,x0,N,tol):
    xk,k=x0,0
    while k<N:
        gk=A@xk-b
        if np.linalg.norm(gk)<tol:
            res=1
            break
        pk=-gk
        ak=np.squeeze(-(gk.T@pk)/(pk.T@A@pk))
        xk=xk+ak*pk
        k+=1
        if k>=N:
            res=0
            break
    fk=np.squeeze((xk.T@A@xk)/2.0-b.T@xk)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}

# Función para pruebas
def proof_grad_max_quadratic(data_A,data_b,x0,N,tol):
    owd=os.getcwd()
    A=np.load(owd+data_A)
    b=np.load(owd+data_b).reshape(-1,1) # Lo consideramos como vector columna
    '''
    Checar propiedades de la matriz A
    '''
    print('El número de filas de la matriz A es %d'%(A.shape[0]))
    dist_sym=np.linalg.norm(A-A.T)
    print('||A-A.T||= ',dist_sym)
    if dist_sym<tol:
        print('Es una matriz simétrica')
        eig_min=np.min(np.real_if_close(np.linalg.eigvals(A)))
        print('El valor propio más pequeño es: ',eig_min)
        if eig_min>0:
            print('La matriz A es positiva definida')
            x0=x0.reshape(-1,1) # vector columna desde el principio
            dic_results=grad_max_quadratic(A,b,x0,N,tol)
            if dic_results['res']==1:
                print('El método de descenso máximo con paso exacto CONVERGE')
                print('k = ',dic_results['k'])
                print('fk = ',dic_results['fk'])
                print('||gk|| = ',np.linalg.norm(dic_results['gk']))
                xk=np.squeeze(dic_results['xk'])
                if xk.shape[0]>=3:
                    print('Primeras 3 coordenadas de xk son: ',xk[:3])
                    print('Últimas 3 coordenadas de xk son: ',xk[-3:])
                else:
                    print('xk = ',xk)
                # Punto crítico
                x_ast=np.squeeze(np.linalg.solve(A,b))
                print('||xk-x*|| = ',np.linalg.norm(xk-x_ast))
            elif dic_results['res']==0:
                print('El método de descenso máximo con paso exacto NO CONVERGE')
                print('k = ',dic_results['k'])
                print('fk = ',dic_results['fk'])
                print('||gk|| = ',np.linalg.norm(dic_results['gk']))
                xk=np.squeeze(dic_results['xk'])
                if xk.shape[0]>=3:
                    print('Primeras 3 coordenadas de xk',xk[:3])
                    print('Últimas 3 coordenadas de xk',xk[-3:])
                else:
                    print('xk = ',xk)
                # Punto crítico
                x_ast=np.squeeze(np.linalg.solve(A,b))
                print('||xk-x*|| = ',np.linalg.norm(xk-x_ast))
        else:
            print('La matriz A no es positiva definida, no es posible hacer el método de descenso con paso exacto')
    else:
        print('La matriz no es simétrica, no es posible hacer el método de descenso máximo con paso exacto')
        return 

'''
Ejercicio 2
''' 
# Funcion de Rosenbrock
def f_Rosenbrock(x):
    x_squeezed=x.copy()
    x_squeezed=np.squeeze(x_squeezed)
    x1,x2=x_squeezed.T
    return 100.0*(x2-x1**2)**2+(1.0-x1)**2

# Gradiente de Rosenbrock
def grad_Rosenbrock(x):
    x_squeezed=x.copy()
    x_squeezed=np.squeeze(x_squeezed)
    x1,x2=x_squeezed.T
    g1=400.0*(x1**2-x2)*x1+2.0*(x1-1.0)
    g2=200.0*(x2-x1**2)
    return np.array([g1,g2]).reshape((-1,1)) # vector columna

# Algoritmo backtracking para tamaño de paso  
def backtracking(f,fk,gk,xk,pk,a0,rho,c):
    a=a0
    while f(xk+a*pk)>fk+c*a*np.squeeze(gk.T@pk):
        a=rho*a
    return a

# Descenso Máximo con backtracking
def grad_max(f,g,x0,N,tol,a0,rho,c):
    xk,k=x0,0
    while k<N:
        gk=g(xk)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        pk=-gk
        fk=f(xk)
        ak=backtracking(f,fk,gk,xk,pk,a0,rho,c)
        xk=xk+ak*pk
        k+=1
        if k>=N:
            res=0
            break
    fk=f(xk)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}

# Funciones para pruebas
def proof_grad_max(f,g,x0,N,tol,rho):
    a0,c=2.0,1e-4 # Tamaño inicial y factor de proporción fijo para las pruebas
    x0=x0.reshape(-1,1) # vector columna desde el principio
    dic_results=grad_max(f,g,x0,N,tol,a0,rho,c)
    if dic_results['res']==1:
        print('El algoritmo de descenso máximo con backtracking CONVERGE')
        print('k = ',dic_results['k'])
        print('xk = ',np.squeeze(dic_results['xk']))
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
    if dic_results['res']==0:
        print('El algoritmo de descenso máximo con backtracking NO CONVERGE')
        print('k = ',dic_results['k'])
        print('xk = ',np.squeeze(dic_results['xk']))
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))    


    


    
    