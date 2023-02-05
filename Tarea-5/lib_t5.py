import numpy as np

'''
Ejercicio 1
'''
# Función cuadrática
def quad_fun(x,args):
    A=args[0]
    b=args[1]
    return (x.T@A@x)/2.0-b.T@x

# Gradiente función cuadrática
def grad_quad_fun(x,args):
    A=args[0]
    b=args[1]
    return A@x-b

# Descenso máximo paso fijo
def grad_max_fix(f,g,x0,N,tol,a,args):
    xk,k=x0,0
    while k<N:
        gk=g(xk,args)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        pk=-gk
        xk=xk+a*pk
        k+=1
        if k>=N:
            res=0
            break
    gk=g(xk,args)
    fk=np.squeeze(f(xk,args))
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}

# Prueba paso fijo para funciones cuadráticas
def proof_grad_max_fix_quad(f,g,x0,N,tol,a,args):
    dic_results=grad_max_fix(f,g,x0,N,tol,a,args)
    # Punto crítico
    x_ast=np.squeeze(np.linalg.solve(args[0],args[1]))
    if dic_results['res']==1:
        print('res = ',dic_results['res'])
        print('El método de descenso máximo con paso fijo CONVERGE')
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
        xk=np.squeeze(dic_results['xk'])
        print('xk = ',xk)
        print('||xk-x*|| = ',np.linalg.norm(xk-x_ast))
    elif dic_results['res']==0:
        print('res = ',dic_results['res'])
        print('El método de descenso máximo con paso exacto NO CONVERGE')
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
        xk=np.squeeze(dic_results['xk'])
        print('xk = ',xk)
        print('||xk-x*|| = ',np.linalg.norm(xk-x_ast))

'''
Ejercicio 2
''' 
# Funcion de Rosenbrock
def f_Rosenbrock(x,args=None):
    x_squeezed=x.copy()
    x_squeezed=np.squeeze(x_squeezed)
    x1,x2=x_squeezed.T
    return 100.0*(x2-x1**2)**2+(1.0-x1)**2

# Gradiente de Rosenbrock
def grad_Rosenbrock(x,args=None):
    x_squeezed=x.copy()
    x_squeezed=np.squeeze(x_squeezed)
    x1,x2=x_squeezed.T
    g1=400.0*(x1**2-x2)*x1+2.0*(x1-1.0)
    g2=200.0*(x2-x1**2)
    return np.array([g1,g2]).reshape((-1,1)) # vector columna

# Prueba paso fijo para funciones cuadráticas
def proof_grad_max_fix(f,g,x0,N,tol,a,args):
    dic_results=grad_max_fix(f,g,x0,N,tol,a,args)
    # Punto crítico
    if dic_results['res']==1:
        print('res = ',dic_results['res'])
        print('El método de descenso máximo con paso fijo CONVERGE')
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
        xk=np.squeeze(dic_results['xk'])
        print('xk = ',xk)
    elif dic_results['res']==0:
        print('res = ',dic_results['res'])
        print('El método de descenso máximo con paso exacto NO CONVERGE')
        print('k = ',dic_results['k'])
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
        xk=np.squeeze(dic_results['xk'])
        print('xk = ',xk)
