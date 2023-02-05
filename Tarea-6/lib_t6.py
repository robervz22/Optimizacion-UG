from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor,LinAlgError,cho_solve

# Funcion de Rosenbrock para graficar
def f_Rosenbrock_graph(x1,x2):
    return 100.0*(x2-x1**2)**2+(1.0-x1)**2

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

# Hessiana de Rosenbrock
def hess_Rosenbrock(x):
    x_squeezed=x.copy()
    x_squeezed=np.squeeze(x_squeezed)
    x1,x2=x_squeezed.T
    g11=1200.0*x1**2+2-400.0*x2
    g22=200.0
    g12=-400.0*x1
    return np.array([[g11,g12],[g12,g22]]) # matrix de 2x2

'''
Ejercicio 1
'''
# Anota un punto
def annotate_pt(text,xy,xytext,color):
    plt.plot(xy[0],xy[1],marker='P',markersize=10,c=color)
    plt.annotate(text,xy=xy,xytext=xytext,
                 # color=color,
                 arrowprops=dict(arrowstyle="->",
                 color = color,
                 connectionstyle='arc3'))

# Método de Newton con paso fijo
def newton_fix_step(f,g,H,x0,N,tol):
    trajectory=[x0]
    xk,k=x0,0
    while k<N:
        gk=g(xk)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        Hk=H(xk)
        try:
            L,lower=cho_factor(Hk)
            pk=cho_solve((L,lower),-gk)
            xk=xk+pk
            trajectory.append(xk)
            k+=1
        except LinAlgError:
            res=0
            print('No se pudo realizar la factorización de Cholesky')
            break
        if k>=N:
            res=0
            break
    gk=g(xk)
    fk=np.squeeze(f(xk))
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res},np.array(trajectory)

# Prueba del Newton con paso fijo, dibujando trayectorias
def proof_newton_fix_step_Rosenbrock(f,g,H,x0,N,tol,f_graph):
    dic_results,trajectory=newton_fix_step(f,g,H,x0,N,tol)
    if dic_results['res']==1:
        '''
        Convergencia y último punto de la trayectoria
        '''
        print('El método de Newton con paso fijo CONVERGE')
        print('El punto final es: ')
        print('xk = ',np.squeeze(dic_results['xk']))
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
        print('k = ',dic_results['k'])
        '''
        Filtración de la trayectoria
        '''
        x_ast=np.array([1.0,1.0])
        dist_max=np.linalg.norm(np.array([-1.5,-1.0])-x_ast)
        trajectory_filter=np.array([x for x in trajectory if np.linalg.norm(x-x_ast)<=dist_max])
        '''
        Contornos de nivel
        '''
        x,y=np.arange(-1.5,1.5,0.1),np.arange(-1.0,2.0,0.1)
        X,Y=np.meshgrid(x,y)
        Z=f_graph(X,Y) # La función para graficar tiene input diferente
        levels=[1.0,10.0,50.0,100.0]
        fig, ax = plt.subplots(figsize=(10,8))
        CS = ax.contour(X, Y, Z, levels, colors = ["#91A3E1", "#83B692","#F9ADA0", "#CD6FD5"],linestyles='dashed')
        ax.axhline(0, color='black', alpha=.5, dashes=[2, 4],linewidth=1)
        ax.axvline(0, color='black', alpha=0.5, dashes=[2, 4],linewidth=1)
        ax.clabel(CS,inline=1,fontsize=10)
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')
        ax.set_title('Contornos de nivel\n Función de Rosenbrock')
        '''
        Parte de la trayectoria
        '''
        ax.plot(trajectory_filter[:,0],trajectory_filter[:,1],marker='o',c='firebrick')
        # Annotate the point found at last iteration
        annotate_pt('Mínimo Encontrado',
                (trajectory_filter[-1,0],trajectory_filter[-1,1]),
                (1.5,0.0),'green')
        iter = trajectory_filter.shape[0]
        for w,i in zip(trajectory_filter,range(iter-1)):
            # Annotate with arrows to show history
            plt.annotate("",
                        xy=w, xycoords='data',
                        xytext=trajectory_filter[i+1,:], textcoords='data',
                        arrowprops=dict(arrowstyle='<|-',mutation_scale=20,
                        connectionstyle='arc3,rad=0.'))     
    if dic_results['res']==0:
        print('El método de Newton con paso fijo NO CONVERGE')

'''
Ejercicio 2
'''
# Algoritmo backtracking para tamaño de paso  
def backtracking(f,fk,gk,xk,pk,a0,rho,c):
    a=a0
    while f(xk+a*pk)>fk+c*a*np.squeeze(gk.T@pk):
        a=rho*a
    return a

# Método de Newton con backtracking
def newton_backtracking(f,g,H,x0,N,tol,a0,rho,c):
    trajectory=[x0]
    xk,k=x0,0
    while k<N:
        gk=g(xk)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        Hk=H(xk)
        try:
            L,lower=cho_factor(Hk)
            pk=cho_solve((L,lower),-gk)
            # Backtracking
            fk=f(xk)
            ak=backtracking(f,fk,gk,xk,pk,a0,rho,c)
            xk=xk+ak*pk
            trajectory.append(xk)
            k+=1
        except LinAlgError:
            res=0
            print('No se pudo realizar la factorización de Cholesky')
            break
        if k>=N:
            res=0
            break
    gk=g(xk)
    fk=np.squeeze(f(xk))
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res},np.array(trajectory)

# Prueba del Newton con backtracking, dibujando trayectorias
def proof_newton_backtracking_Rosenbrock(f,g,H,x0,N,tol,a0,rho,c,f_graph):
    dic_results,trajectory=newton_backtracking(f,g,H,x0,N,tol,a0,rho,c)
    if dic_results['res']==1:
        '''
        Convergencia y último punto de la trayectoria
        '''
        print('El método de Newton con paso fijo CONVERGE')
        print('El punto final es: ')
        print('xk = ',np.squeeze(dic_results['xk']))
        print('fk = ',dic_results['fk'])
        print('||gk|| = ',np.linalg.norm(dic_results['gk']))
        print('k = ',dic_results['k'])
        '''
        Filtración de la trayectoria
        '''
        x_ast=np.array([1.0,1.0])
        dist_max=np.linalg.norm(np.array([-1.5,-1.0])-x_ast)
        trajectory_filter=np.array([x for x in trajectory if np.linalg.norm(x-x_ast)<=dist_max])
        '''
        Contornos de nivel
        '''
        x,y=np.arange(-1.5,1.5,0.1),np.arange(-1.0,2.0,0.1)
        X,Y=np.meshgrid(x,y)
        Z=f_graph(X,Y) # La función para graficar tiene input diferente
        levels=[1.0,10.0,50.0,100.0]
        fig, ax = plt.subplots(figsize=(10,8))
        CS = ax.contour(X, Y, Z, levels, colors = ["#91A3E1", "#83B692","#F9ADA0", "#CD6FD5"],linestyles='dashed')
        ax.axhline(0, color='black', alpha=.5, dashes=[2, 4],linewidth=1)
        ax.axvline(0, color='black', alpha=0.5, dashes=[2, 4],linewidth=1)
        ax.clabel(CS,inline=1,fontsize=10)
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')
        ax.set_title('Contornos de nivel\n Función de Rosenbrock')
        '''
        Parte de la trayectoria
        '''
        ax.plot(trajectory_filter[:,0],trajectory_filter[:,1],marker='o',c='firebrick')
        # Annotate the point found at last iteration
        annotate_pt('Mínimo Encontrado',
                (trajectory_filter[-1,0],trajectory_filter[-1,1]),
                (1.5,0.0),'green')
        iter = trajectory_filter.shape[0]
        for w,i in zip(trajectory_filter,range(iter-1)):
            # Annotate with arrows to show history
            plt.annotate("",
                        xy=w, xycoords='data',
                        xytext=trajectory_filter[i+1,:], textcoords='data',
                        arrowprops=dict(arrowstyle='<|-',mutation_scale=20,
                        connectionstyle='arc3,rad=0.'))     
    if dic_results['res']==0:
        print('El método de Newton con paso fijo NO CONVERGE')        

