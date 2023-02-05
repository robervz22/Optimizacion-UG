import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor,LinAlgError,cho_solve
import sys
'''
Ejercicio 1
'''
ITER_MAX=10000 # Iteraciones máximas para backtracking

# Funcion 
def f_eje1(x):
    t1=np.sum(x**2)
    aux=np.arange(1,len(x)+1,1)
    s=0.5*np.sum(aux*x)
    t2=s**2
    t3=s**4
    value=t1+t2+t3
    return value

# Gradiente
def grad_f_eje1(x):
    aux=np.arange(1,len(x)+1,1)
    s=0.5*np.sum(aux*x)
    grad=2.0*x+s*aux+2.0*s**3*aux
    return grad

# Hessiana
def hess_f_eje1(x):
    aux=np.arange(1,len(x)+1,1)
    s=0.5*np.sum(aux*x)
    constant=0.5+3*s**2
    v1=aux.reshape(-1,1)
    hess=constant*v1@v1.T
    hess[np.diag_indices_from(hess)]+=2.0
    return hess

# Algoritmo backtracking para tamaño de paso  
def backtracking(f,fk,gk,xk,pk,a0,rho,c):
    a=a0
    i=0
    while f(xk+a*pk)>fk+c*a*np.squeeze(gk.T@pk) and i <ITER_MAX:
        a=rho*a
        i+=1
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

# Funcion para pruebas
def proof_grad_max(f,g,x0,N,tol,rho):
    a0,c=2.0,1e-4 # Tamaño inicial y factor de proporción fijo para las pruebas
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

'''
Ejercicio 2
'''
# Residual ejemplo 
def R(z,paramf):
    x,y=paramf.T
    value=z[0]-z[1]*np.exp(-np.exp(z[2]+z[3]*np.log(x)))-y
    return value

# Matriz Jacobiana ejemplo
def J(z,paramf):
    x,y=paramf.T
    jac=np.empty((len(x),4))

    aux1=np.exp(-np.exp(z[2]+z[3]*np.log(x)))
    aux2=aux1*np.exp(z[2]+z[3]*np.log(x))

    jac[:,0]=np.ones(len(x))
    jac[:,1]=-aux1
    jac[:,2]=z[1]*aux2
    jac[:,3]=z[1]*aux2*np.log(x)

    return jac

# Levenberg-Marquart Mínimos Cuadrados No lineales
def levenberg_marquardt_nlls(R,J,z0,N,tol,mu_ref,paramf):
    res=0 # Si se queda en ese valor el algoritmo no converge
    zk,k=z0,0
    Rk_old=R(z0,paramf)
    Jk_old=J(z0,paramf)
    fk_old=0.5*Rk_old.T@Rk_old
    A, g = Jk_old.T@Jk_old, Jk_old.T@Rk_old
    mu=np.min([mu_ref,np.max(np.diag(A))])
    while k<N:
        try:
            pk=np.linalg.solve(A+mu*np.eye(A.shape[0]),-g)
            if np.linalg.norm(pk)<tol:
                res=1
                break
            '''
            k+1 data
            '''
            k+=1
            zk_old=zk
            zk=zk+pk
            Rk_new=R(zk,paramf)
            fk_new=0.5*Rk_new.T@Rk_new
            '''
            parametro rho
            '''
            denominator=np.squeeze(-0.5*pk.T@g+0.5*mu*pk.T@pk)
            rho=(fk_new-fk_old)/denominator
            if rho<0.25:
                mu=2.0*mu
            elif rho>0.75:
                mu=mu/3.0
            Jk=J(zk,paramf)
            A, g = Jk.T@Jk, Jk.T@Rk_new
        except (LinAlgError,OverflowError,RuntimeError,RuntimeWarning):
            print(f'Proceso Fallo')
            '''
            Datos antes del fallo
            '''
            Rk_old=R(zk_old,paramf)
            Jk_old=J(zk_old,paramf)
            fk_old=0.5*Rk_old.T@Rk_old
            A, g = Jk_old.T@Jk_old, Jk_old.T@Rk_old
            pk_old=np.linalg.solve(A+mu*np.eye(A.shape[0]),-g)
            return {'zk':zk_old,'fk':fk_old,'k':k,'|pk|':np.linalg.norm(pk_old),'res':res}
            pass

    return {'zk':zk,'fk':fk_new,'k':k,'|pk|':np.linalg.norm(pk),'res':res}

# Prueba del algoritmo de Levenberg-Marquart
def proof_levenberg_marquardt_nlls(R,J,z0,N,tol,mu_ref,paramf):
    dic_results=levenberg_marquardt_nlls(R,J,z0,N,tol,mu_ref,paramf)
    if dic_results['res']==0:
        zk=dic_results['zk']
        Rk=R(zk,paramf)
        norm_pk=dic_results['|pk|']
        print('El algoritmo de Levenberg-Marquardt NO CONVERGE')
        print('z0 = ',z0)
        R0=R(z0,paramf)
        f0=0.5*R0.T@R0
        print('f(z0) = ',f0)
        print('zk = ',zk)
        fk=0.5*Rk.T@Rk
        print('f(zk) = ',fk)
        print('|pk| = ',norm_pk)
        print('k = ',dic_results['k'])
    else:
        zk=dic_results['zk']
        Rk=R(zk,paramf)
        norm_pk=dic_results['|pk|']
        print('El algoritmo de Levenberg-Marquardt CONVERGE')
        print('z0 = ',z0)
        R0=R(z0,paramf)
        f0=0.5*R0.T@R0
        print('f(z0) = ',f0)
        print('zk = ',zk)
        fk=0.5*Rk.T@Rk
        print('f(zk) = ',fk)
        print('|pk| = ',norm_pk)
        print('k = ',dic_results['k'])
        '''
        Gráfica del ajuste obtenido
        '''
        x,y=paramf.T
        x_linspace=np.linspace(np.min(x),np.max(x),num=1000)
        y_model_k=zk[0]-zk[1]*np.exp(-np.exp(zk[2]+zk[3]*np.log(x_linspace)))
        y_model_ini=z0[0]-z0[1]*np.exp(-np.exp(z0[2]+z0[3]*np.log(x_linspace)))

        plt.style.use('seaborn')
        plt.plot(x,y,'ro',label='muestra')
        plt.plot(x_linspace,y_model_k,'b-',label='punto final')
        plt.plot(x_linspace,y_model_ini,'g-',label='punto inicial')
        plt.title('Levenberg-Marquardt\n'+r'$f(\mathbf{z}_k)$=%.4f'%(fk))
        plt.legend()
        plt.show()

'''
Ejercicio 3
'''
# Funcion f
def f_eje3(x):
    x1,x2=x.T
    return 2.0*(x1**2+x2**2-1.0)-x1

# Funcion g
def g_eje3(x):
    x1,x2=x.T
    return (x1**2+x2**2-1.0)**2

# Funcion objetivo
def f_opt_eje3(x,mu):
    return f_eje3(x)+g_eje3(x)*mu/2.0

# Gradiente
def grad_f_opt_eje3(x,mu):
    x1,x2=x.T
    grad=np.empty(2)
    grad[0]=4.0*x1+2*mu*(x1**2+x2**2-1.0)*x1-1.0
    grad[1]=4.0*x2+2*mu*(x1**2+x2**2-1.0)*x2
    return grad

# Método de Newton con paso fijo considerando la penalizacion mu
def newton_fix_step_eje3(f,g,H,x0,N,tol,mu):
    trajectory=[x0]
    xk,k=x0,0
    while k<N:
        gk=g(xk,mu)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        Hk=H(xk,mu)
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
    gk=g(xk,mu)
    fk=np.squeeze(f(xk,mu))
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res},np.array(trajectory)

# Algoritmo backtracking para tamaño de paso considerando la penalización mu
def backtracking_eje3(f,fk,gk,xk,pk,a0,rho,c,mu):
    a=a0
    i=0
    while f(xk+a*pk,mu)>fk+c*a*np.squeeze(gk.T@pk) and i <ITER_MAX:
        a=rho*a
        i+=1
    return a

# Descenso Máximo con backtracking con la penalización mu
def grad_max_eje3(f,g,x0,N,tol,a0,rho,c,mu):
    xk,k=x0,0
    while k<N:
        gk=g(xk,mu)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        pk=-gk
        fk=f(xk,mu)
        ak=backtracking_eje3(f,fk,gk,xk,pk,a0,rho,c,mu)
        xk=xk+ak*pk
        k+=1
        if k>=N:
            res=0
            break
    fk=f(xk,mu)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}