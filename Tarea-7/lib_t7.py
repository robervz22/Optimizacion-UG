import numpy as np
import matplotlib.pyplot as plt
from pytest import param
import scipy.linalg as linalg


'''
Ejercicio 1
'''
# Gauss-Newton Mínimos Cuadrados No Lineales
def gauss_newton_nlls(R,J,z0,N,tol,paramf):
    res=0 # Si se queda en ese valor el algoritmo no converge
    zk,k=z0,0
    while k<N:
        Rk=R(zk,paramf)
        Jk=J(zk,paramf)
        A=Jk.T@Jk
        g=Jk.T@Rk
        pk=np.linalg.solve(A,-g)
        if np.linalg.norm(pk)<tol:
            res=1 # Converge
            break
        k+=1
        zk=zk+pk
    return {'zk':zk,'Rk':Rk,'k':k,'|pk|':np.linalg.norm(pk),'res':res}

# Residual ejemplo 
def R(z,paramf):
    x,y=paramf.T
    value=z[0]*np.sin(z[1]*x+z[2])-y
    return value

# Matriz Jacobiana ejemplo
def J(z,paramf):
    x,y=paramf.T
    jac=np.empty((len(x),3))
    jac[:,0]=np.sin(z[1]*x+z[2])
    jac[:,1]=z[0]*x*np.cos(z[1]*x+z[2])
    jac[:,2]=z[0]*np.cos(z[1]*x+z[2])
    return jac

# Prueba del algoritmo Gauss-Newton
def proof_gauss_newton_nlls(R,J,z0,N,tol,paramf):
    dic_results=gauss_newton_nlls(R,J,z0,N,tol,paramf)
    if dic_results['res']==0:
        zk=dic_results['zk']
        Rk=dic_results['Rk']
        norm_pk=dic_results['|pk|']
        print('El algoritmo de Gauss-Newton NO CONVERGE')
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
        Rk=dic_results['Rk']
        norm_pk=dic_results['|pk|']
        print('El algoritmo de Gauss-Newton CONVERGE')
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
        y_model=zk[0]*np.sin(zk[1]*x_linspace+zk[2])
        plt.style.use('seaborn')
        plt.plot(x,y,'ro',label='muestra')
        plt.plot(x_linspace,y_model,'b-',label='estimación')
        plt.title('Gauss-Newton\n'+r'$f(\mathbf{z}_k)$=%.4f'%(fk))
        plt.legend()
        plt.show()

'''
Ejercicio 2
'''
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
        pk=np.linalg.solve(A+mu*np.eye(A.shape[0]),-g)
        if np.linalg.norm(pk)<tol:
            res=1
            break
        '''
        k+1 data
        '''
        k+=1
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
        y_model=zk[0]*np.sin(zk[1]*x_linspace+zk[2])
        plt.style.use('seaborn')
        plt.plot(x,y,'ro',label='muestra')
        plt.plot(x_linspace,y_model,'b-',label='estimación')
        plt.title('Levenberg-Marquardt\n'+r'$f(\mathbf{z}_k)$=%.4f'%(fk))
        plt.legend()
        plt.show()