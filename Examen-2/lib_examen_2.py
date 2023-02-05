import numpy as np
import matplotlib.pyplot as plt
from pytest import param
import scipy.linalg as linalg

N_BACKTRACKING=1000


'''
Ejercicio 1
'''
# Funcion Ejemplo F
def eje1_f(x):
    return 2.0*x[0]**2+x[1]**2-x[0]*x[1]-6.5*x[0]+2.5*x[1]

# Funcion barrera
def barrera_f(x,mu):
    x_squeezed=np.squeeze(x)
    aux=-(np.log(1.0-x_squeezed[0])+np.log(1.0+x_squeezed[0])+np.log(1.0-x_squeezed[1])+np.log(1.0+x_squeezed[1]))
    return (eje1_f(x_squeezed)+aux/mu)

# Gradiente F
def grad_eje1_f(x):
    par1=4.0*x[0]-x[1]-6.5
    par2=2.0*x[1]-x[0]+2.5
    return np.array([par1,par2]).reshape(-1,1)

# Gradiente barrera
def grad_barrera_f(x,mu):
    x_squeezed=np.squeeze(x)
    par1=4.0*x_squeezed[0]-x_squeezed[1]-6.5
    par2=2.0*x_squeezed[1]-x_squeezed[0]+2.5
    aux1=(1.0/(1.0-x_squeezed[0])-1.0/(1.0+x_squeezed[0]))/mu
    aux2=(1.0/(1.0-x_squeezed[1])-1.0/(1.0+x_squeezed[1]))/mu
    return np.array([par1+aux1,par2+aux2]).reshape(-1,1)

# Algoritmo backtracking para tamaño de paso  
def backtracking(f,fk,gk,xk,pk,a0,rho,c,mu):
    a=a0
    k=0
    aux1=f(xk+a*pk,mu)
    aux2=fk+c*a*(gk.T@pk).item()
    while (aux1>aux2 and k<N_BACKTRACKING) or aux1!=aux1:
        a=rho*a
        k+=1
        aux1=f(xk+a*pk,mu)
        aux2=fk+c*a*(gk.T@pk).item()
    return a
# Método BFGS modificado
def mod_BFGS(f,g,x0,H0,N,tol,rho,mu):
    n=len(x0)
    Id=np.eye(n)
    a0,c=1.0,1e-4 # Para backtracking
    xk,res,k=x0.reshape(-1,1),0,0
    Hk=H0
    while k<N:
        gk=g(xk,mu)
        if np.linalg.norm(gk)<tol:
            res=1
            break
        pk=-Hk@gk
        tmp1=(pk.T@gk).item()
        if tmp1>0.0:
            lamb1=1e-5+tmp1/(gk.T@gk).item()
            Hk=Hk+lamb1*Id
            pk=pk-lamb1*gk
        fk=f(xk,mu)
        ak=backtracking(f,fk,gk,xk,pk,a0,rho,c,mu)
        new_xk=xk+ak*pk
        new_gk=g(new_xk,mu)
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
    fk=f(xk,mu)
    gk=g(xk,mu)
    return {'xk':xk,'fk':fk,'gk':gk,'k':k,'res':res}
# Prueba método de barrera
def proof_barrera(f,g,x0,mu0,N_BARRERA,N_BFGS,tol_BARRERA,tol_BFGS,rho,eje1_f):
    k=0
    xk=x0
    muk=mu0
    while k<N_BARRERA:
        dic_results_BGFS=mod_BFGS(f,g,xk,np.eye(len(x0)),N_BFGS,tol_BFGS,rho,muk)
        if dic_results_BGFS['res']==1:
            '''
            print('k = ',k)
            print('mu_k = ',muk)
            print('iteraciones BFGS = ',dic_results_BGFS['k'])
            print('xk = ',dic_results_BGFS['xk'])
            print('fk = ',eje1_f(dic_results_BGFS['xk']))
            '''
            xk_new=dic_results_BGFS['xk']
            # Checo la tolerancia
            if np.linalg.norm(xk_new-xk)<tol_BARRERA:
                print('k = ',k)
                print('mu_k = ',muk)
                print('iteraciones BFGS = ',dic_results_BGFS['k'])
                print('xk = ',xk_new)
                print('fk = ',eje1_f(xk_new))
                return xk_new
            # No se cumple la tolerancia pero estoy en la región factible
            else:
                xk=xk_new
                muk=10*muk
                k+=1
        # No converge el algoritmo
        if dic_results_BGFS['res']==0:
            xk=dic_results_BGFS['xk']
            muk=10*muk
            k+=1
        if k>=N_BARRERA:
            print('El método barrera NO CONVERGE')

'''
Ejercicio 2
'''
# Condiciones KKT
def KKT_cond(tol,b,c,x,lamb,s,A):
    # Condicion 1
    aux_vec1=A.T@lamb+s
    print('Condicion 1: |AT*lamb+s-c| =',np.linalg.norm(aux_vec1-c))
    # Condicion 2
    aux_vec2=A@x 
    print('Condicion 2: |Ax-b| = ',np.linalg.norm(aux_vec2-b))
    # Condicion 3
    ii_neg_x=np.squeeze(np.where(x<0))
    Ex=np.sum(np.abs(x[ii_neg_x]))
    if Ex<tol:
        print('SI se cumple la condicion de no negatividad de x')
    else:
        print('NO se cumple la condicion de no negatividad de x')
    # Condicion 4
    ii_neg_s=np.squeeze(np.where(s<0))
    Es=np.sum(np.abs(s[ii_neg_s]))
    if Es<tol:
        print('SI se cumple la condicion de no negatividad de s')
    else:
        print('NO se cumple la condicion de no negatividad de s')
    # Condicion 5
    aux_vec3=np.abs(x*s)
    if np.sum(aux_vec3)<tol:
        print('SI se cumple la condicion de complentariedad')
    else:
        print('NO se cumple la condicion de complentariedad')
