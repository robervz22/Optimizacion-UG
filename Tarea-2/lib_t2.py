# En este archivo se definen las funciones para los problemas de la tarea 2
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import os
'''
Ejercicio 1
'''
def poly_roots_plot(coef_):
    roots=np.roots(coef_)
    re_roots=[]
    for r in roots:
        if not np.iscomplex(r):
            re_roots.append(r)
    re_roots=np.array(re_roots,dtype=np.float)
    r_min,r_max=np.min(re_roots),np.max(re_roots)
    x=np.linspace(r_min-1.0,r_max+1,num=100)
    y=np.polyval(coef_,x)
    polr=np.polyval(coef_,re_roots)
    #fig, ax = plt.subplots(1,1,figsize=(6,6))
    #ax.plot(x,y,color='b',linestyle='--')
    #ax.scatter(re_roots,polr,color='r',marker='^')
    #ax.axhline(y=0.0,color='k')
    gg=ggplot()+geom_line(aes(x=x,y=y),color='blue')+geom_point(aes(x=re_roots,y=polr),color='red',alpha=0.5,size=3)+geom_abline(slope=0,intercept=0.0,color='black')
    return roots,re_roots,gg

'''
Ejercicio 2
'''
def poly_solve(xy_array,n):
    A=np.vander(xy_array[:,0],n+1)
    c=np.linalg.solve(A.T@A,A.T@xy_array[:,1])
    cod_num=np.linalg.cond(A.T@A)
    return c,cod_num

def poly_reg(data_dir,n,r):
    owd=os.getcwd()
    xy_array=np.load(owd+data_dir)
    c,cod_num=poly_solve(xy_array,n)
    x_min,x_max=np.min(xy_array[:,0]),np.max(xy_array[:,0])
    z=np.linspace(x_min,x_max,num=r)
    pol_c=np.polyval(c,z)
    gg=ggplot()+geom_line(aes(x=z,y=pol_c),color='blue')+geom_point(aes(x=xy_array[:,0],y=xy_array[:,1]),color='red',alpha=0.5)
    return c,cod_num,gg
    