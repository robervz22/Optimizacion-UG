from re import X
import numpy as np

'''
Ejercicio 1
'''
# Positividad variables
def positive_cond(tol,xsol):
    ii_neg=np.squeeze(np.where(xsol<0))
    Ex=np.sum(np.abs(xsol[ii_neg]))
    if Ex<tol:
        print('Se cumplen la condicion de no negatividad')
    else:
        print('NO se cumple la condicion de no negatividad')

# Restricciones
def restriction_cond(tol,xsol,A,b):
    aux_vec=b-A@xsol
    ii_neg=np.squeeze(np.where(aux_vec<0))
    Ex=np.sum(np.abs(aux_vec[ii_neg]))
    if Ex<tol:
        print('Se cumple las restricciones de desigualdad')
    else:
        print('NO se cumplen las restricciones de desigualdad')

'''
Ejercicio 3
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


        