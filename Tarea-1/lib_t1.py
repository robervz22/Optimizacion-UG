# En este archivo se definen las funciones para los problemas de la tarea 1
import numpy as np
'''
Ejercicio 1
'''
# Funciones a utilizar 
'''
Primer función
'''
def f1(x,paramf=None):
    return x**3 - 2*x +1
def der1_f1(x,paramf=None):
    return 3*x**2 - 2
def der2_f1(x,paramf=None):
    return 6*x
'''
Segunda función
'''
def f2(x,paramf=None):
    return 1 + x - 3*x**2/2 + x**3/6 + x**4/4
def der1_f2(x,paramf=None):
    return 1 - 3*x + x**2/2 + x**3
def der2_f2(x,paramf=None):
    return -3 + x + 3*x**2

# Implementación del método de Newton-Raphson.
def NewtonRaphson(x0, fnc, derf,iterMax,tol,paramf=None):
    xk  = x0
    res = 0
    for i in range(iterMax):
        fk = fnc(xk, paramf)
        if fk<tol and fk>-tol:
            res = 1
            break
        else:
            dfx = derf(xk, paramf)
            if dfx!=0:
                xk = xk - fk/dfx
            else:
                res = -1
                break
    if i==iterMax-1:
        res=0 # No se encontró la raíz en el numero de iteraciones proporcionado
    return {'x0':x0,'f(x0)':fnc(x0,paramf),'Raiz':xk,'f(Raiz)':fk,'Iteraciones':i,'Estado':res}

# Algoritmo de Halley
def Halley(x0,fnc,der1_f,der2_f,iterMax,tol,paramf=None):
    xk = x0
    res= 0
    for i in range(iterMax):
        fk = fnc(xk, paramf)
        if fk<tol and fk>-tol:
            res = 1 # Se obtuvo con éxito la raíz
            break
        dfx=der1_f(xk,paramf)
        if dfx!=0:
            d2fx=der2_f(xk,paramf)
            aux= dfx-(d2fx*fk)/(2.0*dfx)
            if aux!=0:
                xk=xk-fk/aux
            else:
                res=-1 # Hubo un problema con el denominador 
                break
        else:
            res=-1 # Hubo un problema con la derivada
            break
    if i==iterMax-1:
        res=0 # No se encontró la raíz en el número de iteraciones proporcionado
    return {'x0':x0,'f(x0)':fnc(x0,paramf),'Raiz':xk,'f(Raiz)':fk,'Iteraciones':i,'Estado':res}

# Definimos una función que nos permita imprimir cada caso según 'res'
def print_results(dic_results):
    if dic_results['Estado']==1:
        print('El algoritmo converge y los resultados son: ')
        for key, value in dic_results.items():
            print(key, ' : ', value)
        return
    if dic_results['Estado']==0:
        print('El algortimo no converge')
        return
    if dic_results['Estado']==-1:
        print('Hubo un problema al evaluar expresiones (denominadores 0)')
        return

'''
Ejercicio 2
'''
# Polinomio de grado 2n
def pol_taylor_cos(x,n):
    aprox,sumand=np.ones(len(x)),np.ones(len(x))
    for i in range(1,n+1):
        sumand=-(sumand*x**2)/(2.0*i*(2.0*i-1.0))
        aprox+=sumand
    return aprox 


'''
Prueba
'''

'''
# Ejercicio 1
iterMax=100
tol=1e-8
x0=[-1000.0,1000.0]
print('\n')
dic_f1_NR=NewtonRaphson(x0[1],f1,der1_f1,iterMax,tol)
print_results(dic_f1_NR)
print('\n')
dic_f1_H=Halley(x0[1],f1,der1_f1,der2_f1,iterMax,tol)
print_results(dic_f1_H)
print('\n')
dic_f2_NR=NewtonRaphson(x0[1],f2,der1_f2,iterMax,tol)
print_results(dic_f2_NR)
print('\n')
dic_f2_H=Halley(x0[1],f2,der1_f2,der2_f2,iterMax,tol)
print_results(dic_f2_H)

# Ejercicio 2
values_even=np.array([2.0*mt.pi,8.0*mt.pi,12.0*mt.pi])
values_odd=np.array([3.0*mt.pi,9.0*mt.pi,13.0*mt.pi])


n=[10,50,100,200]
np.set_printoptions(precision=4)
print("\n cos(2*PI,8*PI,12*PI)")
print(pol_taylor_cos(values_even,n[0]))
print("\n cos(3*PI,9*PI,13*PI)")
print(pol_taylor_cos(values_odd,n[0]))
'''
