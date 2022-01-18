#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:13:00 2020

@author: francesca
"""


#function are written for case with 5 tenants


import numpy
from gurobipy import *

##### to save matrix




#def salvamat(matrice, nomefile):
 #   fo=open(nomefile, 'w')
   # for riga in matrice:
  #      riga=[str(val) for val in riga]
    #    fo.write('\t'.join(riga)+'\n')
   # fo.close()
    
    #######centralized3
def drf3(data, R):
 
    n=5
    nbcont=5
    nris=3
#nbvar=15

    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  

# Second membre
    sec= R

    
    

    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)   
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)

#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c=numpy.zeros(n)
    c[0]=1

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        m.addConstr(x[i] >= 0)#"c%d" % nris+i) 
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
        
    x_f=numpy.zeros((5,1))   
    x_f=x
    
    return(x_f)
    
###########centralized
def drf(data, R):
 
    n=5
    nbcont=5
    nris=2
#nbvar=15

    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  

# Second membre
    sec= R

    
    

    ds=numpy.zeros(n)
    ds_user=numpy.zeros(2)   
    for i in range(0,n):
        for j in range(0,2):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)

#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c=numpy.zeros(n)
    c[0]=1

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        m.addConstr(x[i] >= 0)#"c%d" % nris+i) 
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
        
    x_f=numpy.zeros((5,1))   
    x_f=x
    
    return(x_f)
    
    
  ##################### cra
  
def cra_p_d(D,R):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocprop(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    a=numpy.zeros((5))
    a2=numpy.zeros((5))
    for j in range(0,5):
        a[j]=D[j,1]*x1[j]
        a2[j]=D[j,2]*x1[j]
        
    if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
        x2=drf_up(D[:,1:3],R[1:3],x1)
    else:
        x2=x1
    x=x2
    return(x)




def cra_mmf_d(D,R):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmmf(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    a=numpy.zeros((5))
    a2=numpy.zeros((5))
    for j in range(0,5):
        a[j]=D[j,1]*x1[j]
        a2[j]=D[j,2]*x1[j]
        
    if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
        x2=drf_up(D[:,1:3],R[1:3],x1)
    else:
        x2=x1
    x=x2
    return(x)
    
def cra_mood_d(D,R):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmood(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    a=numpy.zeros((5))
    a2=numpy.zeros((5))
    for j in range(0,5):
        a[j]=D[j,1]*x1[j]
        a2[j]=D[j,2]*x1[j]
        
    if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
        x2=drf_up(D[:,1:3],R[1:3],x1)
    else:
        x2=x1
    x=x2
    return(x)
    
######ocra

def ocra_p_d(D,R):
    ordi=ordine(D,R)##se 1 ho cra se 2  inverto
    cont_reallo=0
    if ordi==1:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if perc[0,:]>1:         
            x1=allocprop(D[:,0],R[0])
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        a2=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,1]*x1[j]
            a2[j]=D[j,2]*x1[j]
            
        if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
            x2=drf_up(D[:,1:3],R[1:3],x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2
    else:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if max(perc[1,:], perc[2,:])>1:         
            x1=drf(D[:,1:3],R[1:3])
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,0]*x1[j]
        if  perc[0,:]>1 and (sum(a)>R[0]): 
            x2=prop_up(D[:,0],R[0],x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2 
    return(x, cont_reallo)

    
def ocra_mmf_d(D,R):
    ordi=ordine(D,R)##se 1 ho cra se 2  inverto
    cont_reallo=0
    if ordi==1:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if perc[0,:]>1:         
            x1=allocmmf(D[:,0],R[0])
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        a2=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,1]*x1[j]
            a2[j]=D[j,2]*x1[j]
            
        if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
            x2=drf_up(D[:,1:3],R[1:3],x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2
    else:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if max(perc[1,:], perc[2,:])>1:         
            x1=drf(D[:,1:3],R[1:3])
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,0]*x1[j]
        if  perc[0,:]>1 and (sum(a)>R[0]): 
            x2=mmf_up(D[:,0],R[0],x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2 
    return(x, cont_reallo)
    
def ocra_mood_d(D,R):
    ordi=ordine(D,R)##se 1 ho cra se 2  inverto
    cont_reallo=0
    if ordi==1:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if perc[0,:]>1:         
            x1=allocmood(D[:,0],R[0])
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        a2=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,1]*x1[j]
            a2[j]=D[j,2]*x1[j]
            
        if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
            x2=drf_up(D[:,1:3],R[1:3],x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2
    else:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if max(perc[1,:], perc[2,:])>1:         
            x1=drf(D[:,1:3],R[1:3])
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,0]*x1[j]
        if  perc[0,:]>1 and (sum(a)>R[0]): 
            x2=mood_up(D[:,0],R[0],x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2 
    return(x, cont_reallo)
    

##########pra
    
def pra_p_d(D,R):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocprop(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    if max( perc[1,:],perc[2,:])>1: 
        x2=drf(D[:,1:3],R[1:3])
    else:
        x2=[1,1,1,1,1]
    x=[min(x1[0],x2[0]), min(x1[1],x2[1]),min(x1[2],x2[2]),min(x1[3],x2[3]),min(x1[4],x2[4])]
    return(x)


def pra_mmf_d(D,R):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmmf(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    if max( perc[1,:],perc[2,:])>1: 
        x2=drf(D[:,1:3],R[1:3])
    else:
        x2=[1,1,1,1,1]
    x=[min(x1[0],x2[0]), min(x1[1],x2[1]),min(x1[2],x2[2]),min(x1[3],x2[3]),min(x1[4],x2[4])]
    return(x)

def pra_mood_d(D,R):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmood(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    if max( perc[1,:],perc[2,:])>1: 
        x2=drf(D[:,1:3],R[1:3])
    else:
        x2=[1,1,1,1,1]
    x=[min(x1[0],x2[0]), min(x1[1],x2[1]),min(x1[2],x2[2]),min(x1[3],x2[3]),min(x1[4],x2[4])]
    return(x)


####pra2
    
def pra2_p_d2(D,R):
    
    cl1=numpy.zeros((1,1))
    cl2=numpy.zeros((1,1))
    cl3=numpy.zeros((1,1))
    
    cl1=sum(D[:,0])/R[0]
    cl2=sum(D[:,1])/R[1]
    cl3=sum(D[:,2])/R[2]

    
    if cl1>cl2 and cl1>cl3:
        clmax=numpy.zeros((1,1))
        clmax=max(cl1,cl2,cl3)
    
        x=[1.0/clmax, 1.0/clmax,1.0/clmax,1.0/clmax,1.0/clmax]
    
    else:
    
        x=drf3(D,R)

    return(x)
    
def pra2_mmf_d2(D,R):

    cl1=numpy.zeros((1,1))
    cl2=numpy.zeros((1,1))
    cl3=numpy.zeros((1,1))
    
    cl1=sum(D[:,0])/R[0]
    cl2=sum(D[:,1])/R[1]
    cl3=sum(D[:,2])/R[2]

    
    if cl1>cl2 and cl1>cl3:
        
    
        x=drf3(D,R)
    
    else:
        x=drf3(D,R)

    return(x)
    
    
def pra2_mood_d2(D,R):
    cl1=numpy.zeros((1,1))
    cl2=numpy.zeros((1,1))
    cl3=numpy.zeros((1,1))
    
    cl1=sum(D[:,0])/R[0]
    cl2=sum(D[:,1])/R[1]
    cl3=sum(D[:,2])/R[2]
    
    if cl1>cl2 and cl1>cl3:
                
        x=sol_ps_mood(D,R)
    
    
    else:
        x=drf3(D,R)

    return(x)


####useful functions
    
    
    
def sol_ps_mood(data, R):
    
   
    nbcont=5
    n=5
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  
# Second membre
    sec= R
    
    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)
    p=numpy.zeros(n)
    div=numpy.zeros(n)
    minimo=numpy.zeros(n)
    massimo=numpy.zeros(n)
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)
        p[i]=list(ds_user).index(ds[i])

    for i in range(0,n):
        minimo[i]=max(sec[int(p[i])]-sum (data[:,int(p[i])])+data[i,int(p[i])],0)
        massimo[i]=min(sec[int(p[i])],data[i,int(p[i])])
        div[i]=massimo[i]-minimo[i]
    print(div)

#
#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c = [1,0,0,0,0]

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS, name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
        for j in lignes:
            if massimo[j]==minimo[j]:
                m.addConstr(r[i]-b[n*i+j]-data[j,int(p[j])]*x[j] <= 0, "c%d" % (n+n*i+j))
            else:
                    m.addConstr(r[i]-b[n*i+j]-(data[j,int(p[j])]*x[j]-minimo[j])/(float(div[j])) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        if massimo[i]==minimo[i]:
            
            m.addConstr(data[i,int(p[i])]*x[i] <= 1 )  
        else:
                
            m.addConstr((data[i,int(p[i])]*x[i]-minimo[i])/(float(div[i])) <= 1 )    
    
             
  
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
       # print ('x%d'%(j+1), '=', x[j].x)
        x[j]=x[j].x
    return (x)    
    
    
def mood_up(D,R,x_max):
    a=numpy.zeros((5))
    x=numpy.zeros((5))
    d_min=numpy.zeros((5))
    d_max=numpy.zeros((5))
    d_new=numpy.zeros((5))
    d_min=dmin(D,R)
    d_max=dmax(D,R)
    for k in range(0,5):
        d_new[k]=d_max[k]-d_min[k]
    R_new=R-sum(d_min)
    for k in range(0,5): 
        a[k]=d_min[k]+(R_new*d_new[k])/sum(d_new)
    for k in range(0,5): 
        x[k]=a[k]/D[k]

    x_f=numpy.zeros((5))
    x_f=x
    x_c=numpy.zeros((6,1))
    x_c=controllo(x_f,x_max)
    
    
    
    while x_c[5,0]==1:
        
        s=0
        
        D_ok=numpy.zeros((0,1))
        x_m=numpy.zeros((0,1))
        R_ok=numpy.zeros((1,1))
        rus=numpy.zeros((0,1))
        for i in range(0,5):
            if x_c[i]==0:
                D_ok= numpy.vstack([D_ok,D[i]])
                x_m=numpy.vstack([x_m,x_max[i]])
                s=s+1
            else:
                rus=numpy.vstack([rus,x_c[i]*D[i]])
        R_ok=R-sum(rus[:,0])
        n=s
        a=numpy.zeros((n))
        x=numpy.zeros((n))
        d_min=numpy.zeros((n))
        d_max=numpy.zeros((n))
        d_new=numpy.zeros((n))
        d_min=dmin_gen(D_ok,R_ok,n)
        d_max=dmax_gen(D_ok,R_ok,n)
        for k in range(0,n):
            d_new[k]=d_max[k]-d_min[k]
        R_new=R_ok-sum(d_min)
        for k in range(0,n): 
            a[k]=d_min[k]+(R_new*d_new[k])/sum(d_new)
        for k in range(0,n): 
            x[k]=a[k]/D[k]
    
        cont=0
        x_f=numpy.zeros((5))
        for i in range (0,5):
            if x_c[i]==0:
                x_f[i]=x[cont]
                cont=cont+1
            else:
                x_f[i]=x_c[i]
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_max)
        
    return(x_f)
 
def dmin_gen(D,R,n):
    dm=numpy.zeros((n))
    for i in range(0,n):
        dm[i]=max(R-sum(D)+D[i],0)
    return(dm)
    
def dmax_gen(D,R,n):
    dM=numpy.zeros((n))
    for i in range(0,n):
        if D[i]>R:
            dM[i]=R
        else:
            dM[i]=D[i] 
    return(dM)    
    
    
    
def mmf_up(D,R,x_max):
    n=5
    x=numpy.zeros((5))
    a_f=numpy.zeros((5))
    x_f=numpy.zeros((5))
    dom=numpy.sort(D)
    pos=sorted(range(len(D)), key=lambda k: D[k])
    if dom[0]<(R/n):
        x[0]=dom[0]    
        if dom[1]<(R-dom[0])/(n-1):
            x[1]=dom[1] 
            if dom[2]<(R-dom[0]-dom[1])/(n-2):
                x[2]=dom[2] 
                if dom[3]<(R-dom[0]-dom[1]-dom[2])/(n-3):
                    x[3]=dom[3] 
                    if dom[4]<(R-dom[0]-dom[1]-dom[2]-dom[3])/(n-4):
                        x[4]=dom[4] 
                    else:
                        x=[dom[0],dom[1],dom[2],dom[3],(R-dom[0]-dom[1]-dom[2]-dom[3])/(n-4)];
                else:
                    x=[dom[0],dom[1],dom[2],(R-dom[0]-dom[1]-dom[2])/(n-3),(R-dom[0]-dom[1]-dom[2])/(n-3)];
            else:
                x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2)];       
        else:
            x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
                              
    else:
        x=[R/n,R/n,R/n,R/n,R/n]
    for i in range(0,5):
        a_f[pos[i]]=x[i]
    for i in range(0,5):
        x_f[i]=a_f[i]/D[i]
   
    x_c=numpy.zeros((6,1))
    x_c=controllo(x_f,x_max)
     
    while x_c[5,0]==1:
        Dnew=numpy.zeros(((x_c[0:5,0] == 0).sum()))
        contat=0
        for lun in range (0,5):
            if x_c[lun,0] == 0:
                Dnew[contat]=D[lun]
                contat=contat+1
        x=numpy.zeros(5)
        if ((x_c[0:5,0] == 0).sum())==1:
            x=[max(Dnew,R)]
        if ((x_c[0:5,0] == 0).sum())==2:  
            x=mmf2(Dnew,R)
        if ((x_c[0:5,0] == 0).sum())==3:
            x=mmf3(Dnew,R)
        if ((x_c[0:5,0] == 0).sum())==4:  
            x=mmf4(Dnew,R)
            
        if ((x_c[0:5,0] == 0).sum())==5:
            x=allocmood(Dnew,R)
        cont=0
        x_f=numpy.zeros((5))
        for i in range (0,5):
            if x_c[i]==0:
                x_f[i]=x[cont]
                cont=cont+1
            else:
                x_f[i]=x_c[i]
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_max)
    
    
    
    return(x_f)    
    
def mmf2(D,R):
    n=2
    x=numpy.zeros((n))
    a_f=numpy.zeros((n))
    x_f=numpy.zeros((n))
    dom=numpy.sort(D)
    pos=sorted(range(len(D)), key=lambda k: D[k])
    if dom[0]<(R/n):
        x[0]=dom[0]    
        if dom[1]<(R-dom[0])/(n-1):
            x[1]=dom[1] 
        else:
            x=[dom[0],(R-dom[0])/(n-1)];  
    else:
        x=[R/n,R/n]
    for i in range(0,n):
        a_f[pos[i]]=x[i]
    for i in range(0,n):
        x_f[i]=a_f[i]/D[i]
    return(x_f)


    
def mmf4(D,R):
    n=4
    x=numpy.zeros((n))
    a_f=numpy.zeros((n))
    x_f=numpy.zeros((n))
    dom=numpy.sort(D)
    pos=sorted(range(len(D)), key=lambda k: D[k])
    if dom[0]<(R/n):
        x[0]=dom[0]    
        if dom[1]<(R-dom[0])/(n-1):
            x[1]=dom[1] 
            if dom[2]<(R-dom[0]-dom[1])/(n-2):
                x[2]=dom[2] 
                if dom[3]<(R-dom[0]-dom[1]-dom[2])/(n-3):
                    x[3]=dom[3] 
                else:
                    x=[dom[0],dom[1],dom[2],(R-dom[0]-dom[1]-dom[2])/(n-3)];
            else:
                x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2)];       
        else:
            x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
    else:
        x=[R/n,R/n,R/n,R/n]
    for i in range(0,n):
        a_f[pos[i]]=x[i]
    for i in range(0,n):
        x_f[i]=a_f[i]/D[i]
    return(x_f)

def mmf3(D,R):
    n=3
    x=numpy.zeros((n))
    a_f=numpy.zeros((n))
    x_f=numpy.zeros((n))
    dom=numpy.sort(D)
    pos=sorted(range(len(D)), key=lambda k: D[k])
    if dom[0]<(R/n):
        x[0]=dom[0]    
        if dom[1]<(R-dom[0])/(n-1):
            x[1]=dom[1] 
            if dom[2]<(R-dom[0]-dom[1])/(n-2):
                x[2]=dom[2] 
            else:
                x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2)];       
        else:
            x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
    else:
        x=[R/n,R/n,R/n]
    for i in range(0,n):
        a_f[pos[i]]=x[i]
    for i in range(0,n):
        x_f[i]=a_f[i]/D[i]
    return(x_f)

    
    
    
def prop_up(D,R,x_max):
    x=numpy.zeros((5))
    for k in range(0,5): 
        x[k]=R/sum(D)
    x_f=numpy.zeros((5))
    x_f=x
    x_c=numpy.zeros((6,1))
    x_c=controllo(x,x_max)
    
    while x_c[5,0]==1:
        
        s=0
        
        D_ok=numpy.zeros((0,1))
        x_m=numpy.zeros((0,1))
        R_ok=numpy.zeros((1,1))
        rus=numpy.zeros((0,1))
        for i in range(0,5):
            if x_c[i]==0:
                D_ok= numpy.vstack([D_ok,D[i]])
                x_m=numpy.vstack([x_m,x_max[i]])
                s=s+1
            else:
                rus=numpy.vstack([rus,x_c[i]*D[i]])
        R_ok=R-sum(rus[:,0])
        n=s
        
        for k in range(0,n): 
            x[k]=R_ok/sum(D_ok)
        
        cont=0
        x_f=numpy.zeros((5))
        for i in range (0,5):
            if x_c[i]==0:
                x_f[i]=x[cont]
                cont=cont+1
            else:
                x_f[i]=x_c[i]
                
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_max)
                   
        
    
    return(x_f)   
    
    
    
    
def ordine(D,R):
    perc=numpy.zeros((3))
    for k in range(0,3): 
        perc[k]=sum(D[:,k])/R[k] 
    p2=perc[2]
    p1=max(perc[0],perc[1])
    if p1>p2:
        x=1
    else:
        x=2
    return(x)
    
    
    
    
    
def dmin(D,R):
    dm=numpy.zeros((5))
    for i in range(0,5):
        dm[i]=max(R-sum(D)+D[i],0)
    return(dm)
    
def dmax(D,R):
    dM=numpy.zeros((5))
    for i in range(0,5):
        if D[i]>R:
            dM[i]=R
        else:
            dM[i]=D[i]
    return(dM)
    
    
    
def allocmood(D,R):
    a=numpy.zeros((5))
    x=numpy.zeros((5))
    d_min=numpy.zeros((5))
    d_max=numpy.zeros((5))
    d_new=numpy.zeros((5))
    d_min=dmin(D,R)
    d_max=dmax(D,R)
    for k in range(0,5):
        d_new[k]=d_max[k]-d_min[k]
    R_new=R-sum(d_min)
    for k in range(0,5): 
        a[k]=d_min[k]+(R_new*d_new[k])/sum(d_new)
    for k in range(0,5): 
        x[k]=a[k]/D[k]
    return(x)
    
    
def allocmmf(D,R):
    n=5
    x=numpy.zeros((5))
    a_f=numpy.zeros((5))
    x_f=numpy.zeros((5))
    dom=numpy.sort(D)
    pos=sorted(range(len(D)), key=lambda k: D[k])
    if dom[0]<(R/n):
        x[0]=dom[0]    
        if dom[1]<(R-dom[0])/(n-1):
            x[1]=dom[1] 
            if dom[2]<(R-dom[0]-dom[1])/(n-2):
                x[2]=dom[2] 
                if dom[3]<(R-dom[0]-dom[1]-dom[2])/(n-3):
                    x[3]=dom[3] 
                    if dom[4]<(R-dom[0]-dom[1]-dom[2]-dom[3])/(n-4):
                        x[4]=dom[4] 
                    else:
                        x=[dom[0],dom[1],dom[2],dom[3],(R-dom[0]-dom[1]-dom[2]-dom[3])/(n-4)];
                else:
                    x=[dom[0],dom[1],dom[2],(R-dom[0]-dom[1]-dom[2])/(n-3),(R-dom[0]-dom[1]-dom[2])/(n-3)];
            else:
                x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2)];       
        else:
            x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
                              
    else:
        x=[R/n,R/n,R/n,R/n,R/n]
    for i in range(0,5):
        a_f[pos[i]]=x[i]
    for i in range(0,5):
        x_f[i]=a_f[i]/D[i]
    return(x_f)   
    
    
    
    
def allocprop(D,R):
    x=numpy.zeros((5,1))
    for k in range(0,5): 
        x[k,:]=R/sum(D)
    return(x)
    
    
    
def drf_up(data, R,x_max):
 
    n=5
    nbcont=5
    nris=2
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  

# Second membre
    sec= R

    
    

    ds=numpy.zeros(n)
    ds_user=numpy.zeros(2)   
    for i in range(0,n):
        for j in range(0,2):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)

#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c=numpy.zeros(n)
    c[0]=1

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        m.addConstr(x[i] >= 0)#"c%d" % nris+i) 
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
        
    x_f=numpy.zeros((5,1))   
    x_f=x
        
    x_c=numpy.zeros((6,1))
    x_c=controllo(x_f,x_max)
    
    while x_c[5,0]==1:
        
        s=0
        
        D_ok=numpy.zeros((0,2))
        x_m=numpy.zeros((0,1))
        R_ok=numpy.zeros((1,2))
        rus=numpy.zeros((0,2))
        for i in range(0,5):
            if x_c[i]==0:
                D_ok= numpy.vstack([D_ok,data[i,:]])
                x_m=numpy.vstack([x_m,x_max[i]])
                s=s+1
            else:
                rus=numpy.vstack([rus,x_c[i]*data[i,:]])
        R_ok[0,0]=R[0]-sum(rus[:,0])
        R_ok[0,1]=R[1]-sum(rus[:,1])
        n=s
        nbcont=s
        nris=2
    #nbvar=15
    
    # Range of plants and warehouses
        lignes = range(nbcont)
    #colonnes = range(nbvar)
    
    
    # Matrice des contraintes
        a =D_ok
      
    
    # Second membre
        sec= R_ok
    
        
    
        ds=numpy.zeros(n)
        ds_user=numpy.zeros(2)   
        for i in range(0,n):
            for j in range(0,2):
                ds_user[j]=D_ok[i,j]/sec[0,j]
            ds[i]=max(ds_user)
    
    #w1=1
    #w2=1
    #w3=1
    
    # Coefficients de la fonction objectif
        c=numpy.zeros(n)
        c[0]=1
    
        m = Model("mogpl5ex")     
            
    # declaration variables de decision
        x = []
    
        for i in range(0,n):
            x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))
    
    
        
    ### r e b
        r=[]
    
        for i in range(0,n):
            r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))
    
        b=[]
        for j in range(0,n):
            for i in range(0,n):
                b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))
    
    
    
    # maj du modele pour integrer les nouvelles variables
        m.update()
    
        obj = LinExpr();
        obj = 0
        obj += c[0] *( r[0]-sum( b[0:n] ))
            
    # definition de l'objectif
        m.setObjective(obj,GRB.MAXIMIZE)
    
    # Definition des contraintes
        for i in range(0,nris):
            m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[0,i], "c%d" % i)
        for i in lignes:
                for j in lignes:
                    m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
        for i in range(0,n):
            m.addConstr(x[i] >= 0)#"c%d" % nris+i) 
            
            
            
    # Resolution
        m.optimize()
    
    
    #    print ("")                
    #    print ('Solution optimale:')
        for j in lignes:
            x[j]=x[j].x
    
        cont=0
        x_f=numpy.zeros((5,1))
        for i in range (0,5):
            if x_c[i]==0:
                x_f[i,0]=x[cont]
                cont=cont+1
            else:
                x_f[i]=x_c[i]
                
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_max)
                
    
    return (x_f)    


def controllo(x,xmax):
    x1=numpy.zeros((6,1))
    for i in range(0,5):     
        if x[i]<xmax[i]:
            x1[i,0]=0 
        else:
            if x[i]==xmax[i]:
                x1[i,0]=xmax[i]
            else:
                x1[i,0]=xmax[i]
                x1[5,0]=1
    return(x1)
    
    
    
    
    
    
    
    

    
#########all min
    
def all_min(D,D_m):
    all_m=numpy.zeros((5,3))
    minim=numpy.zeros((5,1))
    for k in range(0,5): 
        if any(D[k,j] for j in range(0,3))==0:         
            minim[k]==numpy.zeros((1,3))
        else:
            minim[k]=max(D_m[k]/D[k])
    all_m=D*minim
    return(all_m)
    
    
    
def infis(a,R):
    x=0
    for i in range(0,3):
        if numpy.sum(a, axis=0)[i]>R[i]:
            x=x+1
        else:
            x=x+0
    if numpy.sum(x)==0:
        return(0)
    else:
        return(1)

#####rand
def pesco_rand(e2):
    
    p = e2[e2[:,0].argsort()]
    return (p)

def pesco_rand1(pr):
    pr1=numpy.random.permutation(pr)
    ind = numpy.argsort( pr1[:,0] ); 
    o= pr1[ind] 
    o_cont=numpy.zeros((5,3));
    for i in range(0,5):
        o_cont[4-i,:]=o[i,:]
    c1=0;
    c2=0;
    c3=0;
    c4=0;
    n1=0;
    n2=0;
    n3=0;
    n4=0;
    
    for i in range(0,5):
        if o_cont[i,0]==1:
            n1=n1+1
        if o_cont[i,0]==2:
            n2=n2+1
        if o_cont[i,0]==3:
            n3=n3+1
        if o_cont[i,0]==4:
            n4=n4+1
    uno=numpy.zeros((n1,3));
    due=numpy.zeros((n2,3));
    tre=numpy.zeros((n3,3));
    qua=numpy.zeros((n4,3));
    for i in range(0,5):
        if o_cont[i,0]==1:
            uno[c1,:]= o_cont[i,:];
            c1=c1+1;
        if o_cont[i,0]==2:
            due[c2,:]= o_cont[i,:];
            c2=c2+1;
        if o_cont[i,0]==3:
            tre[c3,:]= o_cont[i,:];
            c3=c3+1;       
        if o_cont[i,0]==4:
            qua[c4,:]= o_cont[i,:];
            c4=c4+1;            
    uno = uno[uno[:,1].argsort()]
    due = due[due[:,1].argsort()]
    tre = tre[tre[:,1].argsort()]
    qua = qua[qua[:,1].argsort()]
    
    fin=numpy.zeros((5,3))
    fin=numpy.append(qua,tre,  axis=0)
    fin=numpy.append(fin,due,  axis=0)
    fin=numpy.append(fin,uno,  axis=0)
    return(fin)
#    
    
    

##############
def     aggiunta(D_m,D_new_m,D,D_new,R_pb):
    u=numpy.zeros((5,3))
    for i in range(0,5):
        if D_new_m[i,0]!=0:
            d_prob_m=numpy.zeros((5,3))
            d_prob_m[i,:]=D_new_m[i,:]
            D_prob=D_m+d_prob_m
            d_prob=numpy.zeros((5,3))
            d_prob[i,:]=D_new[i,:]
            D_p=D+d_prob
            dom_min=numpy.sum(D_prob,axis=0)         
            u[i,:]=dom_min<=R_pb
            if numpy.sum(u[i,:],axis=0)==3:
                all_m=all_min(D_p,D_prob)
                x_contr=infis(all_m, R_pb)
                if x_contr==0:
                    u[i,:]=numpy.array([1,1,1])
                else: 
                    u[i,:]=numpy.array([0,0,0])
            else:
                u[i,:]=numpy.array([0,0,0]) 
    return(u)
    
def     ordina(pr):
    o=numpy.zeros((5,2));
    for i in range(0,5):
        o[i,:]=pr[4-i,:];
    return(o)


def     ordina1(pr):
    o=numpy.zeros((5,3));
    for i in range(0,5):
        o[i,:]=pr[4-i,:];
    return(o)

def     ordinacontr(pr):
    pr1=numpy.random.permutation(pr)
    ind = numpy.argsort( pr1[:,0] ); 
    o= pr1[ind] 
    o_cont=numpy.zeros((5,2));
    for i in range(0,5):
        o_cont[4-i,:]=o[i,:]
#    c1=0;
#    c2=0;
#    c3=0;
#    c4=0;
#    for i in range(0,5):
#        if o_cont[i,0]==1:
#            uno[c1,:]= o_cont[i,:];
#            c1=c1+1;
#        if o_cont[i,0]==2:
#            due[c2,:]= o_cont[i,:];
#            c2=c2+1;
#        if o_cont[i,0]==3:
#            tre[c3,:]= o_cont[i,:];
#            c3=c3+1;
##    ind1 = np.random.permutation(); 
#    o= pr[ind] 
    return(o_cont)     
    
#################OWA(x)



def sol(data,data_min, R,n):
    
    
   
    nbcont=n
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  
# Second membre
    sec= R
    minim=numpy.zeros((n,1))
    for i in range(0,n): 
        if any(data[i,j] for j in range(0,3))==0:
             minim[i]==numpy.zeros((1,3))
        else:
            minim[i]=min(max(data_min[i]/data[i]),1)

#
#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c = [1,0,0,0,0]

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS, name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:5] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)  
    for i in range(0,n):
        m.addConstr(x[i] >= minim[i])#"c%d" % nris+i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-x[j] <= 0, "c%d" % (n+n*i+j))

                
  
      
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
       #print ('x%d'%(j+1), '=', x[j].x)
      
        x[j]=x[j].x
    return (x)    

########################## OWA(ds x)
    

def sol_ds(data, data_min, R,n):
    
   
    nbcont=n
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  
# Second membre
    sec= R
    minim=numpy.zeros((5,1))
    for i in range(0,n): 
        if any(data[i,j] for j in range(0,3))==0:
             minim[i]==numpy.zeros((1,3))
        else:
            minim[i]=min(max(data_min[i]/data[i]),1)
    
    

    ds=numpy.zeros(5)
    ds_user=numpy.zeros(3)
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)

#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c = [1,0,0,0,0]

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        m.addConstr(x[i] >= minim[i])#"c%d" % nris+i) 
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
    return (x)    


##########################################OWA(ps)

def sol_ps(data, data_min, R,n):
    
   
    nbcont=n
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  
# Second membre
    sec= R
    minim=numpy.zeros((5,1))
    for i in range(0,n): 
        if any(data[i,j] for j in range(0,3))==0:
             minim[i]==numpy.zeros((1,3))
        else:
            minim[i]=min(max(data_min[i]/data[i]),1)

    
    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)
    p=numpy.zeros(n)
    div=numpy.zeros(n)
    minimo=numpy.zeros(n)
    massimo=numpy.zeros(n)
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)
        p[i]=list(ds_user).index(ds[i])

    for i in range(0,n):
        minimo[i]=min(max(sec[int(p[i])]-sum (data[:,int(p[i])])+data[i,int(p[i])],0),data[i,int(p[i])])
        massimo[i]=min(sec[int(p[i])],data[i,int(p[i])])
        div[i]=massimo[i]-minimo[i]
    print(div)
    
    if any(div[i] for i in range(0,n)) <= 0:
        x=numpy.zeros((5))
        for k in range(0,5):
            if ds[k]==0:
                x[k]=0
            else:
                x[k]=1
                
    else:

#
#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
        c = [1,0,0,0,0]
    
        m = Model("mogpl5ex")     
            
    # declaration variables de decision
        x = []
    
        for i in range(0,n):
            x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,name="x%d" % (i+1)))
    
    
        
    ### r e b
        r=[]
    
        for i in range(0,n):
            r.append(m.addVar(vtype=GRB.CONTINUOUS, name="r%d" % (i+1)))
    
        b=[]
        for j in range(0,n):
            for i in range(0,n):
                b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))
    
    
    
    # maj du modele pour integrer les nouvelles variables
        m.update()
    
        obj = LinExpr();
        obj = 0
        obj += c[0] *( r[0]-sum( b[0:n] ))
            
    # definition de l'objectif
        m.setObjective(obj,GRB.MAXIMIZE)
    
    # Definition des contraintes
        for i in range(0,nris):
            m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
        for i in lignes:
                for j in lignes:
                    if float(div[j])!=0:
                        m.addConstr(r[i]-b[n*i+j]-(data[j,int(p[j])]*x[j]-minimo[j])/(float(div[j])) <= 0, "c%d" % (n+n*i+j))
        for i in range(0,n):
            if float(div[i])!=0:
                m.addConstr((data[i,int(p[i])]*x[i]-minimo[i])/(float(div[i]) )<= 1 )    
        for i in range(0,n):
            m.addConstr(x[i] >= minim[i])#"c%d" % nris+i)
                 
      
    # Resolution
        m.optimize()
    
    
    #    print ("")                
    #    print ('Solution optimale:')
        for j in lignes:
           # print ('x%d'%(j+1), '=', x[j].x)
            x[j]=x[j].x

    return (x)    

#######################OWA(ps ds)
def sol_psds(data, data_min, R,n):
    
   
    nbcont=n
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  
# Second membre
    sec= R
    minim=numpy.zeros((5,1))
    for i in range(0,n): 
        if any(data[i,j] for j in range(0,3))==0:
             minim[i]==numpy.zeros((1,3))
        else:
            minim[i]=min(max(data_min[i]/data[i]),1)

    
    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)
    p=numpy.zeros(n)
    div=numpy.zeros(n)
    minimo=numpy.zeros(n)
    massimo=numpy.zeros(n)
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)
        p[i]=list(ds_user).index(ds[i])

    for i in range(0,n):
        
        minimo[i]=min(max(sec[int(p[i])]-sum (data[:,int(p[i])])+data[i,int(p[i])],0),data[i,int(p[i])])
        massimo[i]=min(sec[int(p[i])],data[i,int(p[i])])
        div[i]=massimo[i]-minimo[i]
        
    if any(div[i] for i in range(0,n)) <= 0:
        x=numpy.zeros((5))
        for k in range(0,5):
            if ds[k]==0:
                x[k]=0
            else:
                x[k]=1
                
    else:

        
    #
    #w1=1
    #w2=1
    #w3=1
    
    # Coefficients de la fonction objectif
        c = [1,0,0,0,0]
    
        m = Model("mogpl5ex")     
            
    # declaration variables de decision
        x = []
    
        for i in range(0,n):
            x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,name="x%d" % (i+1)))
    
    
        
    ### r e b
        r=[]
    
        for i in range(0,n):
            r.append(m.addVar(vtype=GRB.CONTINUOUS, name="r%d" % (i+1)))
    
        b=[]
        for j in range(0,n):
            for i in range(0,n):
                b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))
    
    
    
    # maj du modele pour integrer les nouvelles variables
        m.update()
    
        obj = LinExpr();
        obj = 0
        obj += c[0] *( r[0]-sum( b[0:n] ))
            
    # definition de l'objectif
        m.setObjective(obj,GRB.MAXIMIZE)
    
    # Definition des contraintes
       
        
        for i in range(0,nris):
            m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
        for i in lignes:
                for j in lignes:
                        if float(div[j])!=0:
                            m.addConstr(r[i]-b[n*i+j]-ds[j]*(data[j,int(p[j])]*x[j]-minimo[j])/(float(div[j])) <= 0, "c%d" % (n+n*i+j))
        for i in range(0,n):
            if float(div[i])!=0:
                m.addConstr((data[i,int(p[i])]*x[i]-minimo[i])/(float(div[i])) <= 1 )      
        for i in range(0,n):
            m.addConstr(x[i] >= minim[i])#"c%d" % nris+i)
    # Resolution
        m.optimize()
    
    
    #    print ("")                
    #    print ('Solution optimale:')
        for j in lignes:
           # print ('x%d'%(j+1), '=', x[j].x)
            x[j]=x[j].x
    return (x)    








###################################################################

############# per noms

###################################################################
    











def pra2_p_d(D,R):
    perc=numpy.zeros((3))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    share=numpy.zeros((5,3))
    cong=numpy.zeros((2))
    for k in range(0,3): 
        perc[k]=sum(D[:,k])/R[k]  
    if perc[0]>1:         
        x1=allocprop(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    if max( perc[1],perc[2])>1: 
        x2=drf(D[:,1:3],R[1:3])
    else:
        x2=[1,1,1,1,1]
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_p_pra(share, D,R,x1,x2)
    else:
        x=sol_drf_pra(share, D,R,x1,x2)
    return(x)
    
    

    
    
def sol_ps_mood(data, R):
    
   
    nbcont=5
    n=5
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =data
  
# Second membre
    sec= R
    
    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)
    p=numpy.zeros(n)
    div=numpy.zeros(n)
    minimo=numpy.zeros(n)
    massimo=numpy.zeros(n)
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=data[i,j]/sec[j]
        ds[i]=max(ds_user)
        p[i]=list(ds_user).index(ds[i])

    for i in range(0,n):
        minimo[i]=max(sec[int(p[i])]-sum (data[:,int(p[i])])+data[i,int(p[i])],0)
        massimo[i]=min(sec[int(p[i])],data[i,int(p[i])])
        div[i]=massimo[i]-minimo[i]
    print(div)

#
#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c = [1,0,0,0,0]

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS, name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
        for j in lignes:
            if massimo[j]==minimo[j]:
                m.addConstr(r[i]-b[n*i+j]-data[j,int(p[j])]*x[j] <= 0, "c%d" % (n+n*i+j))
            else:
                    m.addConstr(r[i]-b[n*i+j]-(data[j,int(p[j])]*x[j]-minimo[j])/(float(div[j])) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        if massimo[i]==minimo[i]:
            
            m.addConstr(data[i,int(p[i])]*x[i] <= 1 )  
        else:
                
            m.addConstr((data[i,int(p[i])]*x[i]-minimo[i])/(float(div[i])) <= 1 )    
    
             
  
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
       # print ('x%d'%(j+1), '=', x[j].x)
        x[j]=x[j].x
    return (x)    
    
    
    
    
def pra2_mmf_d(D,R):
    perc=numpy.zeros((3))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    share=numpy.zeros((5,3))
    cong=numpy.zeros((2))
    for k in range(0,3): 
        perc[k]=sum(D[:,k])/R[k]  
    if perc[0]>1:         
        x1=allocmmf(D[:,0],R[0])
    else:
        x1=[1,1,1,1,1]
    if max( perc[1],perc[2])>1: 
        x2=drf(D[:,1:3],R[1:3])
    else:
        x2=[1,1,1,1,1]
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_mmf_pra(share, D,R,x1,x2)
    else:
        x=sol_drf_pra(share, D,R,x1,x2)
    return(x)    
    
    

    
    
    
    
def solpar_3(Dnew, R):
        
    n=len(Dnew[:,0])
    nbcont=len(Dnew[:,0])
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =Dnew[:,0:3]
  

# Second membre
    sec= R

    
    

    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)   
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=Dnew[i,j]/sec[j]
        ds[i]=max(ds_user)

#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c=numpy.zeros(n)
    c[0]=1

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        m.addConstr(x[i] >= 0)#"c%d" % nris+i) 
        
               
        
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
        
    x_f=numpy.zeros((n,1))   
    x_f=x
    
    return(x_f)
    
    
    
def solpar(Dnew, R):
        
    n=len(Dnew[:,0])
    nbcont=len(Dnew[:,0])
    nris=2
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =Dnew[:,0:3]
  

# Second membre
    sec= R

    
    

    ds=numpy.zeros(n)
    ds_user=numpy.zeros(2)   
    for i in range(0,n):
        for j in range(0,2):
            ds_user[j]=Dnew[i,j]/sec[j]
        ds[i]=max(ds_user)

#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c=numpy.zeros(n)
    c[0]=1

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)

# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
            for j in lignes:
                m.addConstr(r[i]-b[n*i+j]-(ds[j]*x[j]) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        m.addConstr(x[i] >= 0)#"c%d" % nris+i) 
        
               
        
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
        
    x_f=numpy.zeros((n,1))   
    x_f=x
    
    return(x_f)
    

def sol_p_pra(share, D,R,x1,x2):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    a1=numpy.zeros((len(drf)))
    for i in range(0,len(drf)):
        x[drf[i]]=x2[drf[i]]
        a1[i]=D[drf[i],0]*x[drf[i]]
    Rnew=R[0]-sum(a1)
    Dnew=numpy.zeros((len(p),3))
    for i in range(0,len(p)):
        ind=p[i]
        Dnew[i,:]=D[ind,:] 
    if sum(Dnew[:,0])>Rnew:
        for i in range(0,len(p)):
            x[p[i]]=Rnew/sum(Dnew[:,0])
    else:
         for i in range(0,len(p)):
            x[p[i]]=1
    return(x)
 

def sol_mmf_pra(share, D,R,x1,x2):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    a1=numpy.zeros((len(drf)))
    for i in range(0,len(drf)):
        x[drf[i]]=x2[drf[i]]
        a1[i]=D[drf[i],0]*x[drf[i]]
    Rnew=R[0]-sum(a1)
    Dnew=numpy.zeros((len(p),3))
    for i in range(0,len(p)):
        ind=p[i]
        Dnew[i,:]=D[ind,:] 
    soluz=numpy.zeros((len(p)))
    if sum(Dnew[:,0])>Rnew:
        soluz=solmmf_par(Dnew[:,0], Rnew, len(p))
        for i in range(0,len(p)):
            x[p[i]]=soluz[i]
    else:
        for i in range(0,len(p)):
            x[p[i]]=1

    return(x)


def solmmf_par(D, R,n):
    x=numpy.zeros(n)
    if n==1:
        x=[min(1,R/D)]
    if n==2:
        x=mmf2(D,R)
    if n==3:
        x=mmf3(D,R)
    if n==4:
        x=mmf4(D,R)
    if n==5:
        x=allocmmf(D,R)
    
    return(x)







def sol_mood_pra(share, D,R,x1,x2):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    a1=numpy.zeros((len(drf)))
    for i in range(0,len(drf)):
        x[drf[i]]=x2[drf[i]]
        a1[i]=D[drf[i],0]*x[drf[i]]
    Rnew=R[0]-sum(a1)
    Dnew=numpy.zeros((len(p),3))
    for i in range(0,len(p)):
        ind=p[i]
        Dnew[i,:]=D[ind,:] 
    soluz=numpy.zeros((len(p)))
    if sum(Dnew[:,0])>Rnew:
        soluz=solmood_par(Dnew[:,0], Rnew, len(p))
        for i in range(0,len(p)):
            x[p[i]]=soluz[i]
    else:
         for i in range(0,len(p)):
            x[p[i]]=1
    return(x)
    

    return(x)
 
def solmood_par(D, R,n):
    a=numpy.zeros((n))
    x=numpy.zeros((n))
    d_min=numpy.zeros((n))
    d_max=numpy.zeros((n))
    d_new=numpy.zeros((n))
    if n==1:
        if D>R:
            x[0]=R/D
        else:
            x[0]=1 
    else:
        for i in range(0,n):
            d_min[i]=max(R-sum(D)+D[i],0)
        for i in range(0,n):
            if D[i]>R:
                d_max[i]=R
            else:
                d_max[i]=D[i]   
        for k in range(0,n):
            d_new[k]=d_max[k]-d_min[k]
        R_new=R-sum(d_min)
        for k in range(0,n): 
            a[k]=d_min[k]+(R_new*d_new[k])/sum(d_new)
        for k in range(0,n): 
            x[k]=a[k]/D[k]
    return(x)

    
    
def sol_drf_pra(share, D,R,x1,x2):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    drf=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    agg2=numpy.zeros((1,3))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    Rnew=numpy.zeros((2))
    a1=numpy.zeros((len(p),2))
    for i in range(0,len(p)):
        x[p[i]]=x1[p[i]]
        a1[i,0]=D[p[i],1]*x[p[i]]
        a1[i,1]=D[p[i],2]*x[p[i]]
    Rnew[0]=R[1]-sum(a1[:,0])
    Rnew[1]=R[2]-sum(a1[:,1]) 
    Dnew=numpy.zeros((len(drf),3))
    for i in range(0,len(drf)):
        ind=drf[i]
        Dnew[i,:]=D[ind,:] 
    soluz=numpy.zeros((len(drf)))
    soluz=solpar(Dnew[:,1:3], Rnew)
    for i in range(0,len(drf)):
        x[drf[i]]=soluz[i]
    return(x)
    
    
    
#######################
    
    
    
def  priority_cra_p_d(D,R,pr):
    n=D[:,0].size
    perc=numpy.zeros((3,1))
    x=numpy.zeros((n,1))
    x1=numpy.zeros((n,1))
    x2=numpy.zeros((n))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocprop_pr(D[:,0],R[0],pr)
    else:
        for j in range(0,n):
            x1[j]=1
    a=numpy.zeros((n))
    a2=numpy.zeros((n))
    for j in range(0,n):
        a[j]=D[j,1]*x1[j]
        a2[j]=D[j,2]*x1[j]   
    if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
        x2=drf_up_pr(D[:,1:3],R[1:3],x1,pr)
    else:
        x2=x1
    x=x2
    return(x)
    
def  priority_cra_mood_d(D,R,pr):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmood_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    a=numpy.zeros((5))
    a2=numpy.zeros((5))
    for j in range(0,5):
        a[j]=D[j,1]*x1[j]
        a2[j]=D[j,2]*x1[j]
        
    if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
        x2=drf_up_pr(D[:,1:3],R[1:3],x1,pr)
    else:
        x2=x1
    x=x2
    return(x)
    
def  priority_cra_mmf_d(D,R,pr):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmmf_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    a=numpy.zeros((5))
    a2=numpy.zeros((5))
    for j in range(0,5):
        a[j]=D[j,1]*x1[j]
        a2[j]=D[j,2]*x1[j]
        
    if max( perc[1,:],perc[2,:])>1 and (sum(a)>R[1] or sum(a2)>R[2]): 
        x2=drf_up_pr(D[:,1:3],R[1:3],x1,pr)
    else:
        x2=x1
    x=x2
    return(x)
    
    
def allocmmf_pr(D,R_ok,pr):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((5))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,5):
        inser=[D[j],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            n4=n4+1
    a=numpy.zeros((5))
    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok:
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D1[:,0])
            else:
                soldrf=numpy.zeros((n1))
                soldrf=solmood_par(D1[:,0],R_ok,n1)
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok:
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D2[:,0])
            else:
                soldrf=numpy.zeros((n2))
                soldrf=solmood_par(D2[:,0],R_ok,n2)
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                n3=0
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok:
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D3[:,0]) 
            else:
                soldrf=numpy.zeros((n3))
                soldrf=solmood_par(D3[:,0],R_ok,n3)
                for k in range(0,n3): 
                    soldrf[k]=R_ok/sum(D3[:,0])
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k] 
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok:
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D4[:,0])
            else:
                soldrf=numpy.zeros((n4))
                soldrf=solmood_par(D4[:,0],R_ok,n4)
                for k in range(0,n4): 
                    soldrf[k]=R_ok/sum(D4[:,0])
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                j=4
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
    for j in range(0,5):  
        x[j]=a[j]
    
    return(x)





def allocmood_pr(D,R_ok,pr):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((5))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,5):
        inser=[D[j],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            n4=n4+1
    a=numpy.zeros((5))
    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok:
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D1[:,0])
            else:
                soldrf=numpy.zeros((n1))
                soldrf=solmmf_par(D1[:,0],R_ok,n1)
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok:
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D2[:,0])
            else:
                soldrf=numpy.zeros((n2))
                soldrf=solmmf_par(D2[:,0],R_ok,n2)
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                n3=0
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok:
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D3[:,0]) 
            else:
                soldrf=numpy.zeros((n3))
                soldrf=solmmf_par(D3[:,0],R_ok,n3)
                for k in range(0,n3): 
                    soldrf[k]=R_ok/sum(D3[:,0])
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k] 
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok:
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D4[:,0])
            else:
                soldrf=numpy.zeros((n4))
                soldrf=solmmf_par(D4[:,0],R_ok,n4)
                for k in range(0,n4): 
                    soldrf[k]=R_ok/sum(D4[:,0])
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                j=4
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
    for j in range(0,5):  
        x[j]=a[j]
    
    return(x)




    
    
    
    
def allocprop_pr(D,R_ok,pr):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    n=D.size
    x=numpy.zeros((n))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,n):
        inser=[D[j],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            n4=n4+1
    a=numpy.zeros((n))
    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok:
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D1[:,0])
            else:
                soldrf=numpy.zeros((n1))
                for k in range(0,n1): 
                    soldrf[k]=R_ok/sum(D1[:,0])
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok:
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D2[:,0])
            else:
                soldrf=numpy.zeros((n2))
                for k in range(0,n2): 
                    soldrf[k]=R_ok/sum(D2[:,0])
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                n3=0
                n4=0
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok:
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D3[:,0]) 
            else:
                soldrf=numpy.zeros((n3))
                for k in range(0,n3): 
                    soldrf[k]=R_ok/sum(D3[:,0])
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k] 
                n4=0
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok:
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=R_ok-sum(D4[:,0])
            else:
                soldrf=numpy.zeros((n4))
                for k in range(0,n4): 
                    soldrf[k]=R_ok/sum(D4[:,0])
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                j=4
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
    for j in range(0,n):  
        x[j]=a[j]
    
    return(x)
    
    
def drf_up_pr(D,R_ok,x1,pr):
    D1=numpy.zeros((0,3))
    D2=numpy.zeros((0,3))
    D3=numpy.zeros((0,3))
    D4=numpy.zeros((0,3))
    inser=numpy.zeros((1,3))
    n=D[:,0].size
    x=numpy.zeros((n))
    x1_1=numpy.zeros((0,1))
    x1_2=numpy.zeros((0,1))
    x1_3=numpy.zeros((0,1))
    x1_4=numpy.zeros((0,1))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,n):
        inser=[D[j,0],D[j,1],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            x1_1=numpy.insert(x1_1, n1, x1[j],  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            x1_2=numpy.insert(x1_2, n2, x1[j],  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            x1_3=numpy.insert(x1_3, n3, x1[j],  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            x1_4=numpy.insert(x1_4, n4, x1[j],  axis=0)
            n4=n4+1
            
    a=numpy.zeros((n))
    a_sol=numpy.zeros((n))
    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok[0] and sum(D1[:,1])<R_ok[1]:
                a=alloc1_up_2ris(D1,R_ok,x1,n1)
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n1))
                soldrf=solpar_up(D1,R_ok,x1,n1)
                for k in range(0,n1):
                    pos=D1[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])] 
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok[0] and sum(D2[:,1])<R_ok[1]:
                a_sol=alloc1_up_2ris(D2,R_ok,x1,n2)
                for i in range(0,n):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a_sol[k], D[k,1]*a_sol[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n2))
                soldrf=solpar_up(D2,R_ok,x1,n2)
                for k in range(0,n2):
                    pos=D2[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n3=0
                n4=0
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok[0] and sum(D3[:,1])<R_ok[1]:
                a_sol=alloc1_up_2ris(D3,R_ok,x1,n3)
                for i in range(0,n):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a_sol[k], D[k,1]*a_sol[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n3))
                soldrf=solpar_up(D3,R_ok,x1,n3)
                for k in range(0,n3):
                    pos=D3[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n4=0
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a_sol[k], D[k,1]*a_sol[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok[0] and sum(D4[:,1])<R_ok[1]:
                a_sol=alloc1_up_2ris(D4,R_ok,x1,n4)
                for i in range(0,n):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n4))
                soldrf=solpar_up(D4,R_ok,x1,n4)
                for k in range(0,n4):
                    pos=D4[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
           
    for j in range(0,n):  
        x[j]=a[j]
    
    
    
    return(x)
    
def alloc1_up_2ris(D,R,x_up,n1):
    a=numpy.zeros(5)
    for k in range(0,n1):
        pos=D[k,2]
        pos=pos.astype(int)
        a[pos]=1
    x_c=numpy.zeros((6,1))
    x_c=controllo(a,x_up)
    if x_c[5]==1:
        for i in range(0,n1):
            pos=D[i,2]
            pos=pos.astype(int)
            if x_c[pos]>0:
                a[pos]=x_up[pos]
    return(a)
    
    
def solpar_up(D,R,x1,n):
    x_f=numpy.zeros((5))
    x_sol=numpy.zeros((n))
    x_sol=solpar(D[:,0:2], R)    
    for k in range(0,n):
        pos=D[k,2]
        pos=pos.astype(int)
        x_f[pos]=x_sol[k]

    x_c=numpy.zeros((6,1))
    x_c=controllo(x_f,x1)
    
    while x_c[5,0]==1:
        s=0
        t=0
        D_ok=numpy.zeros((0,3))
        x_m=numpy.zeros((0,1))
        R_ok=numpy.zeros((2))
        rus=numpy.zeros((0,2))
        for i in range(0,n):
            pos=D[i,2]
            pos=pos.astype(int)
            if x_c[pos]==0:
                D_ok=numpy.insert(D_ok, s, D[i,:],  axis=0)
                x_m=numpy.insert(x_m, s, x1[pos],  axis=0)
                s=s+1
            else:
                rus=numpy.insert(rus, t, x_c[pos]*D[i,0:2],  axis=0)
                t=t+1
        R_ok[0]=R[0]-sum(rus[:,0])
        R_ok[1]=R[1]-sum(rus[:,1])
        x_sol_n=numpy.zeros((s))
        x_sol_n=solpar(D_ok[:,0:2],R_ok)   
        cont=0
        for k in range(0,n):
            pos=D[k,2]
            pos=pos.astype(int)
            if x_c[pos]==0:
                x_f[pos]=x_sol_n[cont]
                cont=cont+1
            else:
                x_f[pos]=x_c[pos]
            
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x1)    
    
    return(x_f)    
    
    
    
    
    
    
def pra_p_d_pr(D,R,pr):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocprop_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    if max( perc[1,:],perc[2,:])>1: 
        x2=drf_pr(D[:,1:3],R[1:3],pr)
    else:
        x2=[1,1,1,1,1]
    x=[min(x1[0],x2[0]), min(x1[1],x2[1]),min(x1[2],x2[2]),min(x1[3],x2[3]),min(x1[4],x2[4])]
    return(x)


def pra_mmf_d_pr(D,R,pr):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmmf_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    if max( perc[1,:],perc[2,:])>1: 
        x2=drf_pr(D[:,1:3],R[1:3],pr)
    else:
        x2=[1,1,1,1,1]
    x=[min(x1[0],x2[0]), min(x1[1],x2[1]),min(x1[2],x2[2]),min(x1[3],x2[3]),min(x1[4],x2[4])]
    return(x)

def pra_mood_d_pr(D,R,pr):
    perc=numpy.zeros((3,1))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    for k in range(0,3): 
        perc[k,:]=sum(D[:,k])/R[k]  
    if perc[0,:]>1:         
        x1=allocmood_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    if max( perc[1,:],perc[2,:])>1: 
        x2=drf_pr(D[:,1:3],R[1:3],pr)
    else:
        x2=[1,1,1,1,1]
    x=[min(x1[0],x2[0]), min(x1[1],x2[1]),min(x1[2],x2[2]),min(x1[3],x2[3]),min(x1[4],x2[4])]
    return(x)
    
    
def drf_pr(D,R_ok,pr):
    D1=numpy.zeros((0,3))
    D2=numpy.zeros((0,3))
    D3=numpy.zeros((0,3))
    D4=numpy.zeros((0,3))
    inser=numpy.zeros((1,3))
    n=D[:,0].size
    x=numpy.zeros((n))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,n):
        inser=[D[j,0],D[j,1],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
          
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
         
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
         
            n4=n4+1
            
    a=numpy.zeros((n))
    a_sol=numpy.zeros((n))
    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok[0] and sum(D1[:,1])<R_ok[1]:
                for k in range(0,n1):
                    pos=D1[k,2]
                    pos=pos.astype(int)
                    a[pos]=1
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n1))
                soldrf=solpar(D1[:,0:2],R_ok)
                for k in range(0,n1):
                    pos=D1[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])] 
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok[0] and sum(D2[:,1])<R_ok[1]:
                for k in range(0,n2):
                    pos=D2[k,2]
                    pos=pos.astype(int)
                    a[pos]=1
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a_sol[k], D[k,1]*a_sol[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n2))
                soldrf=solpar(D2[:,0:2],R_ok)
                for k in range(0,n2):
                    pos=D2[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                n3=0
                n4=0
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok[0] and sum(D3[:,1])<R_ok[1]:
                for k in range(0,n3):
                    pos=D3[k,2]
                    pos=pos.astype(int)
                    a[pos]=1
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a_sol[k], D[k,1]*a_sol[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n3))
                soldrf=solpar(D3[:,0:2],R_ok)
                for k in range(0,n3):
                    pos=D3[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                n4=0
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a_sol[k], D[k,1]*a_sol[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok[0] and sum(D4[:,1])<R_ok[1]:
                for k in range(0,n4):
                    pos=D4[k,2]
                    pos=pos.astype(int)
                    a[pos]=1
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
            else:
                soldrf=numpy.zeros((n4))
                soldrf=solpar(D4[:,0:2],R_ok)
                for k in range(0,n4):
                    pos=D4[k,2]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                Rus_k=numpy.zeros((n,2))
                for k in range(0,n):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1])]
           
    for j in range(0,n):  
        x[j]=a[j]
    
    
    
    return(x)    
    
    
def ocra_p_d_pr(D,R,pr):
    n=D[:,0].size
    ordi=ordine(D,R)##se 1 ho cra se 2  inverto
    cont_reallo=0
    if ordi==1:
        x=priority_cra_p_d(D,R,pr)
    else:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((n,1))
        x1=numpy.zeros((n,1))
        x2=numpy.zeros((n,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if max(perc[1,:], perc[2,:])>1:         
            x1=drf_pr(D[:,1:3],R[1:3],pr)
        else:
            for k in range(0,n):
                x1[k]=1
        a=numpy.zeros((n))
        for j in range(0,n):
            a[j]=D[j,0]*x1[j]
        if  perc[0,:]>1 and (sum(a)>R[0]): 
            x2=allocprop_pr_up(D[:,0],R[0],pr,x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2 
    return(x, cont_reallo)
    
def ocra_mmf_d_pr(D,R,pr):
    ordi=ordine(D,R)##se 1 ho cra se 2  inverto
    cont_reallo=0
    if ordi==1:
        x=priority_cra_mmf_d(D,R,pr)
    else:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if max(perc[1,:], perc[2,:])>1:         
            x1=drf_pr(D[:,1:3],R[1:3],pr)
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,0]*x1[j]
        if  perc[0,:]>1 and (sum(a)>R[0]): 
            x2=allocmmf_pr_up(D[:,0],R[0],pr,x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2 
    return(x, cont_reallo)
    
def ocra_mood_d_pr(D,R,pr):
    ordi=ordine(D,R)##se 1 ho cra se 2  inverto
    cont_reallo=0
    if ordi==1:
        x=priority_cra_mood_d(D,R,pr)
    else:
        perc=numpy.zeros((3,1))
        x=numpy.zeros((5,1))
        x1=numpy.zeros((5,1))
        x2=numpy.zeros((5,1))
        for k in range(0,3): 
            perc[k,:]=sum(D[:,k])/R[k]  
        if max(perc[1,:], perc[2,:])>1:         
            x1=drf_pr(D[:,1:3],R[1:3],pr)
        else:
            x1=[1,1,1,1,1]
        a=numpy.zeros((5))
        for j in range(0,5):
            a[j]=D[j,0]*x1[j]
        if  perc[0,:]>1 and (sum(a)>R[0]): 
            x2=allocmood_pr_up(D[:,0],R[0],pr,x1)
            cont_reallo=cont_reallo+1
        else:
            x2=x1
        x=x2 
    return(x, cont_reallo)
    
def allocmmf_pr_up(D,R_ok,pr,x1):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((5))
    x1_1=numpy.zeros((0,1))
    x1_2=numpy.zeros((0,1))
    x1_3=numpy.zeros((0,1))
    x1_4=numpy.zeros((0,1))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,5):
        inser=[D[j],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            x1_1=numpy.insert(x1_1, n1, x1[j],  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            x1_2=numpy.insert(x1_2, n2, x1[j],  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            x1_3=numpy.insert(x1_3, n3, x1[j],  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            x1_4=numpy.insert(x1_4, n4, x1[j],  axis=0)
            n4=n4+1
            
    a=numpy.zeros((5))
    a_sol=numpy.zeros((5))

    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok:
                a=alloc1_up_1ris(D1,R_ok,x1,n1)
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n1))
                soldrf=mmf_up_n(D1,R_ok,x1,n1)
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D2,R_ok,x1,n2)
                for i in range(0,5):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n2))
                soldrf=mmf_up_n(D2,R_ok,x1,n2)
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n3=0
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D3,R_ok,x1,n3)
                for i in range(0,5):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n3))
                soldrf=mmf_up_n(D3,R_ok,x1,n3)
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D4,R_ok,x1,n4)
                for i in range(0,5):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n4))
                soldrf=mmf_up_n(D4,R_ok,x1,n4)
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k=D[k]*a[k]
               # R_ok=R_ok-sum(Rus_k)
           
    for j in range(0,5):  
        x[j]=a[j]
        
    return(x)
    

def allocprop_pr_up(D,R_ok,pr,x1):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    n=D.size
    x=numpy.zeros((n))
    x1_1=numpy.zeros((0,1))
    x1_2=numpy.zeros((0,1))
    x1_3=numpy.zeros((0,1))
    x1_4=numpy.zeros((0,1))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,n):
        inser=[D[j],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            x1_1=numpy.insert(x1_1, n1, x1[j],  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            x1_2=numpy.insert(x1_2, n2, x1[j],  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            x1_3=numpy.insert(x1_3, n3, x1[j],  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            x1_4=numpy.insert(x1_4, n4, x1[j],  axis=0)
            n4=n4+1
            
    a=numpy.zeros((n))
    a_sol=numpy.zeros((n))

    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok:
                a=alloc1_up_1ris(D1,R_ok,x1,n1)
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n1))
                soldrf=prop_up_n(D1,R_ok,x1,n1)
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D2,R_ok,x1,n2)
                for i in range(0,n):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n2))
                soldrf=prop_up_n(D2,R_ok,x1,n2)
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n3=0
                n4=0
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D3,R_ok,x1,n3)
                for i in range(0,n):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n3))
                soldrf=prop_up_n(D3,R_ok,x1,n3)
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n4=0
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D4,R_ok,x1,n4)
                for i in range(0,n):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n4))
                soldrf=prop_up_n(D4,R_ok,x1,n4)
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((n))
                for k in range(0,n):
                    Rus_k=D[k]*a[k]
               # R_ok=R_ok-sum(Rus_k)
           
    for j in range(0,n):  
        x[j]=a[j]
        
    return(x)
    
def prop_up_n(D,R,x_max,n_ok):
    x_up=numpy.zeros((5))
    for k in range(0,5):
        x_up[k]=x_max[k]
    x=numpy.zeros((5))
    for k in range(0,n_ok): 
        x[k]=R/sum(D[:,0])
    x_f=numpy.zeros((5))
    for k in range(0,n_ok):
        pos=D[k,1]
        pos=pos.astype(int)
        x_f[pos]=x[k] 
    x_c=numpy.zeros((6,1))
    x_c=controllo(x_f,x_up)
    
    while x_c[5,0]==1:
        
        s=0
        t=0
        D_ok=numpy.zeros((0,2))
        x_m=numpy.zeros((0,1))
        R_ok=numpy.zeros((1,1))
        rus=numpy.zeros((0,1))
        for i in range(0,n_ok):
            pos=D[i,1]
            pos=pos.astype(int)
            if x_c[pos]==0:
                D_ok=numpy.insert(D_ok, s, D[i,:],  axis=0)
                x_m=numpy.insert(x_m, s,x_max[i],  axis=0)
                s=s+1
            else:
                rus=numpy.insert(rus, t,x_c[i]*D[i,0],  axis=0)
                t=t+1
        R_ok=R-sum(rus[:,0])
        n=s
        x=numpy.zeros((5))
        for k in range(0,n): 
            x[k]=R_ok/sum(D_ok[:,0])
        
        cont=0
        x_f=numpy.zeros((5))
        for i in range (0,5):
            if x_c[i]==0:
                for k in range(0,n):
                    pos=D_ok[k,1]
                    pos=pos.astype(int)
                    x_f[pos]=x[k] 
                    cont=cont+1
            else:
                x_f[i]=x_c[i]
                
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_up)
                   
 #   for i in range(0,n_ok):
  #      x_rest[i]=x_f[i]    
    
    return(x_f)
    
    
def alloc1_up_1ris(D,R,x_up,n1):
    a=numpy.zeros(5)
    for k in range(0,n1):
        pos=D[k,1]
        pos=pos.astype(int)
        a[pos]=1
    x_c=numpy.zeros((6,1))
    x_c=controllo(a,x_up)
    if x_c[5]==1:
        for i in range(0,n1):
            pos=D[i,1]
            pos=pos.astype(int)
            if x_c[pos]>0:
                a[pos]=x_up[pos]
    return(a)
    

    
def mmf_up_n(D_ok,R,x_max,n):
    D=D_ok[:,0]
    x=numpy.zeros((n))
    a_f_ok=numpy.zeros((n))
    x_f=numpy.zeros((5))
    x_f_ok=numpy.zeros((5))
    x_rest=numpy.zeros((n))
    x_c=numpy.zeros((6))
    x_up=numpy.zeros((6))
    for k in range(0,n): 
        x_up[k]=x_max[k]
    
    
    
    if n==5:
        dom=numpy.sort(D)
        pos=sorted(range(len(D)), key=lambda k: D[k])
        if dom[0]<(R/n):
            x[0]=dom[0]    
            if dom[1]<(R-dom[0])/(n-1):
                x[1]=dom[1] 
                if dom[2]<(R-dom[0]-dom[1])/(n-2):
                    x[2]=dom[2] 
                    if dom[3]<(R-dom[0]-dom[1]-dom[2])/(n-3):
                        x[3]=dom[3] 
                        if dom[4]<(R-dom[0]-dom[1]-dom[2]-dom[3])/(n-4):
                            x[4]=dom[4] 
                        else:
                            x=[dom[0],dom[1],dom[2],dom[3],(R-dom[0]-dom[1]-dom[2]-dom[3])/(n-4)];
                    else:
                        x=[dom[0],dom[1],dom[2],(R-dom[0]-dom[1]-dom[2])/(n-3),(R-dom[0]-dom[1]-dom[2])/(n-3)];
                else:
                    x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2)];       
            else:
                x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
                                  
        else:
            x=[R/n,R/n,R/n,R/n,R/n]
        for i in range(0,5):
            a_f_ok[pos[i]]=x[i]
        for i in range(0,5):
            x_f[i]=a_f_ok[i]/D[i]
       
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_up)
         
        while x_c[5,0]==1:
            Dnew=numpy.zeros(((x_c[0:5,0] == 0).sum()))
            contat=0
            for lun in range (0,5):
                if x_c[lun,0] == 0:
                    Dnew[contat]=D[lun]
                    contat=contat+1
            x=numpy.zeros(5)
            if ((x_c[0:5,0] == 0).sum())==1:
                x=[max(Dnew,R)]
            if ((x_c[0:5,0] == 0).sum())==2:  
                x=mmf2(Dnew,R)
            if ((x_c[0:5,0] == 0).sum())==3:
                x=mmf3(Dnew,R)
            if ((x_c[0:5,0] == 0).sum())==4:  
                x=mmf4(Dnew,R)
                
            if ((x_c[0:5,0] == 0).sum())==5:
                x=allocmmf(Dnew,R)
            cont=0
            x_f=numpy.zeros((5))
            for i in range (0,5):
                if x_c[i]==0:
                    x_f[i]=x[cont]
                    cont=cont+1
                else:
                    x_f[i]=x_c[i]
            x_c=numpy.zeros((6,1))
            x_c=controllo(x_f,x_up)
            
    if n==4:
        dom=numpy.sort(D)
        pos=sorted(range(len(D)), key=lambda k: D[k])
        if dom[0]<(R/n):
            x[0]=dom[0]    
            if dom[1]<(R-dom[0])/(n-1):
                x[1]=dom[1] 
                if dom[2]<(R-dom[0]-dom[1])/(n-2):
                    x[2]=dom[2] 
                    if dom[3]<(R-dom[0]-dom[1]-dom[2])/(n-3):
                        x[3]=dom[3] 
                    else:
                        x=[dom[0],dom[1],dom[2],(R-dom[0]-dom[1]-dom[2])/(n-3)];
                else:
                    x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2),(R-dom[0]-dom[1])/(n-2)];       
            else:
                x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
                                  
        else:
            x=[R/n,R/n,R/n,R/n]
            
        for i in range(0,4):
            a_f_ok[pos[i]]=x[i]
        for i in range(0,4):
            x_f_ok[i]=a_f_ok[i]/D[i]
        for i in range(0,5):
            pos=D_ok[k,1]
            pos=pos.astype(int)
            x_f[pos]=x_f_ok[i]   
            
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_up)
         
        while x_c[5,0]==1:
            Dnew=numpy.zeros(((x_c[0:4,0] == 0).sum()))
            contat=0
            for lun in range (0,4):
                if x_c[lun,0] == 0:
                    Dnew[contat]=D[lun]
                    contat=contat+1
            x=numpy.zeros(4)
            if ((x_c[0:4,0] == 0).sum())==1:
                x=[max(Dnew,R)]
            if ((x_c[0:4,0] == 0).sum())==2:  
                x=mmf2(Dnew,R)
            if ((x_c[0:4,0] == 0).sum())==3:
                x=mmf3(Dnew,R)
            if ((x_c[0:4,0] == 0).sum())==4:  
                x=mmf4(Dnew,R)
                
            cont=0
            x_f=numpy.zeros((5))
            for i in range (0,5):
                if x_c[i]==0:
                    x_f[i]=x[cont]
                    cont=cont+1
                else:
                    x_f[i]=x_c[i]
            x_c=numpy.zeros((6,1))
            x_c=controllo(x_f,x_up)
            
    if n==3:
        dom=numpy.sort(D)
        pos=sorted(range(len(D)), key=lambda k: D[k])
        if dom[0]<(R/n):
            x[0]=dom[0]    
            if dom[1]<(R-dom[0])/(n-1):
                x[1]=dom[1] 
                if dom[2]<(R-dom[0]-dom[1])/(n-2):
                    x[2]=dom[2] 
                    
                else:
                    x=[dom[0],dom[1],(R-dom[0]-dom[1])/(n-2)];       
            else:
                x=[dom[0],(R-dom[0])/(n-1),(R-dom[0])/(n-1)];  
                                  
        else:
            x=[R/n,R/n,R/n]
        for i in range(0,3):
            a_f_ok[pos[i]]=x[i]
        for i in range(0,3):
            x_f_ok[i]=a_f_ok[i]/D[i]
        for i in range(0,5):
            pos=D_ok[k,1]
            pos=pos.astype(int)
            x_f[pos]=x_f_ok[i]   
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_up)
         
        while x_c[5,0]==1:
            Dnew=numpy.zeros(((x_c[0:3,0] == 0).sum()))
            contat=0
            for lun in range (0,3):
                if x_c[lun,0] == 0:
                    Dnew[contat]=D[lun]
                    contat=contat+1
            x=numpy.zeros(3)
            if ((x_c[0:3,0] == 0).sum())==1:
                x=[max(Dnew,R)]
            if ((x_c[0:3,0] == 0).sum())==2:  
                x=mmf2(Dnew,R)
            if ((x_c[0:3,0] == 0).sum())==3:
                x=mmf3(Dnew,R)
            cont=0
            x_f=numpy.zeros((5))
            for i in range (0,5):
                if x_c[i]==0:
                    x_f[i]=x[cont]
                    cont=cont+1
                else:
                    x_f[i]=x_c[i]
            x_c=numpy.zeros((6,1))
            x_c=controllo(x_f,x_up)        
            
    if n==2:
        dom=numpy.sort(D)
        pos=sorted(range(len(D)), key=lambda k: D[k])
        if dom[0]<(R/n):
            x[0]=dom[0]    
            if dom[1]<(R-dom[0])/(n-1):
                x[1]=dom[1]     
            else:
                x=[dom[0],(R-dom[0])/(n-1)];  
                                  
        else:
            x=[R/n,R/n]
        for i in range(0,2):
            a_f_ok[pos[i]]=x[i]
        for i in range(0,2):
            x_f_ok[i]=a_f_ok[i]/D[i]
        for i in range(0,5):
            pos=D_ok[k,1]
            pos=pos.astype(int)
            x_f[pos]=x_f_ok[i]   
       
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_up)
         
        while x_c[5,0]==1:
            Dnew=numpy.zeros(((x_c[0:2,0] == 0).sum()))
            contat=0
            for lun in range (0,2):
                if x_c[lun,0] == 0:
                    Dnew[contat]=D[lun]
                    contat=contat+1
            x=numpy.zeros(2)
            if ((x_c[0:2,0] == 0).sum())==1:
                x=[max(Dnew,R)]
            if ((x_c[0:2,0] == 0).sum())==2:  
                x=mmf2(Dnew,R)
            cont=0
            x_f=numpy.zeros((5))
            for i in range (0,5):
                if x_c[i]==0:
                    x_f[i]=x[cont]
                    cont=cont+1
                else:
                    x_f[i]=x_c[i]
            x_c=numpy.zeros((6,1))
            x_c=controllo(x_f,x_up)  
        
    if n==1:
        
        if D <R:
            x[0]=D 
        else:
            x[0]=R
            pos=D_ok[k,1]
            pos=pos.astype(int)
            x_f[pos]=x[0]/D  
       
        if x_f[0]>x_max[0]:
            x_f[0]=x_max[0]
    
    
    return(x_f)






def pra2_p_d_pr(D,R,pr):
    perc=numpy.zeros((3))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    share=numpy.zeros((5,3))
    cong=numpy.zeros((2))
    for k in range(0,3): 
        perc[k]=sum(D[:,k])/R[k]  
    if perc[0]>1:         
        x1=allocprop_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    if max( perc[1],perc[2])>1: 
        x2=drf_pr(D[:,1:3],R[1:3],pr)
    else:
        x2=[1,1,1,1,1]
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_p_pra_pr(share, D,R,x1,x2,pr)
    else:
        x=sol_drf_pra_pr(share, D,R,x1,x2,pr)
    return(x)
    
    
    

def pra2_p_d_pr2(D,R,pr):
    share=numpy.zeros((5,3))
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_p_pra_pr2(share, D,R,pr)
    else:
        x=sol_drf_pra_pr2(share, D,R,pr)
    return(x)

def pra2_mmf_d_pr2(D,R,pr):
    share=numpy.zeros((5,3))
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_drf_pra_pr2(share, D,R,pr)
    else:
        x=sol_drf_pra_pr2(share, D,R,pr)
    return(x)

def pra2_mood_d_pr2(D,R,pr):
    share=numpy.zeros((5,3))
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_mood_pra_pr2(share, D,R,pr)
    else:
        x=sol_drf_pra_pr2(share, D,R,pr)
    return(x)
    
    
    
    
def sol_mood_pra_pr2(share, D,R,pr):
 
    D1=numpy.zeros((0,4))
    D2=numpy.zeros((0,4))
    D3=numpy.zeros((0,4))
    D4=numpy.zeros((0,4))
    D1dr=numpy.zeros((0,2))
    D2dr=numpy.zeros((0,2))
    D3dr=numpy.zeros((0,2))
    D4dr=numpy.zeros((0,2))
    inser=numpy.zeros((1,3))
    x=numpy.zeros((5))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    Rleft=numpy.zeros((3))
    for j in range(0,len(pr)):
        inser=[D[j,0],D[j,1],D[j,2],j]
        if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
        if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
        if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
        if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
    fine=0
    if sum(D1[:,0])<R[0] and sum(D1[:,1])<R[1] and sum(D1[:,2])<R[2]:
        p=numpy.zeros((n1))
        p=D1[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1          
        Rleft=[R[0]-sum(D1[:,0]),R[1]-sum(D1[:,1]),R[2]-sum(D1[:,2])]
    else:
        p=numpy.zeros((n1))
        p=D1[:,3]
        fine=1
        xpar=solparmood(D1,R)
        for i in range(0,len(p)):
            x[int(p[i])]=xpar[i]
    if fine==0 and sum(D2[:,0])<Rleft[0] and sum(D2[:,1])<Rleft[1] and sum(D2[:,2])<Rleft[2]:
        p=numpy.zeros((n2))
        p=D2[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
        Rleft=[Rleft[0]-sum(D2[:,0]),Rleft[1]-sum(D2[:,1]),Rleft[2]-sum(D2[:,2])]
    else:
        p=numpy.zeros((n2))
        p=D2[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            xpar=solparmood(D2,Rleft)
            for i in range(0,len(p)):
                x[int(p[i])]=xpar[i]
    if fine==0 and sum(D3[:,0])<Rleft[0] and sum(D3[:,1])<Rleft[1] and sum(D3[:,2])<Rleft[2]:
        p=numpy.zeros((n3))
        p=D3[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
        Rleft=[Rleft[0]-sum(D3[:,0]),Rleft[1]-sum(D3[:,1]),Rleft[2]-sum(D3[:,2])]
    else:
        p=numpy.zeros((n3))
        p=D3[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            xpar=solparmood(D3,Rleft)  
            for i in range(0,len(p)):
                x[int(p[i])]=xpar[i]
    if fine==0 and sum(D4[:,0])<Rleft[0]  and sum(D4[:,1])<Rleft[1] and sum(D4[:,2])<Rleft[2]:
        p=numpy.zeros((n4))
        p=D4[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
    else:
        p=numpy.zeros((n4))
        p=D4[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            xpar=solparmood(D4,Rleft) 
            for i in range(0,len(p)):
                x[int(p[i])]=xpar[i]
    return(x)   
    





 
    
def solparmood(Dnew, R):
        
    n=len(Dnew[:,0])
    nbcont=len(Dnew[:,0])
    nris=3
#nbvar=15

# Range of plants and warehouses
    lignes = range(nbcont)
#colonnes = range(nbvar)


# Matrice des contraintes
    a =Dnew[:,0:3]
  

# Second membre
    sec= R

    
    
    ds=numpy.zeros(n)
    ds_user=numpy.zeros(3)
    p=numpy.zeros(n)
    div=numpy.zeros(n)
    minimo=numpy.zeros(n)
    massimo=numpy.zeros(n)
    for i in range(0,n):
        for j in range(0,3):
            ds_user[j]=a[i,j]/sec[j]
        ds[i]=max(ds_user)
        p[i]=list(ds_user).index(ds[i])

    for i in range(0,n):
        minimo[i]=max(sec[int(p[i])]-sum (a[:,int(p[i])])+a[i,int(p[i])],0)
        massimo[i]=min(sec[int(p[i])],a[i,int(p[i])])
        div[i]=massimo[i]-minimo[i]
    print(div)
#w1=1
#w2=1
#w3=1

# Coefficients de la fonction objectif
    c=numpy.zeros(n)
    c[0]=1

    m = Model("mogpl5ex")     
        
# declaration variables de decision
    x = []

    for i in range(0,n):
        x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name="x%d" % (i+1)))


    
### r e b
    r=[]

    for i in range(0,n):
        r.append(m.addVar(vtype=GRB.CONTINUOUS,  name="r%d" % (i+1)))

    b=[]
    for j in range(0,n):
        for i in range(0,n):
            b.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="b%d" % ((n)*(j)+i+1)))



# maj du modele pour integrer les nouvelles variables
    m.update()

    obj = LinExpr();
    obj = 0
    obj += c[0] *( r[0]-sum( b[0:n] ))
        
# definition de l'objectif
    m.setObjective(obj,GRB.MAXIMIZE)





# Definition des contraintes
    for i in range(0,nris):
        m.addConstr(quicksum(a[j][i]*x[j] for j in lignes) <= sec[i], "c%d" % i)
    for i in lignes:
        for j in lignes:
            if massimo[j]==minimo[j]:
                m.addConstr(r[i]-b[n*i+j]-a[j,int(p[j])]*x[j] <= 0, "c%d" % (n+n*i+j))
            else:
                    m.addConstr(r[i]-b[n*i+j]-(a[j,int(p[j])]*x[j]-minimo[j])/(float(div[j])) <= 0, "c%d" % (n+n*i+j))
    for i in range(0,n):
        if massimo[i]==minimo[i]:
            
            m.addConstr(a[i,int(p[i])]*x[i] <= 1 )  
        else:
                
            m.addConstr((a[i,int(p[i])]*x[i]-minimo[i])/(float(div[i])) <= 1 )    
 
               
        
        
        
        
# Resolution
    m.optimize()


#    print ("")                
#    print ('Solution optimale:')
    for j in lignes:
        x[j]=x[j].x
        
    x_f=numpy.zeros((n,1))   
    x_f=x
    
    return(x_f)
    
    
    
    
    
    
    
    

def sol_p_pra_pr2(share, D,R,pr):
 
    D1=numpy.zeros((0,4))
    D2=numpy.zeros((0,4))
    D3=numpy.zeros((0,4))
    D4=numpy.zeros((0,4))
    D1dr=numpy.zeros((0,2))
    D2dr=numpy.zeros((0,2))
    D3dr=numpy.zeros((0,2))
    D4dr=numpy.zeros((0,2))
    inser=numpy.zeros((1,3))
    x=numpy.zeros((5))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    Rleft=numpy.zeros((3))
    for j in range(0,len(pr)):
        inser=[D[j,0],D[j,1],D[j,2],j]
        if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
        if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
        if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
        if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
    fine=0
    if sum(D1[:,0])<R[0] and sum(D1[:,1])<R[1] and sum(D1[:,2])<R[2]:
        p=numpy.zeros((n1))
        p=D1[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1        
        Rleft=[R[0]-sum(D1[:,0]),R[1]-sum(D1[:,1]),R[2]-sum(D1[:,2])]
    else:
        p=numpy.zeros((n1))
        p=D1[:,3]
        fine=1
        cong1=[sum(D1[:,0])/R[0],max(sum(D1[:,1])/R[1],sum(D1[:,2])/R[2]) ]  
        congmax=max(cong1)
        for i in range(0,len(p)):
            x[int(p[i])]=1.0/congmax
    if fine==0 and sum(D2[:,0])<Rleft[0] and sum(D2[:,1])<Rleft[1] and sum(D2[:,2])<Rleft[2]:
        p=numpy.zeros((n2))
        p=D2[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
        Rleft=[Rleft[0]-sum(D2[:,0]),Rleft[1]-sum(D2[:,1]),Rleft[2]-sum(D2[:,2])]
    else:
        p=numpy.zeros((n2))
        p=D2[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            cong2=[sum(D2[:,0])/Rleft[0],max(sum(D2[:,1])/Rleft[1],sum(D2[:,2])/Rleft[2]) ]  
            congmax=max(cong2)
            for i in range(0,len(p)):
                x[int(p[i])]=1.0/congmax
    if fine==0 and sum(D3[:,0])<Rleft[0] and sum(D3[:,1])<Rleft[1] and sum(D3[:,2])<Rleft[2]:
        p=numpy.zeros((n3))
        p=D3[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
        Rleft=[Rleft[0]-sum(D3[:,0]),Rleft[1]-sum(D3[:,1]),Rleft[2]-sum(D3[:,2])]
    else:
        p=numpy.zeros((n3))
        p=D3[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            cong3=[sum(D3[:,0])/Rleft[0],max(sum(D3[:,1])/Rleft[1],sum(D3[:,2])/Rleft[2]) ]  
            congmax=max(cong3)
            for i in range(0,len(p)):
                x[int(p[i])]=1.0/congmax
    if fine==0 and sum(D4[:,0])<Rleft[0]  and sum(D4[:,1])<Rleft[1] and sum(D4[:,2])<Rleft[2]:
        p=numpy.zeros((n4))
        p=D4[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
    else:
        p=numpy.zeros((n4))
        p=D4[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            cong4=[sum(D4[:,0])/Rleft[0],max(sum(D4[:,1])/Rleft[1],sum(D4[:,2])/Rleft[2]) ]  
            congmax=max(cong4)
            for i in range(0,len(p)):
                x[int(p[i])]=1.0/congmax
    return(x)       
    
def sol_drf_pra_pr2(share, D,R,pr):
 
    D1=numpy.zeros((0,4))
    D2=numpy.zeros((0,4))
    D3=numpy.zeros((0,4))
    D4=numpy.zeros((0,4))
    D1dr=numpy.zeros((0,2))
    D2dr=numpy.zeros((0,2))
    D3dr=numpy.zeros((0,2))
    D4dr=numpy.zeros((0,2))
    inser=numpy.zeros((1,3))
    x=numpy.zeros((5))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    Rleft=numpy.zeros((3))
    for j in range(0,len(pr)):
        inser=[D[j,0],D[j,1],D[j,2],j]
        if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
        if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
        if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
        if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
    fine=0
    if sum(D1[:,0])<R[0] and sum(D1[:,1])<R[1] and sum(D1[:,2])<R[2]:
        p=numpy.zeros((n1))
        p=D1[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1          
        Rleft=[R[0]-sum(D1[:,0]),R[1]-sum(D1[:,1]),R[2]-sum(D1[:,2])]
    else:
        p=numpy.zeros((n1))
        p=D1[:,3]
        fine=1
        xpar=solpar_3(D1,R)
        for i in range(0,len(p)):
            x[int(p[i])]=xpar[i]
    if fine==0 and sum(D2[:,0])<Rleft[0] and sum(D2[:,1])<Rleft[1] and sum(D2[:,2])<Rleft[2]:
        p=numpy.zeros((n2))
        p=D2[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
        Rleft=[Rleft[0]-sum(D2[:,0]),Rleft[1]-sum(D2[:,1]),Rleft[2]-sum(D2[:,2])]
    else:
        p=numpy.zeros((n2))
        p=D2[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            xpar=solpar_3(D2,Rleft)
            for i in range(0,len(p)):
                x[int(p[i])]=xpar[i]
    if fine==0 and sum(D3[:,0])<Rleft[0] and sum(D3[:,1])<Rleft[1] and sum(D3[:,2])<Rleft[2]:
        p=numpy.zeros((n3))
        p=D3[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
        Rleft=[Rleft[0]-sum(D3[:,0]),Rleft[1]-sum(D3[:,1]),Rleft[2]-sum(D3[:,2])]
    else:
        p=numpy.zeros((n3))
        p=D3[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            xpar=solpar_3(D3,Rleft)  
            for i in range(0,len(p)):
                x[int(p[i])]=xpar[i]
    if fine==0 and sum(D4[:,0])<Rleft[0]  and sum(D4[:,1])<Rleft[1] and sum(D4[:,2])<Rleft[2]:
        p=numpy.zeros((n4))
        p=D4[:,3]
        for i in range(0,len(p)):
            x[int(p[i])]=1  
    else:
        p=numpy.zeros((n4))
        p=D4[:,3]
        if fine==1:
            for i in range(0,len(p)):
                x[int(p[i])]=0 
        else:
            fine=1
            xpar=solpar_3(D4,Rleft) 
            for i in range(0,len(p)):
                x[int(p[i])]=xpar[i]
    return(x)   
    



    
def sol_p_pra_pr(share, D,R,x1,x2,pr):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    if sum(D[:,0])>R[0]:
        soluz_prop_n=all_p_n(D[:,0],R[0],pr,p,drf,x2)
        for i in range(0,5):
            x[i]=soluz_prop_n[i]
    else:
         for i in range(0,5):
            x[i]=1
    return(x)    
     
    
    
    
    
def all_p_n(D,R_ok,pr,p,drf,x2):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    D1dr=numpy.zeros((0,2))
    D2dr=numpy.zeros((0,2))
    D3dr=numpy.zeros((0,2))
    D4dr=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((len(pr)))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    for j in range(0,len(pr)):
        if numpy.isin(j,p):
            inser=[D[j],j]
            if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
            if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
            if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
            if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
        else:
            inser=[D[j],j]
            if pr[j]==1:
                D1dr=numpy.insert(D1dr, n1dr, inser,  axis=0)
                n1=n1+1
                n1dr=n1dr+1
            if pr[j]==2:
                D2dr=numpy.insert(D2dr, n2dr, inser,  axis=0)
                n2=n2+1
                n2dr=n2dr+1
            if pr[j]==3:
                D3dr=numpy.insert(D3dr, n3dr, inser,  axis=0)
                n3=n3+1
                n3dr=n3dr+1
            if pr[j]==4:
                D4dr=numpy.insert(D4dr, n4dr, inser,  axis=0)
                n4=n4+1
                n4dr=n4dr+1
    a=numpy.zeros((len(pr)))
    for j in range(0,4):
        if j==0 and n1>0:
            
            if len(D1dr[:,0])>0:
                Rus_k=numpy.zeros((len(D1dr[:,0])))
                for k in range(0,len(D1dr[:,0])):
                    pos=D1dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D1dr[k,0]*a[pos]                
                R_ok=R_ok-sum(Rus_k)
            if len(D1[:,0])>0:
                if sum(D1[:,0])<R_ok:
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D1[:,0])
                else:
                    soldrf=numpy.zeros((n1-len(D1dr[:,0])))
                    for k in range(0,n1-len(D1dr[:,0])):   
                            soldrf[k]=R_ok/sum(D1[:,0])
                    Rus_k=numpy.zeros((n1-len(D1dr[:,0])))
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D1[k,0]*a[pos]                   
                    R_ok=R_ok-sum(Rus_k)
                    n2=0
                    n3=0
                    n4=0
        if j==1 and n2>0:
            if len(D2dr[:,0])>0:
                Rus_k=numpy.zeros((len(D2dr[:,0])))
                for k in range(0,len(D2dr[:,0])):
                    pos=D2dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D2dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D2[:,0])>0:
                if sum(D2[:,0])<R_ok:
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D2[:,0])
                else:
                    Rus_k=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=numpy.zeros((n2-len(D2dr[:,0])))
                    for k in range(0,n2-len(D2dr[:,0])):   
                            soldrf[k]=R_ok/sum(D2[:,0])
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D2[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
                    n3=0
                    n4=0
        if j==2 and n3>0:
            if len(D3dr[:,0])>0:
                Rus_k=numpy.zeros((len(D3dr[:,0])))
                for k in range(0,len(D3dr[:,0])):
                    pos=D3dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D3dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D3[:,0])>0:
                if sum(D3[:,0])<R_ok:
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D3[:,0])
                else:
                    soldrf=numpy.zeros((n3-len(D3dr[:,0])))
                    for k in range(0,n3-len(D3dr[:,0])):   
                        soldrf[k]=R_ok/sum(D3[:,0])
                    Rus_k=numpy.zeros((n3-len(D3dr[:,0])))
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D3[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
                    n4=0

        if j==3 and n4>0:
            if len(D4dr[:,0])>0:
                Rus_k=numpy.zeros((len(D4dr[:,0])))
                for k in range(0,len(D4dr[:,0])):
                    pos=D4dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D4dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D4[:,0])>0:
                if sum(D4[:,0])<R_ok:
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D4[:,0])
                else:
                    soldrf=numpy.zeros((n4-len(D4dr[:,0])))
                    for k in range(0,n4-len(D4dr[:,0])):   
                            soldrf[k]=R_ok/sum(D4[:,0])
                    Rus_k=numpy.zeros((n4-len(D4dr[:,0])))
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D4[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
    for j in range(0,len(pr)):  
        x[j]=a[j]
    
    return(x)   
    
    
def sol_drf_pra_pr(share, D,R,x1,x2,pr):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    if sum(D[:,1])>R[1] and sum(D[:,2])>R[2] :
        soluz_prop_n=solpar_pr(D[:,1:3],R[1:3],pr,p,drf,x1)
        for i in range(0,5):
            x[i]=soluz_prop_n[i]
    else:
         for i in range(0,5):
            x[i]=1
    return(x)  

    
    
    
def solpar_pr(D,R_ok,pr,p,drf,x2):
    D1=numpy.zeros((0,3))
    D2=numpy.zeros((0,3))
    D3=numpy.zeros((0,3))
    D4=numpy.zeros((0,3))
    D1dr=numpy.zeros((0,3))
    D2dr=numpy.zeros((0,3))
    D3dr=numpy.zeros((0,3))
    D4dr=numpy.zeros((0,3))
    inser=numpy.zeros((1,3))
    x=numpy.zeros((len(pr)))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    for j in range(0,len(pr)):
        if numpy.isin(j,drf):
            inser=[D[j,0],D[j,1],j]
            if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
            if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
            if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
            if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
        else:
            inser=[D[j,0],D[j,1],j]
            if pr[j]==1:
                D1dr=numpy.insert(D1dr, n1dr, inser,  axis=0)
                n1=n1+1
                n1dr=n1dr+1
            if pr[j]==2:
                D2dr=numpy.insert(D2dr, n2dr, inser,  axis=0)
                n2=n2+1
                n2dr=n2dr+1
            if pr[j]==3:
                D3dr=numpy.insert(D3dr, n3dr, inser,  axis=0)
                n3=n3+1
                n3dr=n3dr+1
            if pr[j]==4:
                D4dr=numpy.insert(D4dr, n4dr, inser,  axis=0)
                n4=n4+1
                n4dr=n4dr+1
    a=numpy.zeros((len(pr)))
    for j in range(0,4):
        if j==0 and n1>0:
            if len(D1dr[:,0])>0:
                Rus_k=numpy.zeros((len(D1dr[:,0]),2))
                for k in range(0,len(D1dr[:,0])):
                    pos=D1dr[k,2]
                    pos=pos.astype(int)
                    a[pos]=x2[pos] 
                    Rus_k[k,:]=[D1dr[k,0]*a[pos],D1dr[k,1]*a[pos]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]),R_ok[1]-sum(Rus_k[:,1])]
            if len(D1[:,0])>0:
                if sum(D1[:,0])<R_ok[0] and sum(D1[:,1])<R_ok[1]:
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,2]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=[R_ok[0]-sum(D1[:,0]),R_ok[1]-sum(D1[:,1])]
                else:
                    soldrf=numpy.zeros((n1-len(D1dr[:,0])))
                    soldrf=solpar(D1[:,0:2],R_ok)
                    Rus_k=numpy.zeros((n1-len(D1dr[:,0])))
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,2]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D1[k,0]*soldrf[k]
                    R_ok=[R_ok[0]-sum(D1[:,0]),R_ok[1]-sum(D1[:,1])]
                    n2=0
                    n3=0
                    n4=0
        if j==1 and n2>0:
            if len(D2dr[:,0])>0:
                Rus_k=numpy.zeros((len(D2dr[:,0]),2))
                for k in range(0,len(D2dr[:,0])):
                    pos=D2dr[k,2]
                    pos=pos.astype(int)
                    a[pos]=x2[pos] 
                    Rus_k[k,:]=[D2dr[k,0]*a[pos],D2dr[k,1]*a[pos]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]),R_ok[1]-sum(Rus_k[:,1])]
            if len(D2[:,0])>0:
                if sum(D2[:,0])<R_ok[0] and sum(D2[:,1])<R_ok[1]:
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,2]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=[R_ok[0]-sum(D2[:,0]),R_ok[1]-sum(D2[:,1])]
                else:                    
                    Rus_k=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=solpar(D2[:,0:2],R_ok)
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,2]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D2[k,0]*a[pos]
                    R_ok=[R_ok[0]-sum(D2[:,0]),R_ok[1]-sum(D2[:,1])]
                    n3=0
                    n4=0
        if j==2 and n3>0:
            if len(D3dr[:,0])>0:
                Rus_k=numpy.zeros((len(D3dr[:,0]),2))
                for k in range(0,len(D3dr[:,0])):
                    pos=D3dr[k,2]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]               
                    Rus_k[k,:]=[D3dr[k,0]*a[pos],D3dr[k,1]*a[pos]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]),R_ok[1]-sum(Rus_k[:,1])]
            if len(D3[:,0])>0:
                if sum(D3[:,0])<R_ok[0] and sum(D3[:,1])<R_ok[1]:
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,2]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=[R_ok[0]-sum(D3[:,0]),R_ok[1]-sum(D3[:,1])]
                else: 
                    Rus_k=numpy.zeros((n3-len(D3dr[:,0])))
                    soldrf=numpy.zeros((n3-len(D3dr[:,0])))
                    soldrf=solpar(D3[:,0:2],R_ok)
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,2]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D3[k,0]*a[pos]
                    R_ok=[R_ok[0]-sum(D3[:,0]),R_ok[1]-sum(D3[:,1])]
                    n4=0

        if j==3 and n4>0:
            if len(D4dr[:,0])>0: 
                Rus_k=numpy.zeros((len(D4dr[:,0]),2))
                for k in range(0,len(D4dr[:,0])):
                    pos=D4dr[k,2]
                    pos=pos.astype(int)
                    a[pos]=x2[pos] 
               
        
                    Rus_k[k,:]=[D4dr[k,0]*a[pos],D4dr[k,1]*a[pos]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]),R_ok[1]-sum(Rus_k[:,1])]
            if len(D4[:,0])>0:
                if sum(D4[:,0])<R_ok[0] and sum(D4[:,1])<R_ok[1]:
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,2]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=[R_ok[0]-sum(D4[:,0]),R_ok[1]-sum(D4[:,1])]
                else:
                    Rus_k=numpy.zeros((n4-len(D4dr[:,0])))
                    soldrf=numpy.zeros((n4-len(D4dr[:,0])))
                    soldrf=solpar(D4[:,0:2],R_ok)
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,2]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D4[k,0]*a[pos]
                    R_ok=[R_ok[0]-sum(D4[:,0]),R_ok[1]-sum(D4[:,1])]

    for j in range(0,len(pr)):  
        x[j]=a[j]
    
    return(x)
    
    
    
  
    
def pra2_mmf_d_pr(D,R,pr):
    perc=numpy.zeros((3))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    share=numpy.zeros((5,3))
    cong=numpy.zeros((2))
    for k in range(0,3): 
        perc[k]=sum(D[:,k])/R[k]  
    if perc[0]>1:         
        x1=allocmmf_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    if max( perc[1],perc[2])>1: 
        x2=drf_pr(D[:,1:3],R[1:3],pr)
    else:
        x2=[1,1,1,1,1]
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_mmf_pra_pr(share, D,R,x1,x2,pr)
    else:
        x=sol_drf_pra_pr(share, D,R,x1,x2,pr)
    return(x)   
    
    
def sol_mmf_pra_pr(share, D,R,x1,x2,pr):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    if sum(D[:,0])>R[0]:
        soluz_prop_n=solmmf_par_pr(D[:,0],R[0],pr,p,drf,x2)
        for i in range(0,5):
            x[i]=soluz_prop_n[i]
    else:
         for i in range(0,5):
            x[i]=1
    return(x)    
   
    
    
def solmmf_par_pr(D,R_ok,pr,p,drf,x2):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    D1dr=numpy.zeros((0,2))
    D2dr=numpy.zeros((0,2))
    D3dr=numpy.zeros((0,2))
    D4dr=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((len(pr)))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    for j in range(0,len(pr)):
        if numpy.isin(j,p):
            inser=[D[j],j]
            if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
            if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
            if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
            if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
        else:
            inser=[D[j],j]
            if pr[j]==1:
                D1dr=numpy.insert(D1dr, n1dr, inser,  axis=0)
                n1=n1+1
                n1dr=n1dr+1
            if pr[j]==2:
                D2dr=numpy.insert(D2dr, n2dr, inser,  axis=0)
                n2=n2+1
                n2dr=n2dr+1
            if pr[j]==3:
                D3dr=numpy.insert(D3dr, n3dr, inser,  axis=0)
                n3=n3+1
                n3dr=n3dr+1
            if pr[j]==4:
                D4dr=numpy.insert(D4dr, n4dr, inser,  axis=0)
                n4=n4+1
                n4dr=n4dr+1
    a=numpy.zeros((len(pr)))
    for j in range(0,4):
        if j==0 and n1>0:
            
            if len(D1dr[:,0])>0:
                Rus_k=numpy.zeros((len(D1dr[:,0])))
                for k in range(0,len(D1dr[:,0])):
                    pos=D1dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D1dr[k,0]*a[pos]                
                R_ok=R_ok-sum(Rus_k)
            if len(D1[:,0])>0:
                if sum(D1[:,0])<R_ok:
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D1[:,0])
                else:
                    soldrf=numpy.zeros((n1-len(D1dr[:,0])))
                    soldrf=solmmf_par(D1[:,0],R_ok,n1-len(D1dr[:,0]))
                    Rus_k=numpy.zeros((n1-len(D1dr[:,0])))
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D1[k,0]*a[pos]                   
                    R_ok=R_ok-sum(Rus_k)
                    n2=0
                    n3=0
                    n4=0
        if j==1 and n2>0:
            if len(D2dr[:,0])>0:
                Rus_k=numpy.zeros((len(D2dr[:,0])))
                for k in range(0,len(D2dr[:,0])):
                    pos=D2dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D2dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D2[:,0])>0:
                if sum(D2[:,0])<R_ok:
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D2[:,0])
                else:
                    Rus_k=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=solmmf_par(D2[:,0],R_ok,n2-len(D2dr[:,0]))
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D2[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
                    n3=0
                    n4=0
        if j==2 and n3>0:
            if len(D3dr[:,0])>0:
                Rus_k=numpy.zeros((len(D3dr[:,0])))
                for k in range(0,len(D3dr[:,0])):
                    pos=D3dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D3dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D3[:,0])>0:
                if sum(D3[:,0])<R_ok:
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D3[:,0])
                else:
                    soldrf=numpy.zeros((n3-len(D3dr[:,0])))
                    soldrf=solmmf_par(D3[:,0],R_ok,n3-len(D3dr[:,0]))
                    Rus_k=numpy.zeros((n3-len(D3dr[:,0])))
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D3[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
                    n4=0

        if j==3 and n4>0:
            if len(D4dr[:,0])>0:
                Rus_k=numpy.zeros((len(D4dr[:,0])))
                for k in range(0,len(D4dr[:,0])):
                    pos=D4dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D4dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D4[:,0])>0:
                if sum(D4[:,0])<R_ok:
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D4[:,0])
                else:
                    soldrf=numpy.zeros((n4-len(D4dr[:,0])))
                    soldrf=solmmf_par(D4[:,0],R_ok,n4-len(D4dr[:,0]))
                    Rus_k=numpy.zeros((n4-len(D4dr[:,0])))
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D4[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
    for j in range(0,len(pr)):  
        x[j]=a[j]
    
    return(x)   
    
    
    
    
 
    
    
def pra2_mood_d_pr(D,R,pr):
    
    
    perc=numpy.zeros((3))
    x=numpy.zeros((5,1))
    x1=numpy.zeros((5,1))
    x2=numpy.zeros((5,1))
    share=numpy.zeros((5,3))
    cong=numpy.zeros((2))
    for k in range(0,3): 
        perc[k]=sum(D[:,k])/R[k]  
    if perc[0]>1:         
        x1=allocmood_pr(D[:,0],R[0],pr)
    else:
        x1=[1,1,1,1,1]
    if max( perc[1],perc[2])>1: 
        x2=drf_pr(D[:,1:3],R[1:3],pr)
    else:
        x2=[1,1,1,1,1]
    cong=[sum(D[:,0])/R[0],max(sum(D[:,1])/R[1],sum(D[:,2])/R[2]) ]    
    for i in range(0,5):
        share[i,:]=(D[i,0]/R[0],D[i,1]/R[1],D[i,2]/R[2])
    if cong[0]>cong[1]:
        x=sol_mood_pra_pr(share, D,R,x1,x2,pr)
    else:
        x=sol_drf_pra_pr(share, D,R,x1,x2,pr)
    return(x)   
    
    
def sol_mood_pra_pr(share, D,R,x1,x2,pr):
    x=numpy.zeros((5))
    p=numpy.zeros((0,1))
    agg=numpy.zeros((1,1))
    drf=numpy.zeros((0,1))
    pos=0
    pos2=0
    for i in range(0,5):
        if share[i,0]>max(share[i,1],share[i,2]):
           # agg[0,0]=i
            p=numpy.insert(p,pos,  i)
            pos=pos+1
        else:
           # agg[0,0]=i
            drf=numpy.insert(drf,pos2,  i)
            pos2=pos2+1
    p=p.astype(int)
    drf=drf.astype(int)
    if sum(D[:,0])>R[0]:
        soluz_prop_n=solmmf_mood_pr(D[:,0],R[0],pr,p,drf,x2)
        for i in range(0,5):
            x[i]=soluz_prop_n[i]
    else:
         for i in range(0,5):
            x[i]=1
    return(x)    
    
    
    
def solmmf_mood_pr(D,R_ok,pr,p,drf,x2):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    D1dr=numpy.zeros((0,2))
    D2dr=numpy.zeros((0,2))
    D3dr=numpy.zeros((0,2))
    D4dr=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((len(pr)))
    n1=0
    n2=0
    n3=0
    n4=0
    n1dr=0
    n2dr=0
    n3dr=0
    n4dr=0
    n1p=0
    n2p=0
    n3p=0
    n4p=0
    for j in range(0,len(pr)):
        if numpy.isin(j,p):
            inser=[D[j],j]
            if pr[j]==1:
                D1=numpy.insert(D1, n1p, inser,  axis=0)
                n1=n1+1
                n1p=n1p+1
            if pr[j]==2:
                D2=numpy.insert(D2, n2p, inser,  axis=0)
                n2=n2+1
                n2p=n2p+1
            if pr[j]==3:
                D3=numpy.insert(D3, n3p, inser,  axis=0)
                n3=n3+1
                n3p=n3p+1
            if pr[j]==4:
                D4=numpy.insert(D4, n4p, inser,  axis=0)
                n4=n4+1
                n4p=n4p+1
        else:
            inser=[D[j],j]
            if pr[j]==1:
                D1dr=numpy.insert(D1dr, n1dr, inser,  axis=0)
                n1=n1+1
                n1dr=n1dr+1
            if pr[j]==2:
                D2dr=numpy.insert(D2dr, n2dr, inser,  axis=0)
                n2=n2+1
                n2dr=n2dr+1
            if pr[j]==3:
                D3dr=numpy.insert(D3dr, n3dr, inser,  axis=0)
                n3=n3+1
                n3dr=n3dr+1
            if pr[j]==4:
                D4dr=numpy.insert(D4dr, n4dr, inser,  axis=0)
                n4=n4+1
                n4dr=n4dr+1
    a=numpy.zeros((len(pr)))
    for j in range(0,4):
        if j==0 and n1>0:
            
            if len(D1dr[:,0])>0:
                Rus_k=numpy.zeros((len(D1dr[:,0])))
                for k in range(0,len(D1dr[:,0])):
                    pos=D1dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D1dr[k,0]*a[pos]                
                R_ok=R_ok-sum(Rus_k)
            if len(D1[:,0])>0:
                if sum(D1[:,0])<R_ok:
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D1[:,0])
                else:
                    soldrf=numpy.zeros((n1-len(D1dr[:,0])))
                    soldrf=solmood_par(D1[:,0],R_ok,n1-len(D1dr[:,0]))
                    Rus_k=numpy.zeros((n1-len(D1dr[:,0])))
                    for k in range(0,n1-len(D1dr[:,0])):
                        pos=D1[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D1[k,0]*a[pos]                   
                    R_ok=R_ok-sum(Rus_k)
                    n2=0
                    n3=0
                    n4=0
        if j==1 and n2>0:
            if len(D2dr[:,0])>0:
                Rus_k=numpy.zeros((len(D2dr[:,0])))
                for k in range(0,len(D2dr[:,0])):
                    pos=D2dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D2dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D2[:,0])>0:
                if sum(D2[:,0])<R_ok:
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D2[:,0])
                else:
                    Rus_k=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=numpy.zeros((n2-len(D2dr[:,0])))
                    soldrf=solmood_par(D2[:,0],R_ok,n2-len(D2dr[:,0]))
                    for k in range(0,n2-len(D2dr[:,0])):
                        pos=D2[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D2[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
                    n3=0
                    n4=0
        if j==2 and n3>0:
            if len(D3dr[:,0])>0:
                Rus_k=numpy.zeros((len(D3dr[:,0])))
                for k in range(0,len(D3dr[:,0])):
                    pos=D3dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D3dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D3[:,0])>0:
                if sum(D3[:,0])<R_ok:
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D3[:,0])
                else:
                    soldrf=numpy.zeros((n3-len(D3dr[:,0])))
                    soldrf=solmood_par(D3[:,0],R_ok,n3-len(D3dr[:,0]))
                    Rus_k=numpy.zeros((n3-len(D3dr[:,0])))
                    for k in range(0,n3-len(D3dr[:,0])):
                        pos=D3[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D3[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
                    n4=0

        if j==3 and n4>0:
            if len(D4dr[:,0])>0:
                Rus_k=numpy.zeros((len(D4dr[:,0])))
                for k in range(0,len(D4dr[:,0])):
                    pos=D4dr[k,1]
                    pos=pos.astype(int)
                    a[pos]=x2[pos]
                    Rus_k[k]=D4dr[k,0]*a[pos]
                R_ok=R_ok-sum(Rus_k)
            if len(D4[:,0])>0:
                if sum(D4[:,0])<R_ok:
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,1]
                        pos=pos.astype(int)
                        a[pos]=1
                    R_ok=R_ok-sum(D4[:,0])
                else:
                    soldrf=numpy.zeros((n4-len(D4dr[:,0])))
                    soldrf=solmood_par(D4[:,0],R_ok,n4-len(D4dr[:,0]))
                    Rus_k=numpy.zeros((n4-len(D4dr[:,0])))
                    for k in range(0,n4-len(D4dr[:,0])):
                        pos=D4[k,1]
                        pos=pos.astype(int)
                        a[pos]=soldrf[k]
                        Rus_k[k]=D4[k,0]*a[pos]
                    R_ok=R_ok-sum(Rus_k)
    for j in range(0,len(pr)):  
        x[j]=a[j]
    
    return(x)   
    
    
    
    
def allocmood_pr_up(D,R_ok,pr,x1):
    D1=numpy.zeros((0,2))
    D2=numpy.zeros((0,2))
    D3=numpy.zeros((0,2))
    D4=numpy.zeros((0,2))
    inser=numpy.zeros((1,2))
    x=numpy.zeros((5))
    x1_1=numpy.zeros((0,1))
    x1_2=numpy.zeros((0,1))
    x1_3=numpy.zeros((0,1))
    x1_4=numpy.zeros((0,1))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,5):
        inser=[D[j],j]
        if pr[j]==1:
            D1=numpy.insert(D1, n1, inser,  axis=0)
            x1_1=numpy.insert(x1_1, n1, x1[j],  axis=0)
            n1=n1+1
        if pr[j]==2:
            D2=numpy.insert(D2, n2, inser,  axis=0)
            x1_2=numpy.insert(x1_2, n2, x1[j],  axis=0)
            n2=n2+1
        if pr[j]==3:
            D3=numpy.insert(D3, n3, inser,  axis=0)
            x1_3=numpy.insert(x1_3, n3, x1[j],  axis=0)
            n3=n3+1
        if pr[j]==4:
            D4=numpy.insert(D4, n4, inser,  axis=0)
            x1_4=numpy.insert(x1_4, n4, x1[j],  axis=0)
            n4=n4+1
            
    a=numpy.zeros((5))
    a_sol=numpy.zeros((5))

    for j in range(0,4):
        if j==0 and n1>0:
            if sum(D1[:,0])<R_ok:
                a=alloc1_up_1ris(D1,R_ok,x1,n1)
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n1))
                soldrf=mood_up_n(D1,R_ok,x1,n1)
                for k in range(0,n1):
                    pos=D1[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D2,R_ok,x1,n2)
                for i in range(0,5):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n2))
                soldrf=mood_up_n(D2,R_ok,x1,n2)
                for k in range(0,n2):
                    pos=D2[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n3=0
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D3,R_ok,x1,n3)
                for i in range(0,5):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n3))
                soldrf=mood_up_n(D3,R_ok,x1,n3)
                for k in range(0,n3):
                    pos=D3[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                n4=0
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a_sol[k]
                R_ok=R_ok-sum(Rus_k)
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok:
                a_sol=alloc1_up_1ris(D4,R_ok,x1,n4)
                for i in range(0,5):
                    if a_sol[i]>0:
                        a[i]=a_sol[i]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k[k]=D[k]*a[k]
                R_ok=R_ok-sum(Rus_k)
            else:
                soldrf=numpy.zeros((n4))
                soldrf=mood_up_n(D4,R_ok,x1,n4)
                for k in range(0,n4):
                    pos=D4[k,1]
                    pos=pos.astype(int)
                    a[pos]=soldrf[pos]
                Rus_k=numpy.zeros((5))
                for k in range(0,5):
                    Rus_k=D[k]*a[k]
               # R_ok=R_ok-sum(Rus_k)
           
    for j in range(0,5):  
        x[j]=a[j]
        
    return(x)
    

def mood_up_n(D,R,x_max,n_ok):
    x_up=numpy.zeros((5))
    for k in range(0,5):
        x_up[k]=x_max[k]
    x=numpy.zeros((n_ok))
    x=solmood_par(D[:,0], R,n_ok)
    x_f=numpy.zeros((5))
    for k in range(0,n_ok):
        pos=D[k,1]
        pos=pos.astype(int)
        x_f[pos]=x[k] 
    x_c=numpy.zeros((6,1))
    x_c=controllo(x_f,x_up)
    
    while x_c[5,0]==1:
        
        s=0
        t=0
        D_ok=numpy.zeros((0,2))
        x_m=numpy.zeros((0,1))
        R_ok=numpy.zeros((1,1))
        rus=numpy.zeros((0,1))
        for i in range(0,n_ok):
            pos=D[i,1]
            pos=pos.astype(int)
            if x_c[pos]==0:
                D_ok=numpy.insert(D_ok, s, D[i,:],  axis=0)
                x_m=numpy.insert(x_m, s,x_max[i],  axis=0)
                s=s+1
            else:
                rus=numpy.insert(rus, t,x_c[i]*D[i,0],  axis=0)
                t=t+1
        R_ok=R-sum(rus[:,0])
        n=s
        x=numpy.zeros((n))
        x=solmood_par(D_ok[:,0], R_ok,n)
        
        cont=0
        x_f=numpy.zeros((5))
        for i in range (0,5):
            if x_c[i]==0:
                for k in range(0,n):
                    pos=D_ok[k,1]
                    pos=pos.astype(int)
                    x_f[pos]=x[k] 
                    cont=cont+1
            else:
                x_f[i]=x_c[i]
                
        x_c=numpy.zeros((6,1))
        x_c=controllo(x_f,x_up)
                   
 #   for i in range(0,n_ok):
  #      x_rest[i]=x_f[i]    
    
    return(x_f)
