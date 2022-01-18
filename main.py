
   
import sys
print(sys.path)
sys.path.append('/home/francesca/Desktop/sim_dec')
sys.path.append('/home/francesca/Desktop/sim_dec')

import random 
import numpy 
import matplotlib.pyplot as plt
import math
from functions import * 
import numpy as np 

import pandas as pd
mydata= pd.read_table('/home/francesca/Desktop/sim_dec/set_noms.txt',header = None)
data_pr= pd.read_table('/home/francesca/Desktop/sim_dec/set_n_comp.txt',header = None)
R=pd.read_table('/home/francesca/Desktop/sim_dec/R_noms.txt',header = None)



print(mydata)
print(R)
mydata_imp = np.array(mydata, dtype=np.float)
data_pr_imp = np.array(data_pr, dtype=np.float)
R_imp=np.array(R, dtype=np.float)
n=5

mydata=np.vstack([mydata_imp[:,2],mydata_imp[:,0], mydata_imp[:,1]])
mydata=np.transpose(mydata)
data_pr=np.vstack([data_pr_imp[:,2],data_pr_imp[:,0], data_pr_imp[:,1], data_pr_imp[:,3]])
data_pr=np.transpose(data_pr)
R=np.vstack([R_imp[:,2],R_imp[:,0], R_imp[:,1]])
R=np.transpose(R)



######centralized

x_cent=numpy.zeros((1000,1))

for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    a=numpy.zeros((5))    
    a=drf3(D,R_ok)
    for j in range(0,5): 
        x_cent[5*(i)+j]=a[j]
        
np.savetxt('/home/francesca/Desktop/sim_dec/x_cent.txt',x_cent,fmt='%.2f')        

#salvamat(x_cent,'/home/francesca/Desktop/sim_dec' )

######CRA
x=numpy.zeros((1000,1))
x_mmf=numpy.zeros((1000,1))
x_mood=numpy.zeros((1000,1))

for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    a=numpy.zeros((5))    
    a=cra_p_d(D,R_ok)
    for j in range(0,5):  #####prop+drf
        x[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=cra_mmf_d(D,R_ok)
    for j in range(0,5):  ######mmf+drf
        x_mmf[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=cra_mood_d(D,R_ok)
    for j in range(0,5):  ######mood+drf
        x_mood[5*(i)+j]=a[j]
        
        
np.savetxt('/home/francesca/Desktop/sim_dec/x_cra_p.txt',x,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_cra_mmf.txt',x_mmf,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_cra_mood.txt',x_mood,fmt='%.2f')  

######OCRA
x_o=numpy.zeros((1000,1))
x_mmf_o=numpy.zeros((1000,1))
x_mood_o=numpy.zeros((1000,1))        
reallo_pr=0
reallo_mmf=0
reallo_mood=0
for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    a=numpy.zeros((5))    
    a=ocra_p_d(D,R_ok)[0]
    reallo_pr=ocra_p_d(D,R_ok)[1]+reallo_pr
    for j in range(0,5):  #####prop+drf
        x_o[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=ocra_mmf_d(D,R_ok)[0]
    reallo_mmf=ocra_mmf_d(D,R_ok)[1]+reallo_mmf
    for j in range(0,5):  ######mmf+drf
        x_mmf_o[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=ocra_mood_d(D,R_ok)[0]
    reallo_mood=ocra_mood_d(D,R_ok)[1]+reallo_mood
    for j in range(0,5):  ######mood+drf
        x_mood_o[5*(i)+j]=a[j]
        
np.savetxt('/home/francesca/Desktop/sim_dec/x_ocra_p.txt',x_o,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_ocra_mmf.txt',x_mmf_o,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_ocra_mood.txt',x_mood_o,fmt='%.2f')  
        

        
######PRA-1
x_pra=numpy.zeros((1000,1))
x_mmf_pra=numpy.zeros((1000,1))
x_mood_pra=numpy.zeros((1000,1))        
reallo_pr=0
reallo_mmf=0
reallo_mood=0

for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]    
    a=numpy.zeros((5))    
    a=pra_p_d(D,R_ok)
    for j in range(0,5):  #####prop+drf
        x_pra[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra_mmf_d(D,R_ok)
    for j in range(0,5):  ######mmf+drf
        x_mmf_pra[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra_mood_d(D,R_ok)
    for j in range(0,5):  ######mood+drf
        x_mood_pra[5*(i)+j]=a[j]

np.savetxt('/home/francesca/Desktop/sim_dec/x_pra_p.txt',x_pra,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra_mmf.txt',x_mmf_pra,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra_mood.txt',x_mood_pra,fmt='%.2f')  
      
         


       
######PRA-2
x_p_p=numpy.zeros((1000,1))
x_mmf_p=numpy.zeros((1000,1))
x_mood_p=numpy.zeros((1000,1))        
reallo_pr=0
reallo_mmf=0
reallo_mood=0
for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    a=numpy.zeros((5))    
    a=pra2_p_d2(D,R_ok)
    for j in range(0,5):  #####prop+drf
        x_p_p[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra2_mmf_d2(D,R_ok)
    for j in range(0,5):  ######mmf+drf
        x_mmf_p[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra2_mood_d2(D,R_ok)
    for j in range(0,5):  ######mood+drf
        x_mood_p[5*(i)+j]=a[j]
        
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra2_p.txt',x_p_p,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra2_mmf.txt',x_mmf_p,fmt='%.2f') 
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra2_mood.txt',x_mood_p,fmt='%.2f')  




###########with  prior

######centralized

x_cent=numpy.zeros((1000,1))

for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    pr=data_pr[0+5*(i):5*(i+1),3]
    D1=numpy.zeros((0,4))
    D2=numpy.zeros((0,4))
    D3=numpy.zeros((0,4))
    D4=numpy.zeros((0,4))
    inser=numpy.zeros((1,4))
    n1=0
    n2=0
    n3=0
    n4=0
    for j in range(0,5):
        inser=[D[j,0],D[j,1],D[j,2],j]
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
            if sum(D1[:,0])<R_ok[0] and sum(D1[:,1])<R_ok[1] and sum(D1[:,2])<R_ok[2]:
                for k in range(0,n1):
                    pos=D1[k,3]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=[R_ok[0]-sum(D1[:,0]),R_ok[1]-sum(D1[:,1]),R_ok[2]-sum(D1[:,2]) ]
            else:
                soldrf=numpy.zeros((n1))
                soldrf=solpar_3(D1,R_ok)
                for k in range(0,n1):
                    pos=D1[k,3]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                Rus_k=numpy.zeros((5,3))
                for k in range(0,5):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k], D[k,2]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1]), R_ok[2]-sum(Rus_k[:,2])]
                n2=0
                n3=0
                n4=0
        if j==1 and n2>0:
            if sum(D2[:,0])<R_ok[0] and sum(D2[:,1])<R_ok[1] and sum(D2[:,2])<R_ok[2]:
                for k in range(0,n2):
                    pos=D2[k,3]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=[R_ok[0]-sum(D2[:,0]),R_ok[1]-sum(D2[:,1]),R_ok[2]-sum(D2[:,2]) ]
            else:
                soldrf=numpy.zeros((n2))
                soldrf=solpar_3(D2,R_ok)
                for k in range(0,n2):
                    pos=D2[k,3]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                n3=0
                n4=0
                Rus_k=numpy.zeros((5,3))
                for k in range(0,5):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k], D[k,2]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1]), R_ok[2]-sum(Rus_k[:,2])]
        if j==2 and n3>0:
            if sum(D3[:,0])<R_ok[0] and sum(D3[:,1])<R_ok[1] and sum(D3[:,2])<R_ok[2]:
                for k in range(0,n3):
                    pos=D3[k,3]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=[R_ok[0]-sum(D3[:,0]),R_ok[1]-sum(D3[:,1]),R_ok[2]-sum(D3[:,2]) ]
            else:
                soldrf=numpy.zeros((n3))
                soldrf=solpar_3(D3,R_ok)
                for k in range(0,n3):
                    pos=D3[k,3]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k] 
                n4=0
                Rus_k=numpy.zeros((5,3))
                for k in range(0,5):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k], D[k,2]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1]), R_ok[2]-sum(Rus_k[:,2])]
        if j==3 and n4>0:
            if sum(D4[:,0])<R_ok[0] and sum(D4[:,1])<R_ok[1] and sum(D4[:,2])<R_ok[2]:
                for k in range(0,n4):
                    pos=D4[k,3]
                    pos=pos.astype(int)
                    a[pos]=1
                R_ok=[R_ok[0]-sum(D4[:,0]),R_ok[1]-sum(D4[:,1]),R_ok[2]-sum(D4[:,2]) ]
            else:
                soldrf=numpy.zeros((n4))
                soldrf=solpar_3(D4,R_ok)
                for k in range(0,n4):
                    pos=D4[k,3]
                    pos=pos.astype(int)
                    a[pos]=soldrf[k]
                j=4
                Rus_k=numpy.zeros((5,3))
                for k in range(0,5):
                    Rus_k[k,:]=[D[k,0]*a[k], D[k,1]*a[k], D[k,2]*a[k]]
                R_ok=[R_ok[0]-sum(Rus_k[:,0]), R_ok[1]-sum(Rus_k[:,1]), R_ok[2]-sum(Rus_k[:,2])]
    for j in range(0,5):  
        x_cent[5*(i)+j]=a[j]

np.savetxt('/home/francesca/Desktop/sim_dec/x_cent_pr.txt',x_cent,fmt='%.2f') 


######CRA
x=numpy.zeros((1000,1))
x_mmf=numpy.zeros((1000,1))
x_mood=numpy.zeros((1000,1))

for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    pr=data_pr[0+5*(i):5*(i+1),3]
    a=numpy.zeros((5))    
    a=priority_cra_p_d(D,R_ok,pr)
    for j in range(0,5):  #####prop+drf
        x[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=priority_cra_mmf_d(D,R_ok,pr)
    for j in range(0,5):  ######mmf+drf
        x_mmf[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=priority_cra_mood_d(D,R_ok,pr)
    for j in range(0,5):  ######mood+drf
        x_mood[5*(i)+j]=a[j]
    
np.savetxt('/home/francesca/Desktop/sim_dec/x_cra_p_pr.txt',x,fmt='%.2f')  
np.savetxt('/home/francesca/Desktop/sim_dec/x_cra_mmf_pr.txt',x_mmf,fmt='%.2f')
np.savetxt('/home/francesca/Desktop/sim_dec/x_cra_mood_pr.txt',x_mood,fmt='%.2f')    



######OCRA
x_o=numpy.zeros((1000,1))
x_mmf_o=numpy.zeros((1000,1))
x_mood_o=numpy.zeros((1000,1))        
reallo_pr=0
reallo_mmf=0
reallo_mood=0
for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    pr=data_pr[0+5*(i):5*(i+1),3]
    a=numpy.zeros((5))    
    a=ocra_p_d_pr(D,R_ok,pr)[0]
    reallo_pr=ocra_p_d_pr(D,R_ok,pr)[1]+reallo_pr
    for j in range(0,5):  #####prop+drf
        x_o[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=ocra_mmf_d_pr(D,R_ok,pr)[0]
    reallo_mmf=ocra_mmf_d_pr(D,R_ok,pr)[1]+reallo_mmf
    for j in range(0,5):  ######mmf+drf
        x_mmf_o[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=ocra_mood_d_pr(D,R_ok,pr)[0]
    reallo_mood=ocra_mood_d_pr(D,R_ok,pr)[1]+reallo_mood
    for j in range(0,5):  ######mood+drf
        x_mood_o[5*(i)+j]=a[j]

np.savetxt('/home/francesca/Desktop/sim_dec/x_cora_p_pr.txt',x_o,fmt='%.2f')  
np.savetxt('/home/francesca/Desktop/sim_dec/x_ocra_mmf_pr.txt',x_mmf_o,fmt='%.2f')
np.savetxt('/home/francesca/Desktop/sim_dec/x_ocra_mood_pr.txt',x_mood_o,fmt='%.2f') 

     




######PRA-1
x_pra=numpy.zeros((1000,1))
x_mmf_pra=numpy.zeros((1000,1))
x_mood_pra=numpy.zeros((1000,1))        
reallo_pr=0
reallo_mmf=0
reallo_mood=0
for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    a=numpy.zeros((5))    
    pr=data_pr[0+5*(i):5*(i+1),3]
    a=pra_p_d_pr(D,R_ok,pr)
    for j in range(0,5):  #####prop+drf
        x_pra[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra_mmf_d_pr(D,R_ok,pr)
    for j in range(0,5):  ######mmf+drf
        x_mmf_pra[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra_mood_d_pr(D,R_ok,pr)
    for j in range(0,5):  ######mood+drf
        x_mood_pra[5*(i)+j]=a[j]
        
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra_p_pr.txt',x_pra,fmt='%.2f')  
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra_mmf_pr.txt',x_mmf_pra,fmt='%.2f')
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra_mood_pr.txt',x_mood_pra,fmt='%.2f') 



######PRA-2
x_p_p=numpy.zeros((1000,1))
x_mmf_p=numpy.zeros((1000,1))
x_mood_p=numpy.zeros((1000,1))        
reallo_pr=0
reallo_mmf=0
reallo_mood=0
for i in range(0,200):  
    D=mydata[0+5*(i):5*(i+1),:]
    R_ok=R[i,:]
    a=numpy.zeros((5))    
    pr=data_pr[0+5*(i):5*(i+1),3]
    a=pra2_p_d_pr2(D,R_ok,pr)
    for j in range(0,5):  #####prop+drf
        x_p_p[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra2_mmf_d_pr2(D,R_ok,pr)
    for j in range(0,5):  ######mmf+drf
        x_mmf_p[5*(i)+j]=a[j]
    a=numpy.zeros((5))    
    a=pra2_mood_d_pr2(D,R_ok,pr)
    for j in range(0,5):  ######mood+drf
        x_mood_p[5*(i)+j]=a[j]

np.savetxt('/home/francesca/Desktop/sim_dec/x_pra2_p_pr.txt',x_p_p,fmt='%.2f')  
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra2_mmf_pr.txt',x_mmf_p,fmt='%.2f')
np.savetxt('/home/francesca/Desktop/sim_dec/x_pra2_mood_pr.txt',x_mood_p,fmt='%.2f') 
        
        
