# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:25:17 2022

@author: kavdnhen
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.stats import lognorm
from scipy.stats import beta
import scipy.special

def search_lambda4_vect(h1,D,hs,n,interval):
    def root(x):
        return np.multiply(x,np.tan(np.multiply(x*h1,1/(np.power(np.matrix(D),0.5).T*np.ones((1,n)))))) - np.divide(np.matrix(hs).T*np.ones((1,n)),np.power(np.matrix(D),0.5).T*np.ones((1,n)))
        #return  x*np.tan(x*h1/(np.power(np.matrix(D),0.5).T*np.ones((1,n)))) 
    
    ones = np.ones((1,len(D)))
    #ones = np.ones((1,len(D)))
    numbers = np.linspace(0,n-1,n)
    min_intervals = ones.T*np.matrix((numbers))*np.pi/(h1/np.power(np.matrix(D),0.5).T*np.ones((1,n)))+1e-12
    max_intervals = ones.T*np.matrix((numbers+0.5))*np.pi/(h1/np.power(np.matrix(D),0.5).T*np.ones((1,n)))-1e-12
    
    '''
    j_min = np.floor(np.matrix(A).T*np.matrix(numbers)*0.5-0.5-0.5*ones.T*np.matrix(np.remainder(numbers,2)))
    j_max = np.ceil(np.matrix(A).T*np.matrix(numbers*0.5+0.5)-0.5*ones.T*np.matrix(np.remainder(numbers,2)))
    oplossing = j_max - j_min - 1

    interval_min_actual = np.where(oplossing == 1, min_intervals, 1e6)
    interval_max_actual = np.where(oplossing == 1, max_intervals, 1e6)
    '''
    for j in range(interval):
        left_value = root(min_intervals)
        right_value = root(max_intervals)
        mid_values = root(0.5*(min_intervals+max_intervals))
        interval_min_actual2 = np.where(np.sign(left_value) == np.sign(mid_values), 0.5*(min_intervals+max_intervals), min_intervals)                     
        interval_max_actual2 = np.where(np.sign(right_value) == np.sign(mid_values), 0.5*(min_intervals+max_intervals), max_intervals)
        min_intervals = interval_min_actual2
        max_intervals = interval_max_actual2
    
    return min_intervals

def integral(D,lam,h1,hs):
    term1 = -2*D*lam*hs*np.power(np.cos(h1*lam/np.power(D,0.5)),2)
    term2 = -np.sin(h1*lam/np.power(D,0.5))*(-np.power(lam,2)*np.power(D,1.5)+np.power(hs,2)*np.power(D,0.5))*np.cos(h1*lam/np.power(D,0.5))
    term3 = lam*(D*h1*np.power(lam,2)+h1*np.power(hs,2)+2*D*hs)
    return 1/(2*np.power(hs,2)*lam)*(term1+term2+term3)

def int_erf_AX_sin_Bx(A,B,x):
    T1 = -1/B*np.cos(B*x)*scipy.special.erf(A*x) + 1/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*np.real(scipy.special.erf(A*x-B/(2*A)*1j)+scipy.special.erf(A*x+B/(2*A)*1j))
    T2 = -1/B*np.cos(B*0)*scipy.special.erf(A*0) + 1/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*np.real(scipy.special.erf(A*0-B/(2*A)*1j)+scipy.special.erf(A*0+B/(2*A)*1j))
    return T1-T2
    
def int_erf_AX_cos_Bx(A,B,x):
    T1 = 1/B*np.sin(B*x)*scipy.special.erf(A*x) + np.real(1j/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*(scipy.special.erf(A*x-B/(2*A)*1j)-scipy.special.erf(A*x+B/(2*A)*1j)))
    T2 = 1/B*np.sin(B*0)*scipy.special.erf(A*0) + np.real(1j/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*(scipy.special.erf(A*0-B/(2*A)*1j)-scipy.special.erf(A*0+B/(2*A)*1j)))
    return T1-T2

def total_integral(Dc_prior,D_post,hs,lam,h,t,Cs):
    A = 1/(2*np.power(Dc_prior*t,0.5))
    B = lam/np.power(D_post,0.5)
    termA = -Cs*int_erf_AX_sin_Bx(A,B,h)
    termB = -Cs*np.power(D_post,0.5)*lam/hs*int_erf_AX_cos_Bx(A,B,h)
    return termA + termB

def fn(x,D,lam,hs):
    return np.sin(lam*x/np.power(D,0.5))+np.power(D,0.5)/hs*lam*np.cos(lam*x/np.power(D,0.5))

def Coating(cover,h1,D_before,D_after,hs,trep,T,Cs,nmax,interval):

    som = Cs.copy()

    lambdas = search_lambda4_vect(h1,D_after,hs,nmax,interval)
  
    Dc_prior = np.tile(D_before[:,np.newaxis],(1,nmax))
    D22 = np.tile(D_after[:,np.newaxis],(1,nmax))
    hs_ext = np.tile(hs[:,np.newaxis],(1,nmax))
    Cs_extend = np.tile(Cs[:,np.newaxis],(1,nmax))
    BB = total_integral(Dc_prior,D22,hs_ext,lambdas,h1,trep,Cs_extend).T

    D1_ext = np.tile(D_after[:,np.newaxis],(1,nmax))
    hs_ext = np.tile(hs[:,np.newaxis],(1,nmax))
    integral_term = integral(D1_ext,lambdas,h1,hs_ext).T

    Term2 = fn(np.tile(cover,(nmax,1)),np.tile(D_after,(nmax,1)),lambdas.T,np.tile(hs,(nmax,1)))
    
    som = Cs + np.sum(BB/integral_term*Term2*np.exp(-np.power(lambdas.T,2)*T),axis=0)
    chlorides = np.maximum(som,0.0)
    
    return chlorides

def Coating_PDep(cover,h1,D_before,D_after,hs,trep,T,Cs,nmax,interval,crit):
    chlorides = Coating(cover,h1,D_before,D_after,hs,trep,T,Cs,nmax,interval)
    return sum(np.where(chlorides>crit,1,0))/len(chlorides)

def Coating_over_time(cover,h1,D_before,D_after,hs,trep,T,Cs,nmax,interval):

    som = Cs.copy()
    
    lambdas = search_lambda4_vect(h1,D_after,hs,nmax,interval)

    Dc_prior = np.tile(D_before[:,np.newaxis],(1,nmax))
    D22 = np.tile(D_after[:,np.newaxis],(1,nmax))
    hs_ext = np.tile(hs[:,np.newaxis],(1,nmax))
    Cs_extend = np.tile(Cs[:,np.newaxis],(1,nmax))
    BB = total_integral(Dc_prior,D22,hs_ext,lambdas,h1,trep,Cs_extend).T

    D1_ext = np.tile(D_after[:,np.newaxis],(1,nmax))
    hs_ext = np.tile(hs[:,np.newaxis],(1,nmax))
    integral_term = integral(D1_ext,lambdas,h1,hs_ext).T

    Term2 = fn(np.tile(cover,(nmax,1)),np.tile(D_after,(nmax,1)),lambdas.T,np.tile(hs,(nmax,1)))
    
    res_list = []
    
    for tt in range(1,T):
    
        som = Cs + np.sum(BB/integral_term*Term2*np.exp(-np.power(lambdas.T,2)*tt),axis=0)
        chlorides = np.maximum(som,0.0)
        
        res_list.append(chlorides)
        
    return res_list

def Coating_PDep_over_time(cover,h1,D_before,D_after,hs,trep,T,Cs,nmax,interval,crit):
    res_list = Coating_over_time(cover,h1,D_before,D_after,hs,trep,T,Cs,nmax,interval)
    Pdep_list = []
    #print(res_list)
    for res in res_list:
        Pdep = sum(np.where(res>crit,1,0))/len(cover)
        Pdep_list.append(Pdep)
    return Pdep_list


'''
#Example calculation Probability of depassivation
NoS = int(1)
trep = 20
R_LT = 10

m_a = 0.3
s_a = 0.12
a_a = 0
b_a = 1
       
m_D_RCM = 5
s_D_RCM = 0.2*m_D_RCM
m_D_repair = 0.1#1.27
s_D_repair = 0.2*m_D_repair

m_Cs = 2
s_Cs = 0.75*m_Cs
        
m_cover = 40
s_cover = 8 #mm
        
m_crit = 0.6
b_crit = 1
a_crit = 0
s_crit = 0.2

mortel = 10
nx = 300

#Generate samples
s=np.random.rand(int(1e4),6)

s = np.array(s)
        
m_a_scale = m_a/b_a
s_a_scale = s_a/b_a
alpha = ( (1-m_a_scale)/s_a_scale**2 - 1/m_a_scale )*m_a_scale**2
betaa = alpha*(1/m_a_scale -1)
a_samples = beta.ppf(s[:,0],alpha,betaa,scale=b_a)

sigmalnDconcr=(np.log((s_D_RCM/m_D_RCM)**2+1))**(1/2)
mulnDconcr=np.log(m_D_RCM)-1/2*sigmalnDconcr**2
samplesDconcr= lognorm.ppf(s[:,1],scale=np.exp(mulnDconcr),s=sigmalnDconcr)

sigmalnDmort=(np.log((s_D_repair/m_D_repair)**2+1))**(1/2)
mulnDmort=np.log(m_D_repair)-1/2*sigmalnDmort**2
samplesDmort= lognorm.ppf(s[:,2],scale=np.exp(mulnDmort),s=sigmalnDmort)

sigmalnCs=(np.log((s_Cs/m_Cs)**2+1))**(1/2)
mulnCs=np.log(m_Cs)-1/2*sigmalnCs**2
samplesCs = lognorm.ppf(s[:,3],scale=np.exp(mulnCs),s=sigmalnCs)

sigmalncover=(np.log((s_cover/m_cover)**2+1))**(1/2)
mulncover=np.log(m_cover)-1/2*sigmalncover**2
samplescover = lognorm.ppf(s[:,4],scale=np.exp(mulncover),s=sigmalncover)

m_crit_scale = m_crit/(b_crit-a_crit)
s_crit_scale = s_crit/(b_crit-a_crit)
alpha = ( (1-m_crit_scale)/s_crit_scale**2 - 1/m_crit_scale )*m_crit_scale**2
betaa = alpha*(1/m_crit_scale -1)
crit_samples = beta.ppf(s[:,5],alpha,betaa,scale=(b_crit-a_crit),loc=a_crit)

a = Coating(samplescover,300,samplesDconcr,samplesDconcr,samplesDmort,20,50,samplesCs,10,10)
print(a)
'''