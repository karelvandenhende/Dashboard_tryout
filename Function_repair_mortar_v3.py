import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.stats import lognorm
from scipy.stats import beta
import scipy.special

def smallest_elements(matrix, M):
    result = []
    for row in matrix:
        sorted_row = sorted(row)
        result.append(sorted_row[:M])
    return result

def search_lambda5_vect(P1,P2,D1,D2,n,interval):
    CC = np.matrix(np.power(D1/D2,0.5)).T*np.ones((1,n+1))
    def root(x,AA):
        return np.multiply(np.tan(x),np.tan(np.multiply(x,AA))) - CC

    
    ones = np.ones((1,len(P1)))
    ones2 = np.ones((1,n+1))
    Pa = np.minimum(P1,P2)
    Pb = np.maximum(P1,P2)
    A = (Pa/Pb*ones2.T).T
    numbers_pos = np.linspace(0,n*2,n+1)
    numbers_neg = numbers_pos + 1
    min_intervals_pos = ones.T*np.matrix(numbers_pos*np.pi*0.5) + 0.000001
    max_intervals_pos = ones.T*np.matrix((numbers_pos+1)*np.pi*0.5) - 0.000001
    min_intervals_neg = ones.T*np.matrix(numbers_neg*np.pi*0.5) + 0.000001
    max_intervals_neg = ones.T*np.matrix((numbers_neg+1)*np.pi*0.5) - 0.000001
    
    tan_ax_min = np.tan(np.multiply(min_intervals_pos,A))
    tan_ax_max = np.tan(np.multiply(max_intervals_pos,A))
    

    #CASE 1
    limit_1 = (np.floor(np.multiply(max_intervals_pos,A)/np.pi-0.5)+0.5)*np.pi/A - 0.000001
    #print(limit_1)
    #CASE 2
    limit_2 = (np.floor(np.multiply(max_intervals_pos,A)/np.pi))*np.pi/A + 0.000001
    #print(limit_2)

    interval_min_actual_pos = np.where(tan_ax_min >= 0, 
                                   np.where(tan_ax_max >= 0, 
                                            min_intervals_pos, 
                                            np.where(root(limit_1,A)>0,
                                                     min_intervals_pos,
                                                     1e6)
                                            ),
                                   np.where(tan_ax_max < 0,
                                            1e6,
                                            np.where(root(limit_2,A)<0,
                                                     limit_2,
                                                     1e6
                                                     )
                                            )
                                   )
    
    interval_max_actual_pos = np.where(tan_ax_min >= 0, 
                                   np.where(tan_ax_max >= 0, 
                                            max_intervals_pos, 
                                            np.where(root(limit_1,A)>0,
                                                     limit_1,
                                                     1e6)
                                            ),
                                   np.where(tan_ax_max < 0,
                                            1e6,
                                            np.where(root(limit_2,A)<0,
                                                     max_intervals_pos,
                                                     1e6
                                                     )
                                            )
                                   )
    
    tan_ax_min = np.tan(np.multiply(min_intervals_neg,A))
    tan_ax_max = np.tan(np.multiply(max_intervals_neg,A))
    
    #CASE 1
    limit_2 = (np.floor(np.multiply(max_intervals_neg,A)/np.pi-0.5)+0.5)*np.pi/A + 0.00001

    
    #CASE 2
    limit_1 = (np.floor(np.multiply(max_intervals_neg,A)/np.pi))*np.pi/A - 0.00001
    
    
    
    interval_min_actual_neg = np.where(tan_ax_min <= 0, 
                                   np.where(tan_ax_max <= 0, 
                                            min_intervals_neg, 
                                            np.where(root(limit_1,A)<0,
                                                     min_intervals_neg,
                                                     1e6)
                                            ),
                                   np.where(tan_ax_max > 0,
                                            1e6,
                                            np.where(root(limit_2,A)>0,
                                                     limit_2,
                                                     1e6
                                                     )
                                            )
                                   )
    
    interval_max_actual_neg = np.where(tan_ax_min <= 0, 
                                   np.where(tan_ax_max <= 0, 
                                            max_intervals_neg, 
                                            np.where(root(limit_1,A)<0,
                                                     limit_1,
                                                     1e6)
                                            ),
                                   np.where(tan_ax_max > 0,
                                            1e6,
                                            np.where(root(limit_2,A)>0,
                                                     max_intervals_neg,
                                                     1e6
                                                     )
                                            )
                                   )
    
    interval_min_actual = np. concatenate((interval_min_actual_pos,interval_min_actual_neg),axis=1)#np.where(oplossing == 1, min_intervals, 1e6)
    interval_max_actual = np. concatenate((interval_max_actual_pos,interval_max_actual_neg),axis=1)#np.where(oplossing == 1, max_intervals, 1e6)

    CC = np.matrix(np.power(D1/D2,0.5)).T*np.ones((1,(n+1)*2))
    ones2 = np.ones((1,2*n+2))
    A = (Pa/Pb*ones2.T).T
    for j in range(interval):
        left_value = root(interval_min_actual,A)
        right_value = root(interval_max_actual,A)
        mid_values = root(0.5*(interval_min_actual+interval_max_actual),A)
        interval_min_actual2 = np.where(np.sign(left_value) == np.sign(mid_values), 0.5*(interval_min_actual+interval_max_actual), interval_min_actual)                     
        interval_max_actual2 = np.where(np.sign(right_value) == np.sign(mid_values), 0.5*(interval_min_actual+interval_max_actual), interval_max_actual)
        interval_min_actual = interval_min_actual2
        interval_max_actual = interval_max_actual2
    
    Z = np.array(np.multiply(interval_min_actual,np.matrix((1/Pb)).T*np.ones((1,2*n+2))))
    return np.array(smallest_elements(Z, n))
    
def integral(h1,h2,D1,D2,lam):
    A = lam/np.power(D1,0.5)
    term1 = np.power(1/np.tan(lam*h2/np.power(D2,0.5))/np.sin(lam*h1/np.power(D1,0.5)),2)*(A*h1-np.cos(A*h1)*np.sin(A*h1))/(2*A)
    B = lam/np.power(D2,0.5)
    term2 = np.power(1/np.sin(lam*h2/np.power(D2,0.5)),2)*(B*h2+np.cos(B*h2)*np.sin(B*h2))/(2*B)
    return term1+term2

def int_erf_AX_sin_Bx(A,B,x):
    T1 = -1/B*np.cos(B*x)*scipy.special.erf(A*x) + 1/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*np.real(scipy.special.erf(A*x-B/(2*A)*1j)+scipy.special.erf(A*x+B/(2*A)*1j))
    T2 = -1/B*np.cos(B*0)*scipy.special.erf(A*0) + 1/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*np.real(scipy.special.erf(A*0-B/(2*A)*1j)+scipy.special.erf(A*0+B/(2*A)*1j))
    return T1-T2
    
def int_erf_AX_cos_Bx(A,B,x):
    T1 = 1/B*np.sin(B*x)*scipy.special.erf(A*x) + np.real(1j/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*(scipy.special.erf(A*x-B/(2*A)*1j)-scipy.special.erf(A*x+B/(2*A)*1j)))
    T2 = 1/B*np.sin(B*0)*scipy.special.erf(A*0) + np.real(1j/(2*B)*np.exp(-np.power(B,2)/(4*np.power(A,2)))*(scipy.special.erf(A*0-B/(2*A)*1j)-scipy.special.erf(A*0+B/(2*A)*1j)))
    return T1-T2

def int_second_part2(Dc_prior,D1,D2,lam,h1,h2,t):

    A = 1/(2*np.power(Dc_prior*t,0.5))
    B = lam/np.power(D2,0.5)
    h = h2
    term1 = np.cos(B*h)*int_erf_AX_cos_Bx(A,B,h)
    term2 = np.sin(B*h)*int_erf_AX_sin_Bx(A,B,h)

    return term1 + term2

def total_integral(Dc_prior,D1,D2,lam,h1,h2,t,Cs):
    termA = -Cs/np.tan(lam*h2/np.power(D2,0.5))/np.sin(lam*h1/np.power(D1,0.5))*np.power(D1,0.5)/lam*(1-np.cos(lam*h1/np.power(D1,0.5)))
    termB = -Cs/np.sin(lam*h2/np.power(D2,0.5))*int_second_part2(Dc_prior,D1,D2,lam,h1,h2,t)
    return termA + termB

def fn(x,h1,h2,D1,D2,lam):
    return np.where(x <= h1, 
    1/np.tan(lam*h2/np.power(D2,0.5))/np.sin(lam*h1/np.power(D1,0.5))*np.sin(lam*x/np.power(D1,0.5)),
    1/np.sin(lam*h2/np.power(D2,0.5))*np.cos(lam*(x-h1-h2)/np.power(D2,0.5))         )

def C_paper_vectorized(cover,h1,h2,D1,D2_ini,D2_rep,trep,T,Cs,nmax,interval):

    som = Cs.copy()
        
    lambdas = search_lambda5_vect(h1/np.power(D1,0.5),h2/np.power(D2_rep,0.5),D1,D2_rep,nmax,interval)#*(h1/np.power(D1_single,0.5)))
       
    Dc_prior = np.tile(D2_ini[:,np.newaxis],(1,nmax))
    D11 = np.tile(D1[:,np.newaxis],(1,nmax))
    D22 = np.tile(D2_rep[:,np.newaxis],(1,nmax))
    Cs_extend = np.tile(Cs[:,np.newaxis],(1,nmax))
    BB = total_integral(Dc_prior,D11,D22,lambdas,h1,h2,trep,Cs_extend).T
    
    D1_ext = np.tile(D1[:,np.newaxis],(1,nmax))
    D2_ext = np.tile(D2_rep[:,np.newaxis],(1,nmax))
    integral_term = integral(h1,h2,D1_ext,D2_ext,lambdas).T

    Term2 = fn(np.tile(cover+h1,(nmax,1)),h1,h2,np.tile(D1,(nmax,1)),np.tile(D2_rep,(nmax,1)),lambdas.T)

    som = Cs + np.sum(BB/integral_term*Term2*np.exp(-np.power(lambdas.T,2)*T),axis=0)

    return np.maximum(som,0.0)

def mortar_overlay(cover,h1,h2,D1,D2_ini,D2_rep,trep,T,Cs,Ccr,nmax,interval):
    #plt.plot(C_paper_vectorized(cover,h1,h2,D1,D2,trep,T,Cs,nmax,interval,a_c,b_c,a_m))

    a = C_paper_vectorized(cover,h1,h2,D1,D2_ini,D2_rep,trep,T,Cs,nmax,interval)
    return sum(np.where(a>Ccr,1,0))/len(cover)

def C_paper_vectorized_over_time(cover,h1,h2,D1,D2_ini,D2_rep,trep,T,Cs,nmax,interval):

    som = Cs.copy()
        
    lambdas = search_lambda5_vect(h1/np.power(D1,0.5),h2/np.power(D2_rep,0.5),D1,D2_rep,nmax,interval)#*(h1/np.power(D1_single,0.5)))

    Dc_prior = np.tile(D2_ini[:,np.newaxis],(1,nmax))
    D11 = np.tile(D1[:,np.newaxis],(1,nmax))
    D22 = np.tile(D2_rep[:,np.newaxis],(1,nmax))
    Cs_extend = np.tile(Cs[:,np.newaxis],(1,nmax))
    BB = total_integral(Dc_prior,D11,D22,lambdas,h1,h2,trep,Cs_extend).T
    
    D1_ext = np.tile(D1[:,np.newaxis],(1,nmax))
    D2_ext = np.tile(D2_rep[:,np.newaxis],(1,nmax))
    integral_term = integral(h1,h2,D1_ext,D2_ext,lambdas).T

    Term2 = fn(np.tile(cover+h1,(nmax,1)),h1,h2,np.tile(D1,(nmax,1)),np.tile(D2_rep,(nmax,1)),lambdas.T)

    res_list = []

    for tt in range(1,T):

        som = Cs + np.sum(BB/integral_term*Term2*np.exp(-np.power(lambdas.T,2)*tt),axis=0)
        chlorides = np.maximum(som,0.0)
        
        res_list.append(chlorides)

    return res_list

def mortar_overlay_over_time(cover,h1,h2,D1,D2_ini,D2_rep,trep,T,Cs,Ccr,nmax,interval):
    #plt.plot(C_paper_vectorized(cover,h1,h2,D1,D2,trep,T,Cs,nmax,interval,a_c,b_c,a_m))
    
    res_list = C_paper_vectorized_over_time(cover,h1,h2,D1,D2_ini,D2_rep,trep,T,Cs,nmax,interval)
    
    Pdep_list = []
    for res in res_list:
        Pdep = sum(np.where(res>Ccr,1,0))/len(cover)
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
m_D_repair = 0.5#1.27
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

a = C_paper_vectorized(samplescover,20,300,samplesDmort,samplesDconcr,samplesDconcr,40,50,samplesCs,20,20,0,0)
print(a)
'''
