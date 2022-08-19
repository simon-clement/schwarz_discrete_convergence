import numpy as np
import sys
from scipy.special import exp1

def phi_oh(zet):
    if(zet<0.):
        return (1.-25.*zet)**(-1./3.)
    else:
        return 1+5.*zet

def psi_oh(zet):
    if(zet<0.):
        return -np.sqrt(3)*np.arctan(1./3*np.sqrt(3)*(2*(-25*zet + 1)**(1./3) + 1)) + 3./2*np.log((-25*zet + 1)**(2./3) + (-25*zet + 1)**(1./3) + 1) + np.sqrt(3)*np.arctan(np.sqrt(3)) - 3./2.*np.log(3)
    else:
        return -5*zet
    
def phi_om(zet):
    if(zet<0.):
        return (1-14*zet)**(-1./3.)
    else:
        return 1+5*zet
    
def psi_om(zet):
    if(zet<0.):
        return -np.sqrt(3)*np.arctan(1./3*np.sqrt(3)*(2*(-14*zet + 1)**(1./3) + 1)) + 3./2*np.log((-14*zet + 1)**(2./3) + (-14*zet + 1)**(1./3) + 1) + np.sqrt(3)*np.arctan(np.sqrt(3)) - 3./2.*np.log(3)
    else:
        return -5*zet


def intg_phi_exp(rad_ai, rad_ki, zet1, zet2, lobu, zor):

    n_trapz = 1000

    phi_v = np.ndarray(shape=[n_trapz], dtype=float)
    zeta_v = np.linspace(start=zet1, stop=zet2, num=n_trapz, endpoint=True)
    
    
    n_md = np.size(rad_ai)
    
    for k in range(0,n_trapz):
        phi_v[k] = phi_oh(zeta_v[k])


    intg = 0
    for md in range(0,n_md):
        tmp_v = np.exp(-lobu * zeta_v * rad_ki[md]) * (1. - phi_v) / (zeta_v + zor / lobu)
        intg += rad_ai[md] * np.trapz(tmp_v, zeta_v)

    return intg

def Psiuo(zet):
    zet_to_avg = np.linspace(0., zet)
    if (zet<0):
        x=(1.-15.*zet_to_avg)**.25
        psik=2.*np.log((1.+x)/2.)+np.log((1.+x*x)/2.)-2.*np.arctan(x)+2.*np.arctan(1.)
        x=(1.-10.15*zet_to_avg)**.3333
        psic=1.5*np.log((1.+x+x*x)/3.)-np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.))+4.*np.arctan(1.)/np.sqrt(3.)
        f=zet_to_avg*zet_to_avg/(1+zet_to_avg*zet_to_avg)
        psiuo=np.average((1-f)*psik+f*psic)
    else:
        c=np.minimum(50.,.35*zet_to_avg)
        psiuo=-np.average((1+1.0*zet_to_avg)**1.0
            +.667*(zet_to_avg-14.28)/np.exp(c)+8.525, returned=False)
    return psiuo
      
def Psit_30(zet):
    zet_to_avg = np.linspace(0., zet)
    if(zet<0):
        x=(1.-(15*zet_to_avg))**.5
        psik=2*np.log((1+x)/2)
        x=(1.-(34.15*zet_to_avg))**.3333
        psic=1.5*np.log((1.+x+x*x)/3.)-np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.))+4.*np.arctan(1.)/np.sqrt(3.)
        f=zet_to_avg*zet_to_avg/(1+zet_to_avg*zet_to_avg)
        psit_30=np.average((1-f)*psik+f*psic)
    else:
        c=np.minimum(50.,.35*zet_to_avg)
        psit_30=-np.average((1.+2./3.*zet_to_avg)**1.5
                +.6667*(zet_to_avg-14.28)/np.exp(c)+8.525, returned=False)
    return psit_30

def psit_30(zet):
    if(zet<0):
        x=(1.-(15*zet))**.5 
        psik=2*np.log((1+x)/2) 
        x=(1.-(34.15*zet))**.3333 
        psic=1.5*np.log((1.+x+x*x)/3.)-np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.))+4.*np.arctan(1.)/np.sqrt(3.) 
        f=zet*zet/(1+zet*zet) 
        psit_30=(1-f)*psik+f*psic   
      
    else:
        c=min(50.,.35*zet) 
        psit_30=-((1.+2./3.*zet)**1.5+.6667*(zet-14.28)/np.exp(c)+8.525)

    return psit_30

def psiuo(zet):
    if (zet<0):
        x=(1.-15.*zet)**.25
        psik=2.*np.log((1.+x)/2.)+np.log((1.+x*x)/2.)-2.*np.arctan(x)+2.*np.arctan(1.)
        x=(1.-10.15*zet)**.3333
        psic=1.5*np.log((1.+x+x*x)/3.)-np.sqrt(3.)*np.arctan((1.+2.*x)/np.sqrt(3.))+4.*np.arctan(1.)/np.sqrt(3.)
        f=zet*zet/(1+zet*zet)
        psiuo=(1-f)*psik+f*psic
    else:
        c=min(50.,.35*zet)
        psiuo=-((1+1.0*zet)**1.0+.667*(zet-14.28)/np.exp(c)+8.525)
    return psiuo



def coare_fullsl_rad(du_norm,du_arg,dt,dq, \
                     tatm,qatm,\
                     zu,zt,zq,\
                     zi,\
                     zo1, full_sl, inc_rad,
                     ohl_rad,
                     qsw_net, qlw_net,
                     nits, averaged: bool=False,
                     averaged_oce: bool=False):


    n_out = 7
    output = np.zeros(shape=[nits,n_out], dtype=float)
    
    beta=1.2
    von=.4
    fdg = 1.
    grav = 9.82


    e1 = np.exp(1)
    rad_ai = np.array([0.237, 0.360, 0.179, 0.087, 0.08, 0.0246, 0.025, 0.007, 0.0004], dtype=float)
    rad_ki = 1./np.array([ 34.8,  2.27, 0.0315, 5.48e-3, 8.32e-4, 1.26e-4, 3.13e-4, 7.82e-5, 1.44e-5], dtype=float)

    n_rad = np.size(rad_ki)

    
    rho_atm = 1.2
    rho_oce = 1025.

    n_trapz = 300

    cp_atm = 1005.
    cp_oce = 4190.

    alpha_eos = 1.8e-4
    
    lambda_u = np.sqrt(rho_atm / rho_oce)
    lambda_t = lambda_u * cp_atm / cp_oce


    nu_atm = 1.5e-5
    nu_oce = 1.e-6

    mu_m = nu_oce / nu_atm

    ug=.5

    ut=np.sqrt(du_norm*du_norm+ug*ug)
    u10=ut*np.log(10/1e-4)/np.log(zu/1e-4)
    usr=.035*u10 # turbulent friction velocity (m/s), including gustiness
    zo10=0.011*usr*usr/grav+0.11*nu_atm/usr # roughness length for u (smith 88)
    Cd10=(von/np.log(10/zo10))**2
    Ch10=0.00115
    Ct10=Ch10/np.sqrt(Cd10)
    zot10=10/np.exp(von/Ct10) # roughness length for t
    Cd=(von/np.log(zu/zo10))**2
    Ct=von/np.log(zt/zot10)
    CC=von*Ct/Cd
    Ribcu=-zu/zi/.004/beta**3
    Ribu=grav*zu/tatm*(dt+.61*tatm*dq)/ut**2
    alpha_eos = 1.8e-4


    if (Ribu >0):
        zetu=CC*Ribu/(1+Ribu/Ribcu)
    else:
        zetu=CC*Ribu*(1+27/9*Ribu/CC)


    L10= 1e100 if abs(zetu) < 1e-100 else zu/zetu
    if averaged:
        bigg_atm = np.log(zu/zo10)-1-Psiuo(zu/L10)
        tsr=dt*von*fdg/(np.log(zt/zot10)-1-Psit_30(zt/L10))
    else:
        bigg_atm = np.log(zu/zo10)-psiuo(zu/L10)
        tsr=dt*von*fdg/(np.log(zt/zot10)-psit_30(zt/L10))

    usr=np.maximum(ut*von/bigg_atm, 1e-8)

    qsr=dq*von*fdg/(np.log(zq/zot10)-psit_30(zq/L10))

    # charnock constant - lin par morceau - constant
    if (ut<=10.):
        charn=0.011 
    else:
       if (ut>18):
           charn=0.018
       else:
           charn=0.011+(ut-10)/(18-10)*(0.018-0.011) 

    zo=charn*usr*usr/grav+0.11*nu_atm/usr  

    # usr_oce = lambda_u * usr
    # tsr_oce = lambda_t * tsr
    # lobu_oce = usr_oce**2 / (von * grav * alpha_eos * tsr_oce)

    # zou_oce = - lambda_u * nu_oce / nu_atm * zo10
    # zot_oce = - lambda_t * k0_oce / k0_atm * zot10

    # bigg_oce = np.log(

    dir_tau_atm = 0.
    dir_tau_oce = 0.
    bigg_oce = 0.
    usr_oce = 0.
    tsr_oce = 0.
    
    for i in range(0,nits):
        prev_usr, prev_tsr = usr, tsr
        zet=von*grav*zu/tatm*(tsr*(1+0.61*qatm)+.61*tatm*qsr)/(usr*usr)/(1+0.61*qatm)
        L= 1e100 if abs(zet) < 1e-100 else zu/zet
        # zo=charn*usr*usr/grav+0.11*nu_atm/usr
        # # rr= zo*usr/nu_atm
        # rr=charn*usr*usr*usr/grav/nu_atm + 0.11
        # # zoq=min(1.15e-4,5.5e-5/rr**.6)
        # try:
        #     zoq=1.15e-4 if 11.5*rr**.6 < 5.5 else 5.5e-5/rr**.6
        # except:
        #     zoq = 1.15e-4
        # zot=zoq
        zot = zoq = zo = nu_atm / usr / von
        ut = np.sqrt( du_norm**2 + ug**2)
        if(full_sl):
            if(inc_rad):
                tsr_oce = lambda_t * tsr - (qlw_net + qsw_net) / (rho_oce * cp_oce * lambda_u * usr)
                
                denom = (von * grav * alpha_eos * ( rho_oce*cp_oce * lambda_u * usr * tsr_oce + qlw_net + qsw_net))

                if abs(denom) > 1e-12:
                    lobu_oce = rho_oce * cp_oce * (lambda_u * usr)**3 \
                               / denom
                else:
                    lobu_oce = rho_oce * cp_oce * (lambda_u * usr)**3 \
                               * 1e12
            else:
                denom = von * grav * alpha_eos * lambda_t * tsr
                if abs(denom) > 1e-12:
                    lobu_oce = (lambda_u * usr)**2 / denom
                else:
                    lobu_oce = (lambda_u * usr)**2 * 1e12
                
            bigg_atm = np.log(zu/zo)-psiuo(zu/L) \
                       + lambda_u * (np.log(-zo1 / \
                    (mu_m * zo / lambda_u )) - \
                    psi_om(-zo1 / lobu_oce))
            bracket_temp = np.log(-zo1 / (mu_m * zot / lambda_u )) - \
                    psi_oh(-zo1 / lobu_oce)

            if averaged:
                bigg_atm -= 1 -psiuo(zu/L) + Psiuo(zu/L)
            if averaged_oce and abs(zo1) > 1e-2:
                bigg_atm -= lambda_u
                bracket_temp -= 1

            usr=np.maximum(ut*von/bigg_atm, 1e-8)

            dt_eff = dt
            if(inc_rad):
                if(ohl_rad):

                    zeta1 = -zo1 / lobu_oce
                    zor_oce = mu_m * zot / lambda_u

                    intg_phi = intg_phi_exp( rad_ai, rad_ki, 0., zeta1, \
                                             lobu_oce, mu_m * zot / lambda_u )
                    

                    denom_rad = 1. / (von * rho_oce * cp_oce * lambda_u * usr)
                    
                    dt_eff += qlw_net * denom_rad * bracket_temp
                    integrated_shortwave_frac_sl = \
                        np.sum( rad_ai * e1**(rad_ki*zor_oce) * \
                        ( exp1(rad_ki*zor_oce) - \
                            exp1(rad_ki*(-zo1+zor_oce)))) - intg_phi

                    dt_eff += qsw_net * denom_rad * \
                            integrated_shortwave_frac_sl
                else:
                    dt_eff += (qsw_net+qlw_net) / (von * rho_oce * cp_oce * lambda_u * usr) * bracket_temp
            tsr= dt_eff*von*fdg/( np.log(zt/zot)-psit_30(zt/L) \
                             + lambda_t * bracket_temp )
            if averaged:
                tsr= dt_eff*von*fdg/( np.log(zt/zot)-1-Psit_30(zt/L) \
                                 + lambda_t * bracket_temp )
            qsr=dq*von*fdg/(np.log(zq/zoq)-psit_30(zq/L) )
        else:
            if averaged:
                bigg_atm = np.log(zu/zo)-1-Psiuo(zu/L)
                tsr=dt*von*fdg/(np.log(zt/zot)-1-Psit_30(zt/L) )
            else:
                bigg_atm = np.log(zu/zo)-psiuo(zu/L)
                tsr=dt*von*fdg/(np.log(zt/zot)-psit_30(zt/L) )

            usr=np.maximum(ut*von/bigg_atm, 1e-8)
            qsr=dq*von*fdg/(np.log(zq/zoq)-psit_30(zq/L) )

        Bf=-grav/tatm*usr*(tsr+.61*tatm*qsr)
        if (Bf > 0):
            ug=beta*(Bf*zi)**.333
        else:
            ug=.2

        tstar_cv = abs(tsr)<1e-12 or \
                abs((prev_tsr - tsr)/tsr) < 0.02
        ustar_cv = abs(usr)<1e-12 or \
                abs((prev_usr - usr)/usr) < 0.02
        if tstar_cv and ustar_cv:
            break
    else: # Convergence failed !
        print("convergence not attained.")
        print(usr, tsr)
        print(prev_usr, prev_tsr)

    if(full_sl):
        return usr, tsr, zo, zot, L, ut, lobu_oce
    else:
        return usr, tsr, zo, zot, L, ut
