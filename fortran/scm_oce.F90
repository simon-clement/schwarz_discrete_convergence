program scm_oce

!++   implicit none
!++   contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!subroutine obl_stp(  z_r     ,      &            ! Depth at cell centers    [m]
!                     z_w     ,      &            ! Depth at cell interfaces [m]
!                     Hz      ,      &            ! Cell thickness           [m]
!                     unudge  ,      &            ! zonal      geostrophic current [m/s]
!                     vnudge  ,      &            ! meridional geostrophic current [m/s]
!                     tnudge  ,      &            ! tracer reference profile (for nudging) [Celsius/PSU]
!                     u       ,      &            ! zonal velocity [m/s]
!                     v       ,      &            ! meridional velocity [m/s]
!                     t       ,      &            ! active tracers [Celsius/PSU]
!                     turb    ,      &            ! GLS variables TKE + length scale
!                     rho0    ,      &            ! Reference constant density [kg/m3]
!                     rho1    ,      &            ! Density perturbation [kg/m3]
!                     Akv     ,      &            ! Turbulent viscosity  [m2/s]
!                     Akt     ,      &            ! Turbulent diffusivity [m2/s]
!                     r_D     ,      &            ! bottom drag (r_D = C_D |u1|)
!                     sustr   ,      &            ! zonal wind stress [m2/s2 = (N/m2) / (kg/m3) ]
!                     svstr   ,      &            ! meridional wind stress  [m2/s2 = (N/m2) / (kg/m3)]
!                     srflx   ,      &            ! solar radiation
!                     stflx1  ,      &            ! net heat flux
!                     stflx2  ,      &            ! net freshwater flux
!                     dtdz_bot,      &            ! Vertical derivative of tracers at the bottom (edge maintenance condition)
!                     delta   ,      &            ! nudging coefficient
!                     f       ,      &            ! Coriolis parameter
!                     Ricr    ,      &            ! Critical Richardson number (for KPP)
!                     hbls    ,      &            ! Surface boundary layer depth
!                     dt      ,      &            ! Time-step
!                     dpdx    ,      &            ! forcing-term for u equation
!                     trb_scheme ,   &            ! Choice of turbulence scheme
!                     sfunc_opt,     &            ! Choice of stability function for GLS
!                     lin_eos ,      &            ! Boolean for use of linear equation of state
!                     alpha   ,      &            ! Thermal expansion coefficient in linear EOS
!                     T0      ,      &            ! Reference temperature in linear EOS
!                     Zob     ,      &            ! Bottom roughness length
!                     Neu_bot ,      &            ! Bottom boundary condition for GLS prognostic variables
!                     nstp    ,      &            ! time n
!                     nnew    ,      &            ! time n+1
!                     N       ,      &            ! Number of vertical grid points        )
!                     ntra    ,      &
!                     ntime   , ngls       )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       use scm_tke

       implicit none
       INTEGER                                  :: ntra,ntime,ngls
	     PARAMETER(ntra  = 2, ntime = 2, ngls  = 2)
	     REAL(8)                                  :: nuwm,nuws,hbls
	     PARAMETER(nuwm = 1.0e-4, nuws = 0.1e-4)     !<-- valeur minimum pour la diffusivite turbulente
       INTEGER                                  :: nstp,N,nnew
	     INTEGER                                  :: nb_steps,output_freq,nout
       LOGICAL                                  :: lin_eos,Neu_bot

	     REAL(8), ALLOCATABLE, DIMENSION(:,:   )  :: u,v
	     REAL(8), ALLOCATABLE, DIMENSION(:,:,: )  :: t
	     REAL(8), ALLOCATABLE, DIMENSION(:     )  :: Akv, Akv_tmp
	     REAL(8), ALLOCATABLE, DIMENSION(:,:   )  :: Akt, Akt_tmp
	     REAL(8), ALLOCATABLE, DIMENSION(:     )  :: rho1,rho
	     REAL(8), ALLOCATABLE, DIMENSION(:,:,: )  :: turb
	     REAL(8), ALLOCATABLE, DIMENSION(:     )  :: delta,unudge,vnudge
	     REAL(8), ALLOCATABLE, DIMENSION(:,:   )  :: tnudge
	     REAL(8)                                  :: dtdz_bot(ntra)
	     REAL(8), ALLOCATABLE, DIMENSION(:     )  :: bvf, FC,CF
	     REAL(8), ALLOCATABLE, DIMENSION(:     )  :: swr_frac
! Grid variables
	     REAL(8), ALLOCATABLE, DIMENSION(:     )  :: z_r,z_w,Hz
	     REAL(8)                                  :: hc,theta_s,hmax, ds
! Surface forcings
       REAL(8)                                  :: sustr,svstr,srflx,stflx(ntra)
       REAL(8)                                  :: ustars,heatloss,Qswmax
! Parameters
       REAL(8)                                  :: f,dt,dpdx,time
       REAL(8)                                  :: rho0,alpha,T0,N0
       REAL(8)                                  :: r_D,zOb,gamma
	     PARAMETER (gamma=0.55)
       REAL(8)                                  :: cff, cff1, cff2, cff3
	     INTEGER                                  :: k,itrc,kt
       REAL(8)                                  :: cp,grav,rpi
	     PARAMETER (cp = 3985.0d0, grav = 9.81, rpi = 4.*ATAN(1.) )
!       REAL(8)   :: c1,c2,c3,c4,c5,c6,cb1,cb2,cb3,cb4,cb5,cbb,
	     REAL(8)                                  :: sc_r,sc_w
!=========================================================
! General parameters
!=========================================================
     N       = 50; hmax = 50.; hc = 50.; theta_s = 6.5             !++ Vertical grid parameters
	   dt      = 30.
	   rho0    = 1024.
	   f       = 0.
	   dpdx    = 0.
	   r_D     = 0.
	   zOb     = 0.
     Neu_bot = .TRUE.
     lin_eos = .TRUE.; T0=16.; alpha=0.0002; N0=0.01   !++ Linear EOS
     ustars  = 0.01; heatloss=0.;Qswmax=0.
     !
	   nb_steps     =  1       !++ nombre de pas de temps  192 heures = 8 jours
     output_freq  =   120         !++ frequence de stockage de la solution (toutes les heures ici)
     nout         = INT( (nb_steps-1)/output_freq  + 1 )              !++ nombre d'instants stockes pendant la simulation
     dTdz_bot(1)  = N0*N0/(alpha*grav)
     dTdz_bot(2)  = 0.d0

!=========================================================
! Allocate arrays
!=========================================================
     ALLOCATE( u(1:N,ntime), v(1:N,ntime), t(1:N,ntime,ntra)  )
     ALLOCATE( Akv(0:N )   , Akt(0:N,ntra), rho1(1:N), rho(1:N)  )
     ALLOCATE( unudge(1:N), vnudge(1:N), tnudge(1:N,ntra), delta(1:N)  )
     ALLOCATE( z_r(1:N), z_w(0:N), Hz(1:N) )
  	 ALLOCATE( turb(0:N,ntime,ngls), bvf(0:N),swr_frac(0:N) )
	   ALLOCATE( FC(0:N), CF(0:N) )
!=========================================================
! Initialization
!=========================================================
     z_w(0) = -hmax
	   cff    = 1.D0/FLOAT(N)
	   cff1   = (hmax-hc)/SINH(theta_s)
	   ds     = 1./FLOAT(N)
     DO k = 1,N
		   sc_r   = ds * ( FLOAT(k-N) - 0.5 )
	     z_r(k) = hc * sc_r + cff1*SINH(theta_s*sc_r)
       sc_w   = ds * FLOAT(k-N)
		   z_w(k) = hc * sc_w + cff1*SINH(theta_s*sc_w)
	   ENDDO

	   DO k = 1,N
	     Hz(k) = z_w(k) - z_w(k-1)
	   ENDDO
! No large scale forcing
     unudge(:) = 0.; vnudge(:) = 0.; tnudge(:,:) = 0.; delta(:) = 0.
! Initial conditions
     u(1:N,1:2) = 0.; v(1:N,1:2) = 0.; t(1:N,1:2,2) = 35.
	   DO k = 1,N
	     t(k,1:2,1) = T0 - N0*N0*ABS(z_r(k))/(alpha*grav)
  	 ENDDO
!
	   Akv(0:N  ) = 0.  ; Akt(0:N  ,1) = 0.  ; Akt(0:N  ,2) = 0.
	   Akv(1:N-1) = nuwm; Akt(1:N-1,1) = nuws; Akt(1:N-1,2) = nuws
     turb(0:N,1:2,1:2) = 0.
	   hbls       = ABS(z_w(N-1))
! Solar radiation penetration
     call  lmd_swfrac (N, swr_frac, Hz )
!=========================================================
! MAIN LOOP
!=========================================================
     DO kt = 0,nb_steps

       output_freq  = 1.
       nstp = 1 + MOD(kt  ,2)
		   nnew = 1 + MOD(kt+1,2)
		   time = dt*FLOAT(kt)

       !++ Surface forcings
		   sustr    = ustars*ustars
		   svstr    = 0.
		   srflx    = MAX( COS(2.*rpi*(time/86400. - 0.5)), 0. )* Qswmax / (rho0*cp)  ! flux solaire (variation diurne) [C m / s]
       stflx(1) = srflx - heatloss / (rho0*cp)                                    ! flux de chaleur nette  [C m / s]
       stflx(2) = 0.                                                              ! flux d'eau douce [psu m / s]

        IF(lin_eos) THEN
           CALL  rho_lin_eos    (N, rho1,bvf, t(:,nstp,:),z_r,rho0,alpha,T0 )
        ELSE
           CALL  rho_eos        (N, rho1,bvf, t(:,nstp,:),z_r,rho0          )
        ENDIF

  		  CALL  tke_stp(Hz,z_r,u,v,bvf,turb(:,:,1),turb(:,:,2),Akv,Akt,r_D,sustr,svstr,      &
                                        dt,Zob,Neu_bot,nstp,nnew,N,ntra,ngls,ntime)
!
! Tracer equations: vertical viscosity and solar penetration
!--------- ---------- -------- ----- --- -------- ---------
!
        do itrc=1,2
!-----------
          if (itrc.eq.1) then
             FC(N)=stflx(itrc)     !<-- surface heat flux (including
                                    !    latent and solar components)
             do k=N-1,1,-1
               FC(k)=srflx*swr_frac(k)   !<-- penetration of solar heat flux
             enddo
          else
             FC(N)=stflx(itrc)      !<-- salinity (fresh water flux)
             do k=N-1,1,-1
                 FC(k)=0.D0
             enddo
          endif
!
! Bottom flux
!------
          FC(0) = nuws*dTdz_bot(itrc)  !<-- Neumann BC at the bottom
!
! Implicit integration for vertical diffusion
!------
          do k=1,N
            t(k,nnew,itrc)=Hz(k)*t(k,nstp,itrc)       &
                             +dt*(FC(k)-FC(k-1))            !++ dQs/dz
          enddo

          FC(1)=2.D0*dt*Akt(1,itrc)/(Hz(2)+Hz(1))       !--> resolve
          cff=1./(Hz(1)+FC(1))                          ! tri-diagonal
          CF(1)=cff*FC(1)                               ! system...
          t(1,nnew,itrc)=cff*t(1,nnew,itrc)

          do k=2,N-1
            FC(k)=2.D0*dt*Akt(k,itrc)/(Hz(k+1)+Hz(k))
            cff=1./(Hz(k)+FC(k)+FC(k-1)*(1.-CF(k-1)))
            CF(k)=cff*FC(k)
            t(k,nnew,itrc)=cff*( t(k,nnew,itrc) +FC(k-1)             &
                                      *t(k-1,nnew,itrc))
          enddo
          t(N,nnew,itrc)=(t(N,nnew,itrc)+FC(N-1)*t(N-1,nnew,itrc))   &
                                  /(Hz(N)+FC(N-1)*(1.-CF(N-1)))
          do k=N-1,1,-1
             t(k,nnew,itrc)=t(k,nnew,itrc)+CF(k)*t(k+1,nnew,itrc)  !<-- tracer value of implicit diffusion
          enddo
!
! Nudging (toward tnudge)
!------
          do k=N-1,1,-1
              t(k,nnew,itrc)=t(k,nnew,itrc) - dt*delta(k)*(           &
                            t(k,nnew,itrc)-tnudge(k,itrc) )
          enddo
!-----------
         enddo   ! <-- itrc
!-----------


!
! Momentum equations: Coriolis terms and vertical viscosity
!--------- ---------- -------- ----- --- -------- ---------
!


!
! Coriolis term  (forward-backward)
!------

        IF(nstp==1) THEN
           do k=1,N
              cff       = f * (   v(k,nstp) - vnudge(k) )
              u(k,nnew) =         u(k,nstp) + dt * cff
              cff       = f * (   u(k,nnew) - unudge(k) )
              v(k,nnew) = Hz(k)*( v(k,nstp) - dt * cff  )
              u(k,nnew) = Hz(k)*( u(k,nnew)             )
           enddo
        ELSE
           do k=1,N
              cff       = f * (   u(k,nstp) - unudge(k) )
              v(k,nnew) =         v(k,nstp) - dt * cff
              cff       = f * (   v(k,nnew) - vnudge(k) )
              u(k,nnew) = Hz(k)*( u(k,nstp) + dt * cff  )
              v(k,nnew) = Hz(k)*( v(k,nnew)             )
           enddo
        ENDIF

!
! Apply surface forcing
!------
        u(N,nnew)=u(N,nnew) + dt*sustr    !<-- sustr is in m2/s2 here
        v(N,nnew)=v(N,nnew) + dt*svstr
!
! Resolve tri-diagonal system
!------
        FC(1)=2.D0*dt*Akv(1)/(Hz(2)+Hz(1)) !<--     c(1)     ! system
        cff=1./(Hz(1)+FC(1)+dt*r_D)        !<-- 1 / b(1) implicit bottom drag appears here
        CF(1)=cff*FC(1)                    !<-- q(1)
        u(1,nnew)=cff*u(1,nnew)
        v(1,nnew)=cff*v(1,nnew)
        do k=2,N-1
          FC(k)=2.D0*dt*Akv(k)/(Hz(k+1)+Hz(k))
          cff=1.D0/(Hz(k)+FC(k)+FC(k-1)*(1.D0-CF(k-1)))
          CF(k)=cff*FC(k)
          u(k,nnew)=cff*(u(k,nnew)+FC(k-1)*u(k-1,nnew))
          v(k,nnew)=cff*(v(k,nnew)+FC(k-1)*v(k-1,nnew))
        enddo
        cff=1./( Hz(N) +FC(N-1)*(1.-CF(N-1)) )
        u(N,nnew)=cff*(u(N,nnew)+FC(N-1)*u(N-1,nnew))
        v(N,nnew)=cff*(v(N,nnew)+FC(N-1)*v(N-1,nnew))
!
! Finalize and apply damping term
!------

        do k=N-1,1,-1
          u(k,nnew)=u(k,nnew)+CF(k)*u(k+1,nnew) - dt*dpdx
          v(k,nnew)=v(k,nnew)+CF(k)*v(k+1,nnew)
        enddo
       !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
        IF(MOD(kt,output_freq)==0) THEN
	        print*,'kt = ',kt,' umax = ',MAXVAL(ABS(u(1:N,nnew))),' vmax = ',MAXVAL(ABS(v(1:N,nnew))),   &
		     &                '(tmin,tmax) = (',MINVAL(t(1:N,nnew,1)),',',MAXVAL(t(1:N,nnew,1)),')'
	      ENDIF

      ENDDO

      open(1, file = 't_final_tke.out')
        DO k=1,N
          write(1,*) t(k,nnew,1), z_r(k)
        ENDDO
      close(1)

      open(1, file = 'Akt_final_tke.out')
        DO k=0,N
          write(1,*) Akt(k,1), z_w(k)
        ENDDO
       close(1)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end program
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!











!===================================================================================================
       SUBROUTINE rho_eos (N,rho1,bvf,t,z_r,rho0)
!---------------------------------------------------------------------------------------------------
      implicit none
!
!-- Equation Of State variables to compute oceanic density ------------------------------------
!
      integer, intent(in   )                   :: N
      real(8), intent(  out)                   :: bvf (0:N)
      real(8), intent(  out)                   :: rho1(1:N)
      real(8), intent(in   )                   :: t  (1:N,2)
      real(8), intent(in   )                   :: z_r(1:N  )
      real(8), intent(in   )                   :: rho0
! local variables
      real(8)                                  :: Ts,Tt,sqrtTs
      integer                                  :: k
      real(8)                                  :: K0(N),K1(N),K2(N)
      real(8)                                  :: r0, cff
      real(8) A00, A01, A02, A03, A04, A10, A11, A12, A13
      real(8) AS0, AS1, AS2, B00, B01, B02, B03, B10, B11
      real(8) B12, BS1, E00, E01, E02, E10, E11, E12
      real(8) QR , Q01, Q02, Q03, Q04, Q05, Q10, Q11
      real(8) Q12, Q13, Q14, QS0, QS1, QS2, Q20
      real(8), parameter                       :: g=9.81
! parameter values
      parameter(A00=+19092.56 ,  A01=+209.8925   , A02=-3.041638,     &
               A03=-1.852732e-3, A04=-1.361629e-5, A10=104.4077  ,    &
               A11=-6.500517   , A12=+0.1553190  , A13=2.326469e-4,   &
               AS0=-5.587545   , AS1=+0.7390729  , AS2=-1.909078e-2,  &
               B00=+4.721788e-1, B01=+1.028859e-2, B02=-2.512549e-4,  &
               B03=-5.939910e-7, B10=-1.571896e-2, B11=-2.598241e-4,  &
               B12=+7.267926e-6, BS1=+2.042967e-3,                    &
               E00=+1.045941e-5, E01=-5.782165e-10,E02=+1.296821e-7,  &
               E10=-2.595994e-7, E11=-1.248266e-9, E12=-3.508914e-9)
      parameter(QR=+999.842594 , Q01=+6.793952e-2, Q02=-9.095290e-3,  &
               Q03=+1.001685e-4, Q04=-1.120083e-6, Q05=+6.536332e-9,  &
               Q10=+0.824493   , Q11=-4.08990e-3 , Q12=+7.64380e-5,   &
               Q13=-8.24670e-7 , Q14=+5.38750e-9 , QS0=-5.72466e-3,   &
               QS1=+1.02270e-4 , QS2=-1.65460e-6 , Q20=+4.8314e-4)
!---------------------------------------------------------------------------------------------------
      r0 = QR-1000.d0
! Compute density anomaly via Equation Of State (EOS) for seawater
!-------
      do k=1,N
!----
        Tt       = t(k,1)
        Ts       = t(k,2)
        sqrtTs   = sqrt(Ts)

        K0(k)    =    A00+Tt*(A01+Tt*(A02+Tt*(A03+Tt*A04)))            &
                + Ts*( A10+Tt*(A11+Tt*(A12+Tt*A13))+sqrtTs*(          &
                                    AS0+Tt*(AS1+Tt*AS2) ))

        K1(k)   =    B00+Tt*(B01+Tt*(B02+Tt*B03))+Ts*( B10            &
                +    Tt*(B11+Tt*B12)+sqrtTs*BS1)

        K2(k)   =    E00+Tt*(E01+Tt*E02)+Ts*(E10+Tt*(E11+Tt*E12))

        rho1(k) =   r0+Tt*(Q01+Tt*(Q02+Tt*(Q03+Tt*(Q04+Tt*Q05))))      &
                       +Ts*(Q10+Tt*(Q11+Tt*(Q12+Tt*(Q13+Tt*Q14)))      &
                            +sqrtTs*(QS0+Tt*(QS1+Tt*QS2))+Ts*Q20)
!----
    enddo
!----
    do k=1,N-1
        cff    = 1./(z_r(k+1)-z_r(k))
        bvf(k) = -cff*(g/rho0)*(rho1(k+1)-rho1(k))  ! Brunt-Vaisala frequency
    enddo
    bvf(0) = bvf(1  )
    bvf(N) = bvf(N-1)
!---------------------------------------------------------------------------------------------------
END SUBROUTINE rho_eos
!===================================================================================================



!===================================================================================================
       SUBROUTINE rho_lin_eos (N,rho1,bvf,t,z_r,rho0,Tcoef,T0)
!---------------------------------------------------------------------------------------------------
      implicit none
!
!-- Equation Of State variables to compute oceanic density ------------------------------------
!
      integer, intent(in   )                   :: N
      real(8), intent(  out)                   :: bvf (0:N)
      real(8), intent(  out)                   :: rho1(1:N)
      real(8), intent(in   )                   :: t  (1:N,2)
      real(8), intent(in   )                   :: z_r(1:N  )
      real(8), intent(in   )                   :: rho0, Tcoef, T0
! local variables
      real(8)                                  :: Ts,Tt,sqrtTs
      integer                                  :: k
      real(8)                                  :: K0(N),K1(N),K2(N)
      real(8)                                  :: r0, cff
      real(8), parameter                       :: g=9.81
!---------------------------------------------------------------------------------------------------
! Compute density anomaly via linear Equation Of State (EOS)
!-------
      do k=1,N
!----
         rho1(k)= rho0*( 1. - Tcoef*( t(k,1) - T0 ) )     ! + Scoef*(t(i,j,k,nrhs,isalt)-S0)
!----
    enddo
!----
    do k=1,N-1
        cff    = 1./(z_r(k+1)-z_r(k))
        bvf(k) = -cff*(g/rho0)*(rho1(k+1)-rho1(k))  ! Brunt-Vaisala frequency
    enddo
    bvf(0) = bvf(1  )
    bvf(N) = bvf(N-1)

!---------------------------------------------------------------------------------------------------
END SUBROUTINE rho_lin_eos
!===================================================================================================








!
!===================================================================================================
subroutine lmd_swfrac(N,swr_frac,Hz)
!---------------------------------------------------------------------------------------------------
      implicit none
      integer,intent(in   )   :: N
      real(8),intent(  out)   :: swr_frac(0:N)
      real(8),intent(in   )   :: Hz      (1:N)
! local variables
      integer k, Jwt
      real(8) swdk1,swdk2,xi1,xi2
      real(8) mu1(5),mu2(5), r1(5), attn1, attn2
!
! Compute fraction of solar shortwave flux penetrating to specified
! depth due to exponential decay in Jerlov water type.
!
! output: swr_frac     shortwave (radiation) fractional decay.
!
! Reference:
! Paulson, C.A., and J.J. Simpson, 1977: Irradiance measurements
! in the upper ocean, J. Phys. Oceanogr., 7, 952-956.
!----------

      mu1(1)=0.35    ! reciprocal of the absorption coefficient
      mu1(2)=0.6     ! for each of the two solar wavelength bands
      mu1(3)=1.0     ! as a function of Jerlov water type (Paulson
      mu1(4)=1.5     ! and Simpson, 1977) [dimensioned as length,
      mu1(5)=1.4     ! meters];

      mu2(1)=23.0
      mu2(2)=20.0
      mu2(3)=17.0
      mu2(4)=14.0
      mu2(5)=7.9

      r1(1)=0.58     ! fraction of the total radiance for
      r1(2)=0.62     ! wavelength band 1 as a function of Jerlov
      r1(3)=0.67     ! water type (fraction for band 2 is always
      r1(4)=0.77     ! r2=1-r1);
      r1(5)=0.78
                     ! set Jerlov water type to assign everywhere
      Jwt=1          ! (an integer from 1 to 5).

      attn1=-1./mu1(Jwt)
      attn2=-1./mu2(Jwt)

      swdk1=r1(Jwt)               ! surface, then attenuate
      swdk2=1.-swdk1              ! them separately throughout
      swr_frac(N)=1.              ! the water column.

      do k=N,1,-1
         xi1=attn1*Hz(k)
         if (xi1 .gt. -20.) then        ! this logic to avoid
            swdk1=swdk1*exp(xi1)        ! computing exponent for
         else                           ! a very large argument
            swdk1=0.
         endif
         xi2=attn2*Hz(k)
         if (xi2 .gt. -20.) then
            swdk2=swdk2*exp(xi2)
         else
            swdk2=0.
         endif
         swr_frac(k-1)=swdk1+swdk2
      enddo
!---------------------------------------------------------------------------------------------------
end subroutine lmd_swfrac
!===================================================================================================
!
