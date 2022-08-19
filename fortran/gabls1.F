      program gabls1
      implicit none
      integer, parameter            ::  jpka    = 65
      real(8), parameter            ::  ff_t    = 1.39e-04
      real(8), parameter            ::  rhoa    = 1.3223
      real(8), parameter            ::  zrgh    = 0.1
      real(8), parameter            ::  betam   = 4.8
      real(8), parameter            ::  betah   = 7.8
      real(8), parameter            ::  zmax    = 400.
      real(8), parameter            ::  ceps    = 0.845      !: namelist parameter
      real(8), parameter            ::  cm      = 0.126      !: namelist parameter
      real(8), parameter            ::  ce      = 0.34       !: namelist parameter
      real(8), parameter            ::  tke_min = 1.e-6      !: minimum TKE
      real(8), parameter            ::  avm_bak = 1.e-4      !: background viscosity
      real(8), parameter            ::  avt_bak = 1.e-5      !: background diffusivity
      real(8), parameter            ::  vonKar  = 0.4        !: Von Karman constant
      real(8), parameter            ::  t_end_h =   9.       !: total simulation time in hours
      real(8), parameter            ::  Sch     = ce / cm
      real(8), parameter            ::  mxl_min = (avm_bak / cm)
     &                                          / SQRT( tke_min )
      real(8), parameter            ::  Ct      = 0.143
      real(8), parameter            ::  grav    = 9.81
      real(8)                       ::  u_abl   (1:jpka, 1:2 )
      real(8)                       ::  v_abl   (1:jpka, 1:2 )
      real(8)                       ::  t_abl   (1:jpka, 1:2 )
      real(8)                       ::  avm     (1:jpka      )
      real(8)                       ::  avt     (1:jpka      )
      real(8)                       ::  mxlm    (1:jpka      )
      real(8)                       ::  mxld    (1:jpka      )
      real(8)                       ::  tke     (1:jpka, 1:2 )
      real(8)                       ::  pu_dta  (1:jpka      )
      real(8)                       ::  pv_dta  (1:jpka      )
      real(8)                       ::  ght     (1:jpka      )
      real(8)                       ::  ghw     (1:jpka      )
      real(8)                       ::  e3t     (1:jpka      )
      real(8)                       ::  e3w     (1:jpka      )
      integer                       ::  jk,kt,ntimes,nt_n,nt_a
      integer                       ::  iter
      real(8)                       ::  cff,esfc,Lsfc,ustar2,Cd_du
      real(8)                       ::  cfft,Ch_du,tstar,sst,wndm,deltaT
      real(8)                       ::  Cd,Ch,hol,phim,phis,ustar
      real(8)                       ::  rdt,time
      character*5                   ::  suffix
!---------- Define vertical grid
      cff    = zmax / (jpka-1)
      ght(2) = 0.5*cff
      ghw(2) =     cff
      do jk=3,jpka
        ght(jk) = ght(jk-1) + cff
        ghw(jk) = ghw(jk-1) + cff
      end do
      ghw(1) = 0.; e3t(1) = 0.; ght(1) = 0.
      !
      do jk = 2,jpka
        e3t(jk) = ghw(jk)-ghw(jk-1)
      end do
      !
      do jk=1,jpka-1
        e3w(jk)  = ght(jk+1)-ght(jk)
      end do
      e3w(jpka) = ghw(jpka) - ght(jpka)
!--------- Initialize variables
      do jk=1,jpka
         pu_dta(jk  ) =  8.d0
         pv_dta(jk  ) =  0.d0
         u_abl (jk,1) = pu_dta(jk  )
         v_abl (jk,1) = pv_dta(jk  )
         u_abl (jk,2) = pu_dta(jk  )
         v_abl (jk,2) = pv_dta(jk  )
         t_abl (jk,1) = 265.
         if( ght(jk).gt.100. )
     &          t_abl(jk,1) = 265. + 0.01 * (ght(jk)-100.)
         t_abl(jk,2) = t_abl(jk,1)
         tke  (jk,1) = tke_min
         if( ght(jk).le.250. )
     &          tke(jk,1) = tke_min + 0.4*(1.-ghw(jk)/250.)**3
         tke   (jk,2) = tke  (jk,1)
         avm   (jk  ) = avm_bak
         avt   (jk  ) = avt_bak
         mxlm  (jk  ) = mxl_min
         mxld  (jk  ) = mxl_min
      end do
!--------- Time loop
      rdt    = 10.
      ntimes = nint(t_end_h * 3600. / rdt)
      ntimes=3240

      !TODO remettre ntimes+1
      do kt=1,ntimes
        time  = rdt*float(kt-1)               !<-- time in seconds
        nt_n  = 1 + MOD( kt  , 2)             !<-- time index for prognostic variables
        nt_a  = 1 + MOD( kt+1, 2)
        !==== Compute surface boundary conditions
        sst    = 265 - 0.25*(time/3600.)
        wndm   = SQRT(u_abl( 2,nt_n)**2 + v_abl(2,nt_n)**2)  ! wndm
        deltaT = t_abl( 2,nt_n) - sst                        ! deltaT
        ! first guess dfor ustar and tstar
        if (deltaT .gt. 0) then
           tstar = 0.0180 * deltaT
        else
           tstar = 0.0327 * deltaT
        endif
        ustar  = (vonKar / log( 1+ght(2)/zrgh ) )*wndm
        do iter = 1,5
           hol   = vonKar*ght(2)*grav*(tstar/t_abl( 2,nt_n))
     &                                       / (ustar*ustar)
           phim  = - betam * hol
           phis  = - betah * hol
           Cd    = vonKar * vonKar / ( log(1+ ght(2)/zrgh ) - phim )**2
           Ch    = vonKar * SQRT(Cd) / ( log(1+ ght(2)/zrgh ) - phis )
           ustar =        SQRT(Cd) * wndm
           tstar = ( Ch / SQRT(Cd) ) * deltaT
        enddo
        ustar2 = ustar*ustar
        Cd_du  = Cd * wndm
        Ch_du  = Ch * wndm
        esfc   = ustar2 / SQRT( cm * ceps )
        Lsfc   = (vonKar / Cm) * SQRT(SQRT(Cm*ceps)) * (ght(2)+zrgh)
        !==== Compute turbulent viscosity
        Call zdf_tke(jpka,nt_n,nt_a,u_abl(:,nt_n),v_abl(:,nt_n),
     &    t_abl(:,nt_n),avm,avt,tke,mxlm,mxld,e3w,e3t,ghw,
     &    rdt,ustar2,zrgh,avm_bak,avt_bak,cm,ceps,Sch,Ct,
     &    tke_min,mxl_min)

        !==== Advance single-column model by one time-step
        Call step_abl(kt,jpka,nt_n,nt_a,
     &    u_abl,v_abl,t_abl,avm,avt,pu_dta,pv_dta,e3t,e3w,
     &    rdt,cd_du,ch_du,sst,ff_t)
      enddo

      print*,'ustar = ',SQRT(ustar2)
      suffix = 'gabls1'

      open(1, file = 'u_final_'//trim(suffix)//'.out')
      DO jk=2,jpka
         write(1,*) u_abl(jk,nt_a), ght(jk)
      ENDDO
      close(1)

      open(1, file = 'v_final_'//trim(suffix)//'.out')
      DO jk=2,jpka
         write(1,*) v_abl(jk,nt_a), ght(jk)
      ENDDO
      close(1)

      open(1, file = 't_final_'//trim(suffix)//'.out')
      DO jk=2,jpka
         write(1,*) t_abl(jk,nt_a), ght(jk)
      ENDDO
      close(1)

      open(1, file = 'tke_final_'//trim(suffix)//'.out')
      DO jk=1,jpka
         write(1,*) tke(jk,nt_a), ghw(jk)
      ENDDO
      close(1)

      open(1, file = 'mxl_final_'//trim(suffix)//'.out')
      DO jk=1,jpka
         write(1,*) mxlm(jk), ghw(jk)
      ENDDO
      close(1)

      end program gabls1

!        Call zdf_tke(jpka,nt_n,nt_a,u_abl(:,nt_n),v_abl(:,nt_n),t_abl(:,nt_n),
!     &    avm,avt,tke,mxlm,mxld,e3w,e3t,ghw,
!     &    rdt,ustar2,zrgh,avm_bak,avt_bak,cm,ceps,Sch,Ct,tke_min,mxl_min)

      subroutine zdf_tke(jpka,nt_n,nt_a,uu,vv,tt,avm,avt,tke,mxlm,
     &    mxld,e3w,e3t,ghw,
     &    rdt,ustar2,zrgh,avm_bak,avt_bak,cm,ceps,Sch,
     &    Ct,tke_min,mxl_min)
      implicit none
      integer                    :: jpka,nt_n,nt_a
      real(8)                    :: uu(jpka),vv(jpka),avm(jpka)
      real(8)                    :: tt(jpka),avt(jpka)
      real(8)                    :: tke(jpka,2),mxlm(jpka),mxld(jpka)
      real(8)                    :: ghw(jpka),e3w(jpka),e3t(jpka)
      real(8)                    :: rdt,zrgh,avm_bak,cm,ceps,Sch,tke_min
      real(8)                    :: ustar2,mxl_min,Ct,avt_bak
      real(8), parameter         ::  itvref  = 1./283
      real(8), parameter         ::  grav    = 9.81
      ! local variables
      real(8)                    :: CF (1:jpka),shear,buoy,xx
      real(8)                    :: bn2(1:jpka),sh2(1:jpka)
      real(8)                    :: esfc,etop,cff,dU,dV,cff1,cff2
      real(8)                    :: a(jpka),b(jpka),c(jpka)
      real(8)                    :: lup(jpka),ldw(jpka)
      integer                    :: jk
      real(8), parameter         :: vonKar = 0.4
      real(8), parameter         :: Ric    = 0.143
      real(8), parameter         :: phimax = (1.-2.2)/2.2
      real(8), parameter         :: Rod = 0.2

      ! Compute vertical shear source term
      do jk = 2, jpka-1
        cff     = 1.0 / e3w( jk )**2
        dU      = cff* (uu(jk+1)-uu(jk) )**2
        dV      = cff* (vv(jk+1)-vv(jk) )**2
        sh2(jk) = dU+dV   !<-- sh2 = ( ( du/dz )^2 + ( dv/dz )^2 )
      enddo

      ! Compute brunt-vaisala frequency
      do jk = 2, jpka-1
        cff  = grav * itvref / e3w( jk )
        cff1 =  tt(jk+1) - tt(jk)
        bn2(jk) = cff1 * cff
      enddo
      ! Terms for the tridiagonal problem
      do jk = 2, jpka-1
        shear   =   Avm(jk)*sh2(jk)
        buoy    = - Avt(jk)*bn2(jk)
        a( jk ) = - 0.5*rdt*Sch*(Avm(jk)+Avm(jk-1)) / e3t( jk   ) ! lower-diagonal
        c( jk ) = - 0.5*rdt*Sch*(Avm(jk)+Avm(jk+1)) / e3t( jk+1 ) ! upper-diagonal
        if( (buoy + shear) .gt. 0.) then    ! Patankar trick
           b( jk ) = e3w(jk) - a(jk) - c(jk) + e3w(jk) * rdt *
     &              Ceps * SQRT(tke(jk,nt_n)) / mxld( jk )           ! diagonal
           tke(jk,nt_a) = e3w(jk) * ( tke(jk,nt_n)
     &                                + rdt * (shear+buoy) )   ! right-hand-side
        else
           b( jk ) = e3w(jk) - a(jk) - c(jk) + e3w(jk) * rdt *
     &              Ceps * SQRT(tke(jk,nt_n)) / mxld( jk )           ! diagonal
     &            - e3w(jk) * rdt * buoy / tke(jk,nt_n)
          tke(jk,nt_a) = e3w(jk) * ( tke(jk,nt_n) + rdt * shear )   ! right-hand-side
        endif
      end do

      ! boundary conditions
      esfc = MAX( ustar2 / SQRT( cm * ceps ), tke_min )
      tke(1,nt_a) = esfc; a(1) = 0.;c(1)=0.;b(1)=1.

      a ( jpka ) = - 0.5*rdt*Sch*(Avm(jpka-1)+Avm(jpka))/e3t( jpka )
      c ( jpka ) = 0.
      b ( jpka ) = e3w(jpka-1) - a(jpka )

      tke(jpka, nt_a ) = e3w(jpka-1) * tke( jpka, nt_n )

      !!
      !! Matrix inversion (Thomas algorithm)
      !! ----------------------------------------------------------
      cff         =    1. / b(1)
      CF (1     ) = - cff * c(1)
      tke(1,nt_a) =   cff * tke(1,nt_a)
      do jk = 2, jpka
        cff    = 1. / ( b(jk) + a(jk) * CF(jk-1) )
        CF(jk) = - cff * c(jk)
        tke(jk,nt_a) = cff * ( tke(jk,nt_a)-a(jk)*tke(jk-1,nt_a) )
      enddo

      do jk = jpka-1,1,-1
        tke(jk,nt_a) = tke(jk,nt_a) + CF(jk) * tke(jk+1,nt_a)
      enddo

      tke(1:jpka,nt_a) = MAX( tke(1:jpka,nt_a), tke_min )

      !!
      !!  Diagnostic mixing length computation
      !! ----------------------------------------------------------
      xx        = -(LOG(Ceps)-3.*LOG(Cm)+4.*LOG(vonKar))/LOG(16.)
      cff       = (vonKar / Cm) * SQRT(SQRT(Cm*ceps)) * zrgh     ! (ght(2)+zrgh)
      lup( 1  ) = cff
      ldw( 1  ) = cff
      lup(jpka) = mxl_min
      ldw(jpka) = mxl_min
!
! Mixing length computation
      do jk = 2, jpka-1
        buoy     = MAX(bn2(jk),1.E-12)
        !cff      = SQRT( 2.*tke(jk, nt_a) / buoy )
        cff1     = Rod*SQRT(sh2(jk)) +
     &    SQRT(Rod*Rod*sh2(jk)+2.*buoy)
        cff      = 2.*SQRT(tke(jk,nt_a))/cff1
        mxlm(jk) = MAX( mxl_min, cff )
      enddo

      ! Limit mxl
      do jk = jpka-1,1,-1
        lup(jk) = MIN( lup(jk+1) + (ghw(jk+1)-ghw(jk)), mxlm(jk) )
      enddo
      !
      do jk = 2, jpka
        ldw(jk) = MIN( ldw(jk-1) + (ghw(jk)-ghw(jk-1)) , mxlm(jk) )
      enddo

      do jk = 1,jpka
         cff = ( 0.5*( ldw( jk )**(1./xx) + lup( jk )**(1./xx) ) )**xx
!         cff = SQRT( ldw( jk )*lup( jk ) )
         mxlm ( jk ) = MAX( cff, mxl_min )
         mxld ( jk ) = MIN ( ldw( jk ),  lup( jk ) )
      enddo
      !!
      !!  Turbulent viscosity computation
      !! ----------------------------------------------------------
      do jk = 1, jpka
        cff     = MAX(phimax, Ric * mxlm(jk) * mxld(jk)
     &          * MAX(bn2(jk),1.E-12)/tke(jk,nt_a))
        cff1    =  1. / ( 1. + cff )   !<-- phi_z(z)
        cff     = mxlm( jk ) * SQRT( tke( jk, nt_a ) )
        avm(jk) = MAX( Cm * cff       , avm_bak   )
        avt(jk) = MAX( Ct * cff *cff1 , avt_bak   )
      enddo

      end subroutine

      subroutine debug_double(jpka, nt_a, variable, ghw)
      implicit none
      integer,intent(in   )      :: jpka, nt_a
      real(8),intent(inout)      :: variable(jpka, 2)
      real(8),intent(in   )      :: ghw(jpka)
      integer                    :: jk
        
      open(1, file = 'debug.out')
      DO jk=1,jpka
         write(1,*) variable(jk, nt_a), ghw(jk)
      ENDDO
      close(1)

      end subroutine

      subroutine debug_simple(jpka, variable, ghw)
      implicit none
      integer,intent(in   )      :: jpka
      real(8),intent(inout)      :: variable(jpka)
      real(8),intent(in   )      :: ghw(jpka)
      integer                    :: jk

      open(1, file = 'debug.out')
      DO jk=1,jpka
         write(1,*) variable(jk), ghw(jk)
      ENDDO
      close(1)

      end subroutine

      subroutine step_abl(kt,jpka,nt_n,nt_a,u_abl,v_abl,t_abl,
     &        avm,avt,uG,vG,e3t,e3w,rdt,cd_du,ch_du,sst,ff_t)
      implicit none
      integer,intent(in   )      :: jpka,nt_n,nt_a,kt
      real(8),intent(inout)      :: u_abl(jpka,2),v_abl(jpka,2)
      real(8),intent(inout)      :: t_abl(jpka,2)
      real(8),intent(in   )      :: uG(jpka),vG(jpka)
      real(8),intent(in   )      :: avm(jpka),avt(jpka)
      real(8),intent(in   )      :: e3w(jpka),e3t(jpka)
      real(8),intent(in   )      :: rdt,cd_du,ff_t,sst,ch_du
      ! local variables
      real(8)                    :: CF(1:jpka)
      real(8)                    :: cff,cff1,cff2
      real(8)                    :: a(jpka),b(jpka),c(jpka)
      integer                    :: jk
      !
      !  Temperature
      !-------------
      do jk = 3, jpka-1
        a(jk) = - rdt * Avt(jk-1 ) / e3w( jk-1 )          ! lower-diagonal
        c(jk) = - rdt * Avt(jk   ) / e3w( jk   )          ! upper-diagonal
        b(jk) = e3t(jk) - a(jk) - c(jk)  !       diagonal
      enddo
      ! Boundary conditions
      a(2) = 0.
      c(2) = - rdt * Avt(2) / e3w(2)
      ! Homogeneous Neumann at the top
      a(jpka) = - rdt * Avt(jpka- 1 ) / e3w(jpka-1)
      c(jpka) = 0.
      b(jpka) = e3t(jpka) - a(jpka)
      do jk = 3, jpka-1
        t_abl(jk,nt_a) = e3t(jk) * t_abl(jk,nt_n)   ! initialize right-hand-side
      enddo
      b(2) = e3t(2) - c(2) + rdt * Ch_du
      t_abl (   2, nt_a) = e3t(    2 ) * t_abl  (2   , nt_n)
     &                                       + rdt*Ch_du*sst
      t_abl (jpka, nt_a) = e3t( jpka ) * t_abl  (jpka, nt_n)
      !!
      !! Matrix inversion
      !! ----------------------------------------------------------
      cff           =  1. / b(2)
      CF(2)         = - cff * c(2)
      t_abl(2,nt_a) =   cff * t_abl(2,nt_a)
      do jk = 3,jpka
        cff = 1./( b(jk) + a(jk) * CF (jk-1 ) )
        CF(jk) = - cff * c(jk)
        t_abl(jk,nt_a) = cff*(t_abl(jk,nt_a)-a(jk)*t_abl(jk-1,nt_a))
      enddo
      do jk = jpka-1,2,-1
        t_abl(jk,nt_a) = t_abl(jk,nt_a)+CF(jk)*t_abl(jk+1,nt_a)
      enddo
      !
      !  Coriolis + geostrophic guide
      !-------------
      do jk = 2, jpka
        if( MOD( kt, 2 ) == 0 ) then
          cff = ff_t * ( v_abl (jk,nt_n) - vG(jk) )
          u_abl(jk,nt_a) = u_abl(jk,nt_n) + rdt * cff
          cff = ff_t * ( u_abl (jk,nt_a) - uG(jk)  )
          v_abl(jk,nt_a) = e3t(jk) *( v_abl(jk, nt_n ) - rdt * cff )
          u_abl(jk,nt_a) = e3t(jk) *  u_abl(jk, nt_a )
        else
          cff = ff_t * ( u_abl (jk,nt_n) - uG(jk)  )
          v_abl(jk,nt_a) = v_abl(jk,nt_n) - rdt * cff
          cff = ff_t * ( v_abl (jk,nt_a) - vG(jk)  )
          u_abl(jk,nt_a) = e3t(jk) *( u_abl(jk,nt_n) + rdt * cff )
          v_abl(jk,nt_a) = e3t(jk) *  v_abl(jk,nt_a)
        endif     !
      enddo
      !
      !  Vertical diffusion for u_abl
      !-------------
      do jk = 3, jpka-1
        a(jk) = - rdt * Avm(jk-1 ) / e3w( jk-1 )  ! lower-diagonal
        c(jk) = - rdt * Avm(jk   ) / e3w( jk   )  ! upper-diagonal
        b(jk) = e3t(jk) - a(jk) - c(jk)                             !       diagonal
      enddo
      !++ Surface boundary condition
      a(2) = 0.
      c(2) = - rdt * Avm(2) / e3w(2)
      b(2) = e3t(2) - c(2) + rdt * cd_du
      !++ Top Neumann B.C.
      a( jpka ) = - rdt * Avm( jpka - 1 ) / e3w( jpka - 1)
      c( jpka ) = 0.
      b( jpka ) = e3t( jpka ) - a( jpka )
      !!
      !! Matrix inversion
      !! ----------------------------------------------------------
      cff            =   1. / b( 2 )
      CF( 2 )        =  - cff * c( 2 )
      u_abl (2,nt_a) =    cff * u_abl(2,nt_a)

      do jk = 3, jpka
        cff = 1. / ( b( jk ) + a( jk ) * CF (jk-1) )
        CF(jk) = - cff * c( jk )
        u_abl(jk,nt_a) = cff * (u_abl(jk,nt_a)-a(jk)*u_abl(jk-1,nt_a))
      enddo

      do jk = jpka-1,2,-1
        u_abl(jk,nt_a) = u_abl(jk,nt_a) + CF(jk) * u_abl(jk+1,nt_a)
      enddo
      !
      !  Vertical diffusion for v_abl
      !-------------
      do jk = 3, jpka-1
        a( jk ) = -rdt * Avm(jk-1 ) / e3w( jk-1 )   ! lower-diagonal
        c( jk ) = -rdt * Avm(jk   ) / e3w( jk   )   ! upper-diagonal
        b( jk ) = e3t(jk) - a( jk ) - c( jk )       !       diagonal
      enddo
      !++ Surface boundary condition
      a( 2 ) = 0.
      c( 2 ) = - rdt * Avm(2) / e3w( 2 )
      b( 2 ) = e3t( 2 ) - c( 2 ) + rdt * cd_du
      !++ Top Neumann B.C.
      a( jpka ) = - rdt * Avm( jpka - 1 ) / e3w( jpka - 1 )
      c( jpka ) = 0.
      b( jpka ) = e3t( jpka ) - a( jpka )
      !!
      !! Matrix inversion
      !! ----------------------------------------------------------
      cff       =  1. / b( 2 )
      CF ( 2 )  =    - cff * c( 2 )
      v_abl (2,nt_a) = cff * v_abl   ( 2, nt_a )

      do jk = 3, jpka
        cff = 1. / ( b( jk ) + a( jk ) * CF   (jk-1) )
        CF(jk) = - cff * c( jk )
        v_abl(jk,nt_a) = cff*(v_abl(jk,nt_a)-a(jk)*v_abl(jk-1,nt_a))
      enddo

      do jk = jpka-1,2,-1
        v_abl(jk,nt_a) = v_abl(jk,nt_a) + CF(jk) * v_abl(jk+1,nt_a)
      enddo
      !
      end subroutine
