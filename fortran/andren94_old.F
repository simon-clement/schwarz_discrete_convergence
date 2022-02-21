      program andren94
      implicit none
      integer, parameter            ::  jpka    = 41         !: number of vertical levels + 1
      real(8), parameter            ::  ff_t    = 1.e-04     !: Coriolis parameter
      real(8), parameter            ::  zrgh    = 0.1        !: roughness length
      real(8), parameter            ::  zmax    = 1500.      !: model top
      real(8), parameter            ::  ceps    = 0.845      !: constant for TKE eq.
      real(8), parameter            ::  cm      = 0.126      !: constant for TKE eq.
      real(8), parameter            ::  ce      = 0.34       !: constant for TKE eq.
      real(8), parameter            ::  Sch     = ce / cm    !: Schmidt number
      real(8), parameter            ::  tke_min = 1.e-6      !: minimum TKE
      real(8), parameter            ::  avm_bak = 1.e-4      !: background viscosity
      real(8), parameter            ::  vonKar  = 0.4        !: Von Karman constant
      real(8), parameter            ::  t_end_h = 28.        !: total simulation time in hours
      real(8), parameter            :: mxl_min  = (avm_bak / cm)    !: minimum mixing length
     &                                         / sqrt( tke_min )
      real(8)                       ::  u_abl   (1:jpka, 1:2 )  !: zonal wind
      real(8)                       ::  v_abl   (1:jpka, 1:2 )  !: meridional wind
      real(8)                       ::  avm     (1:jpka      )  !: turbulent viscosity
      real(8)                       ::  mxlm    (1:jpka      )  !: master mixing length
      real(8)                       ::  mxld    (1:jpka      )  !: dissipative mixing length
      real(8)                       ::  tke     (1:jpka, 1:2 )  !: turbulent kinetic energy
      real(8)                       ::  pu_dta  (1:jpka      )  !: zonal geostrophic wind
      real(8)                       ::  pv_dta  (1:jpka      )  !: meridional geostrophic wind
      real(8)                       ::  ght     (1:jpka      )  !: height at cell centers
      real(8)                       ::  ghw     (1:jpka      )  !: height at cell interfaces
      real(8)                       ::  e3t     (1:jpka      )  !: layer thickness
      real(8)                       ::  e3w     (1:jpka      )  !: distance between cell centers
      integer                       ::  jk,kt,ntimes,nt_n,nt_a
      real(8)                       ::  cff,ustar2,Cd_du
      real(8)                       ::  rdt,time
      character*5                   ::  suffix
      !!
      !! Define vertical grid
      !! ----------------------------------------------------------
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
      !!
      !! Initialize fields
      !! ----------------------------------------------------------
      do jk=1,jpka
         pu_dta(jk  ) = 10.d0
         pv_dta(jk  ) =  0.d0
         u_abl (jk,1) = pu_dta(jk  )
         v_abl (jk,1) = pv_dta(jk  )
         u_abl (jk,2) = pu_dta(jk  )
         v_abl (jk,2) = pv_dta(jk  )
         tke   (jk,1) = tke_min
         tke   (jk,2) = tke_min
         avm   (jk  ) = avm_bak
         mxlm  (jk  ) = mxl_min
         mxld  (jk  ) = mxl_min
      end do
      !!
      !! TIME LOOP
      !! ----------------------------------------------------------
      rdt    = 60.
      ntimes = nint(t_end_h * 3600. / rdt)

      do kt=1,ntimes+1
        time  = rdt*float(kt-1)                   !<-- time in seconds
        nt_n  = 1 + MOD( kt  , 2)             !<-- time index for prognostic variables
        nt_a  = 1 + MOD( kt+1, 2)
        !==== Compute surface boundary conditions
        cff    = u_abl(2,nt_n)**2 + v_abl(2,nt_n)**2
        ustar2 =       cff * (vonKar / log( ght(2)/zrgh ) )**2
        Cd_du  = sqrt(cff) * (vonKar / log( ght(2)/zrgh ) )**2
        !==== Compute turbulent viscosity
        Call zdf_tke(jpka,nt_n,nt_a,u_abl(:,nt_n),v_abl(:,nt_n),
     &    avm,tke,mxlm,mxld,e3w,e3t,ghw,
     &    rdt,ustar2,zrgh,avm_bak,cm,ceps,Sch,tke_min,mxl_min)
        !==== Advance single-column model by one time-step
        Call step_abl(kt,jpka,nt_n,nt_a,
     &            u_abl,v_abl,avm,pu_dta,pv_dta,e3t,e3w,rdt,cd_du,ff_t)
      enddo

      print*,'ustar = ',SQRT(ustar2)
      print*,'umax  = ',MAXVAL(ABS(u_abl(:,nt_a)))
      print*,'umin  = ',MINVAL(ABS(u_abl(:,nt_a)))
      print*,'vmax  = ',MAXVAL(ABS(v_abl(:,nt_a)))
      print*,'vmin  = ',MINVAL(ABS(v_abl(:,nt_a)))


      if(jpka ==  41) then
        suffix = 'N0040'
      elseif(jpka == 161) then
        suffix = 'N0160'
      elseif(jpka == 321) then
        suffix = 'N0320'
      elseif(jpka == 641) then
        suffix = 'N0640'
      elseif(jpka == 1281) then
        suffix = 'N1280'
      else
        suffix = 'xxxxx'
      endif

      open(1, file = 'u_final_'//trim(suffix)//'.out')
      DO jk=2,jpka
         write(1,*) u_abl(jk,nt_a), ght(jk)*ff_t/SQRT(ustar2)
      ENDDO
      close(1)

      open(1, file = 'v_final_'//trim(suffix)//'.out')
      DO jk=2,jpka
         write(1,*) v_abl(jk,nt_a), ght(jk)*ff_t/SQRT(ustar2)
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

      end program andren94


      subroutine zdf_tke(jpka,nt_n,nt_a,uu,vv,avm,tke,mxlm,mxld,e3w,
     &  e3t,ghw,rdt,ustar2,zrgh,avm_bak,cm,ceps,Sch,tke_min,mxl_min)
      implicit none
      integer                    :: jpka,nt_n,nt_a
      real(8)                    :: uu(jpka),vv(jpka),avm(jpka)
      real(8)                    :: tke(jpka,2),mxlm(jpka),mxld(jpka)
      real(8)                    :: ghw(jpka),e3w(jpka),e3t(jpka)
      real(8)                    :: rdt,zrgh,avm_bak,cm,ceps,Sch,tke_min
      real(8)                    :: ustar2,mxl_min
      ! local variables
      real(8)                    :: CF(1:jpka),shear,xx,sh2
      real(8)                    :: esfc,etop,cff,dU,dV
      real(8)                    :: a(jpka),b(jpka),c(jpka)
      real(8)                    :: lup(jpka),ldw(jpka)
      integer                    :: jk
      real(8), parameter         :: vonKar = 0.4
      logical                    :: simpleMxl = .TRUE.

      ! Compute vertical shear source term
      do jk = 2, jpka-1
        cff     = 1.0 / e3w( jk )**2
        dU      = cff* Avm(jk) * (uu(jk+1)-uu(jk) )**2
        dV      = cff* Avm(jk) * (vv(jk+1)-vv(jk) )**2
        CF(jk)  = dU+dV   !<-- sh2 = Km ( ( du/dz )^2 + ( dv/dz )^2 )
      enddo
      ! Terms for the tridiagonal problem
      do jk = 2, jpka-1
        shear   = CF( jk )
        a( jk ) = - 0.5*rdt*Sch*(Avm(jk)+Avm(jk-1)) / e3t( jk   ) ! lower-diagonal
        c( jk ) = - 0.5*rdt*Sch*(Avm(jk)+Avm(jk+1)) / e3t( jk+1 ) ! upper-diagonal
        b( jk ) = e3w(jk) - a(jk) - c(jk) + e3w(jk) * rdt *
     &            Ceps * SQRT(tke(jk,nt_n)) / mxld( jk )           ! diagonal
        tke(jk,nt_a) = e3w(jk) * ( tke(jk,nt_n) + rdt * shear )   ! right-hand-side
      end do
      ! boundary conditions
      esfc = MAX( ustar2 / SQRT( cm * ceps ), tke_min )
      etop = tke_min
      tke(1,nt_a) = esfc; a(1) = 0.;c(1)=0.;b(1)=1.
      a ( jpka ) = - 0.5*rdt*Sch*(Avm(jpka-1)+Avm(jpka))/e3t( jpka )
      c ( jpka ) = 0.
      b ( jpka ) = e3w(jpka) - a(jpka )
      tke(jpka, nt_a ) = e3w(jpka) * tke( jpka, nt_n )
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
      cff       = (vonKar / Cm) * SQRT(SQRT(Cm*ceps)) * zrgh
      lup( 1  ) = cff
      ldw( 1  ) = cff
      lup(jpka) = mxl_min
      ldw(jpka) = mxl_min

      if(SimpleMXL) then
! Simple mxl computation
        do jk = 2,jpka-1
          lup(jk) = ghw(jpka) - ghw(jk)
          ldw(jk) = ghw(jk  ) - ghw( 1)
        enddo
      else
! Advanced mxl computation
        do jk = 2, jpka-1
          cff      = 1.0 / e3w( jk )**2
          dU       = cff* (uu(jk+1)-uu(jk) )**2
          dV       = cff* (vv(jk+1)-vv(jk) )**2
          sh2      = SQRT(dU+dV)
          cff      = SQRT(tke(jk,nt_a))/( 0.175 * sh2 )
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
        !
      endif

      do jk = 1,jpka
         cff = ( 0.5*( ldw( jk )**(1./xx) + lup( jk )**(1./xx) ) )**xx
         mxlm ( jk ) = MAX( cff, mxl_min )
         mxld ( jk ) = MIN ( ldw( jk ),  lup( jk ) )
      enddo
      !!
      !!  Turbulent viscosity computation
      !! ----------------------------------------------------------
      do jk = 1, jpka
        cff     = mxlm( jk ) * SQRT( tke( jk, nt_a ) )
        avm(jk) = MAX( Cm * cff, avm_bak   )
      enddo

      end subroutine









      subroutine step_abl(kt,jpka,nt_n,nt_a,u_abl,v_abl,
     &                    avm,uG,vG,e3t,e3w,rdt,cd_du,ff_t)
      implicit none
      integer,intent(in   )      :: jpka,nt_n,nt_a,kt
      real(8),intent(inout)      :: u_abl(jpka,2),v_abl(jpka,2)
      real(8),intent(in   )      :: uG(jpka),vG(jpka)
      real(8),intent(in   )      :: avm(jpka)
      real(8),intent(in   )      :: e3w(jpka),e3t(jpka)
      real(8),intent(in   )      :: rdt,cd_du,ff_t
      ! local variables
      real(8)                    :: CF(1:jpka)
      real(8)                    :: cff
      real(8)                    :: a(jpka),b(jpka),c(jpka)
      integer                    :: jk
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
      a( jpka ) = - rdt * Avm( jpka ) / e3w( jpka )
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
      a( jpka ) = - rdt * Avm( jpka ) / e3w( jpka )
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
