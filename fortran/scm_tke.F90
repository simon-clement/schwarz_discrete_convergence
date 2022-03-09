module scm_tke
!
   implicit none
!
contains

!----------------------------------------------------------------------------------------
subroutine tke_stp( e3t     ,      &            ! Cell thickness           [m]
                    ght     ,      &            ! Depth of cell interfaces
                    u       ,      &            ! zonal velocity [m/s]
                    v       ,      &            ! meridional velocity [m/s]
                    rn2     ,      &            ! Brunt-Vaisala frequency
                    tke     ,      &            ! GLS variables TKE + length scale
                    dissl   ,      &            ! GLS variables TKE + length scale
                    avm     ,      &            ! Turbulent viscosity  [m2/s]
                    avt     ,      &            ! Turbulent diffusivity [m2/s]
                    r_D     ,      &
                    sustr   ,      &            ! zonal wind stress [m2/s2 = (N/m2) / (kg/m3) ]
                    svstr   ,      &            ! meridional wind stress  [m2/s2 = (N/m2) / (kg/m3)]
                    dt      ,      &            ! Time-step
                    Zob     ,      &            ! bottom roughness length
                    Neu_bot ,      &            ! Nature of bottom boundary condition
                    nstp    ,      &            ! time n
                    nnew    ,      &            ! time n+1
                    jpk     ,      &            ! Number of vertical grid points        )
                    ntra    ,      &
                    ngls    ,      &
                    ntime          )
       !------------------------------------------------------------------------
       integer,                                intent(in   ) :: jpk,ntra,ntime,ngls
       integer,                                intent(in   ) :: nstp
       integer,                                intent(in   ) :: nnew
       real(8),dimension( 1:jpk, ntime      ), intent(in   ) :: u
       real(8),dimension( 1:jpk, ntime      ), intent(in   ) :: v
       real(8),dimension( 0:jpk             ), intent(in   ) :: rn2
       real(8),dimension( 0:jpk             ), intent(inout) :: Avm
       real(8),dimension( 0:jpk, ntra       ), intent(inout) :: Avt
       real(8),dimension( 0:jpk, ntime      ), intent(inout) :: tke
       real(8),dimension( 0:jpk, ntime      ), intent(inout) :: dissl
! Grid variables
       real(8),dimension( 1:jpk             ), intent(in   ) ::  e3t,ght
       real(8),                                intent(in   ) ::  sustr
       real(8),                                intent(in   ) ::  svstr
       real(8),                                intent(in   ) ::  dt
       real(8),                                intent(in   ) ::  r_D,Zob
       logical,                                intent(in   ) ::  Neu_bot
       !------------------------------------------------------------------------
! local variables
       integer           ::  jk,tind,jpkm1
       real(8)           ::  zdiag(0:jpk),zd_lw(0:jpk),zd_up(0:jpk),q(0:jpk)
       real(8)           ::  zmxlm(0:jpk),zmxld(0:jpk)
       real(8)           ::  z3du(1:jpk-1),z3dv(1:jpk-1),apdlr(0:jpk)
       real(8)           ::  ri_cri,taum,zesh2,zri,taub
       real(8)           ::  zfact1,zfact2,zfact3,cff,p
       real(8),parameter ::  vkarmn = 0.4
       real(8),parameter ::  avmb   =   1e-04
       real(8),parameter ::  avtb   = 0.1e-04
       real(8)           ::  rmxl_min
       real(8),parameter ::  rn_ebb    = 67.83
       real(8),parameter ::  rn_ediff  = 0.1
       real(8),parameter ::  rn_ediss  = 0.5 * sqrt(2.)
       real(8),parameter ::  rn_emin   = 1.0e-06      ! minimum value of tke
       real(8),parameter ::  rn_emin0  = 1.0e-04      ! surface minimum value of tke
       real(8),parameter ::  rn_lc     =   0.15
       real(8),parameter ::  rn_mxl0   =   0.04
       real(8),parameter ::  rn_bshear =   1.e-20
       real(8),parameter ::  rsmall    =   1.e-20
       logical           ::  ln_lc     = .false.
       real(8),parameter ::  grav      =   9.81
       real(8)           ::  zcof, zzd_up, zzd_lw, zemlm, zemlp, zsqen, zav, zraug, zrn2
       !--------------------------------------------------
       jpkm1    = jpk - 1
       rmxl_min = avmb / ( rn_ediff * sqrt(rn_emin) )   !++ set minimum mixing length so that we recover the background value
       ri_cri   = 2.    / ( 2. + rn_ediss / rn_ediff )  !++ critical Richardson number
       !--------------------------------------------------
       !
       taum   = sqrt( sustr**2+svstr**2 )
       taub   = r_D * sqrt( u(1,nstp)**2 + v(1,nstp)**2  )
       !
       zfact1 = -0.5 * dt
       ! zfact2 =  1.5 * dt * rn_ediss
       ! zfact3 =  0.5      * rn_ediss
       zfact2 =  1.0 * dt * rn_ediss
       zfact3 =  0.0      * rn_ediss
       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       !                     !  Surface boundary condition on tke
       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       tke(1:jpkm1,nnew) = rn_emin
       tke(  jpk  ,nnew) = MAX( rn_emin0, rn_ebb * taum  )
       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       !                     !  Bottom boundary condition on tke
       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       !tke(  0,nnew) = rn_emin      ! hum hum
       tke( 0, nnew ) = MAX( rn_emin, rn_ebb * taub )
       !
       !
       tind = nstp
       DO jk=1,jpkm1    !* Shear production
          cff      =    avm(jk) / ( ght(jk+1)-ght(jk) )**2
          z3du(jk)  = cff*( u(jk+1, tind)-u(jk, tind) )**2
          z3dv(jk)  = cff*( v(jk+1, tind)-v(jk, tind) )**2
       ENDDO

       !* Prandtl number function of Richardson number  akt=pdl(Ri)*akv

	     apdlr(:) = 0.
       DO jk = 1,jpkm1
          !                                          ! shear prod. at w-point weightened by mask
          zesh2  = z3du(jk) + z3dv(jk)
          zri   = MAX( rn2(jk), 0. ) * avm(jk) / ( zesh2 + rn_bshear )  ! local Richardson number
          apdlr(jk) = MAX(  0.1,  ri_cri / MAX( ri_cri , zri ) )
          !
       END DO

       DO jk = 1, jpkm1           !* Matrix and right hand side in tke
          !
          zcof         = zfact1
          zzd_up       = zcof *( avm(jk+1)+avm(jk  ) ) / ( e3t(jk+1)*( ght(jk+1)-ght(jk  ) ) )
          zzd_lw       = zcof *( avm(jk  )+avm(jk-1) ) / ( e3t(jk  )*( ght(jk+1)-ght(jk  ) ) )
          zesh2        = z3du(jk) + z3dv(jk)
          zd_up(jk)    = zzd_up
          zd_lw(jk)    = zzd_lw
          zdiag(jk)    = 1. - zzd_lw - zzd_up + zfact2 * dissl(jk,nstp)
          tke(jk,nnew) = tke(jk,nstp)  &
                       + dt*( zesh2 - avt(jk,1) * rn2(jk) + zfact3 * dissl(jk,nstp)*tke(jk,nstp) )
          !
       END DO

       !* top and bottom Dirichlet boundary condition
       cff = 1. ; IF(Neu_bot) cff = 0.

       tke(    1,nnew) = tke(    1,nnew) - cff * zd_lw(    1) * tke(  0,nnew)
       tke(jpk-1,nnew) = tke(jpk-1,nnew) -       zd_up(jpk-1) * tke(jpk,nnew)

       !* Matrix inversion (Thomas algorithm)

       cff         =      1./zdiag(1  )
       q  (1     ) = - cff * zd_up(1  )
       tke(1,nnew) =   cff * tke  (1  ,nnew)

       DO jk=2,jpkm1
          p  = 1.0/( zdiag(jk)+zd_lw(jk)*q(jk-1) )
          q(jk)        = -zd_up(jk)*p
          tke(jk,nnew) = p*( tke(jk,nnew)-zd_lw(jk)*tke(jk-1,nnew) )
       ENDDO

       DO jk=jpkm1-1,1,-1
          tke(jk,nnew)=tke(jk,nnew)+q(jk)*tke(jk+1,nnew)
       END DO

       tke(0:jpk,nnew)=MAX(tke(0:jpk,nnew),rn_emin)

       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       !                     !  Mixing length
       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       !
       !                     !* Buoyancy length scale: l=sqrt(2*e/n**2)
       zmxlm(0:jpk)  = rmxl_min
       zmxld(0:jpk)  = rmxl_min
       !
       DO jk = 1, jpkm1              ! interior value : l=sqrt(2*e/n^2)
          zrn2      = MAX( rn2(jk) , rsmall )
          zmxlm(jk) = MAX( rmxl_min,  SQRT( 2. * tke(jk,nnew) / zrn2 )  )
       END DO
       !
       !                     !* Physical limits for the mixing length
       !
       ! Limit mxl
       !
       zmxld(0  ) = 0.
       DO jk = 1, jpk
            zmxld(jk) = MIN( zmxld(jk-1) + e3t(jk  ) , zmxlm(jk) )   !<-- ldwn
       END DO

       ! surface mixing length = F(stress)
       zraug      = vkarmn * 2.e5 / grav
       zmxlm(jpk) = MAX( rn_mxl0, zraug * taum )

       DO jk = jpkm1,0,-1
            zmxlm(jk) = MIN( zmxlm(jk+1) + e3t(jk+1) , zmxlm(jk) )   !<-- lup
       END DO
       zmxlm(jpk) = 0.   !<-- ensures that avm(jpk) = 0.


       !
       DO jk = 0, jpk
          zemlm     = MIN ( zmxld(jk),  zmxlm(jk) )
          zemlp     = SQRT( zmxld(jk) * zmxlm(jk) )
          zmxlm(jk) = zemlm
          zmxld(jk) = zemlp
       END DO

       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       !                     !  Vertical eddy viscosity and diffusivity  (avmu, avmv, avt)
       !                     !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

       DO jk = 0, jpk            !* vertical eddy viscosity & diffivity at w-points
          zsqen          = SQRT( tke(jk,nnew) )
          zav            = rn_ediff * zmxlm(jk) * zsqen
          avm  (jk  )    = MAX(             zav,avmb )
          avt  (jk,1)    = MAX( apdlr(jk) * zav,avtb )
          avt  (jk,2)    = MAX( apdlr(jk) * zav,avtb )
          dissl(jk,nnew) = zsqen / zmxld(jk)
          dissl(jk,nstp) = zsqen / zmxld(jk)
       END DO

       return

end subroutine tke_stp


end module scm_tke
