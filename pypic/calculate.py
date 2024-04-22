import numpy as np
from pypic.input_output import *
from scipy.interpolate import RegularGridInterpolator

def calculate_fields(selection, interp=False):
    sp = selection.species
    sim = selection.sim
    qom = sim.qoms[sp]
    qom0 = sim.qoms[0]
    small = 1e-12 #1e-12 * qom

    # save_hdf5_slice(selection, dtype='f4')
    # exit()

    # load_keys = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez',
    #              f'Jx_0', f'Jy_0', f'Jz_0',
    #              f'Jx_1', f'Jy_1', f'Jz_1',
    #              f'rho_0', f'rho_1',
    #              f'N_0', f'N_1',
    #              f'EFx_{sp}', f'EFy_{sp}', f'EFz_{sp}',
    #              f'Pxx_{sp}', f'Pyy_{sp}', f'Pzz_{sp}', f'Pxy_{sp}', f'Pxz_{sp}', f'Pyz_{sp}',
    #              ]
    # fields, _ = load_fields(selection, keys=load_keys)
    # fields = dict(zip(load_keys, fields))
    # B_fields = [fields['Bx'], fields['By'], fields['Bz']]
    # E_fields = [fields['Ex'], fields['Ey'], fields['Ez']]
    # j0_fields = [fields[f'Jx_0'], fields[f'Jy_0'], fields[f'Jz_0']]
    # j_fields = [fields[f'Jx_{sp}'], fields[f'Jy_{sp}'], fields[f'Jz_{sp}']]
    # rho0_fields = [fields[f'rho_0']]
    # rho_fields = [fields[f'rho_{sp}']]
    # n0_fields = [fields[f'N_0']]
    # n_fields = [fields[f'N_{sp}']]
    # EF_fields = [fields[f'EFx_{sp}'], fields[f'EFy_{sp}'], fields[f'EFz_{sp}']]
    # p_fields = [fields[f'Pxx_{sp}'], fields[f'Pyy_{sp}'], fields[f'Pzz_{sp}'],
    #             fields[f'Pxy_{sp}'], fields[f'Pxz_{sp}'], fields[f'Pyz_{sp}']]

    B_fields, _ = load_fields(selection, keys=['Bx', 'By', 'Bz'])
    E_fields, _ = load_fields(selection, keys=['Ex', 'Ey', 'Ez'])
    j_fields, _ = load_fields(selection, keys=[f'Jx_{sp}', f'Jy_{sp}', f'Jz_{sp}'])
    j0_fields, _ = load_fields(selection, keys=[f'Jx_0', f'Jy_0', f'Jz_0'])
    rho_fields, _ = load_fields(selection, keys=[f'rho_{sp}'])
    rho0_fields, _ = load_fields(selection, keys=[f'rho_0'])
    n_fields, _ = load_fields(selection, keys=[f'N_{sp}'])
    n0_fields, _ = load_fields(selection, keys=[f'N_0'])
    EF_fields, _ = load_fields(selection, keys=[f'EFx_{sp}', f'EFy_{sp}', f'EFz_{sp}'])
    EF0_fields, _ = load_fields(selection, keys=[f'EFx_0', f'EFy_0', f'EFz_0'])
    p_fields, _ = load_fields(selection, keys=[f'Pxx_{sp}', f'Pyy_{sp}', f'Pzz_{sp}', f'Pxy_{sp}', f'Pxz_{sp}', f'Pyz_{sp}'])
    p0_fields, _ = load_fields(selection, keys=[f'Pxx_0', f'Pyy_0', f'Pzz_0', f'Pxy_0', f'Pxz_0', f'Pyz_0'])
    # print(f'Starting calculation.')

    # B3D_fields, _ = load_fields(selection, keys=['Bx', 'By', 'Bz'], cut=False)
    # print(f'shape of B3D_fields is {B3D_fields[0].shape}')

    if interp:
        B3D_fields, _ = load_fields(selection, keys=['Bx', 'By', 'Bz'], cut=False)
        n3D_fields, _ = load_fields(selection, keys=['rho_0'], cut=False)

        EF_3D_fields, _ = load_fields(selection, keys=[f'EFx_{sp}', f'EFy_{sp}', f'EFz_{sp}'], cut=False)

        # --- Rotate the field data to physical coordinates
        B3D_fieldsT = [np.transpose(_, (2, 0, 1)) for _ in B3D_fields]
        B3Dx, B3Dy, B3Dz = code_to_phys_rot(B3D_fieldsT, True) * sim.get_scale('B')

        n3D = np.transpose(n3D_fields[0], (2, 0, 1))
        n3D = np.abs(n3D) * sim.get_scale('n', 1) + 1e-12
        # --- Grid points for the field
        x3D = np.linspace(sim.max_phys[0], sim.min_phys[0], B3Dx.shape[0])
        y3D = np.linspace(sim.min_phys[1], sim.max_phys[1], B3Dy.shape[1])
        z3D = np.linspace(sim.min_phys[2], sim.max_phys[2], B3Dz.shape[2])
        # Linear Field Interpolators for each component
        f_Bx = RegularGridInterpolator((x3D, y3D, z3D), B3Dx, bounds_error=False)
        f_By = RegularGridInterpolator((x3D, y3D, z3D), B3Dy, bounds_error=False)
        f_Bz = RegularGridInterpolator((x3D, y3D, z3D), B3Dz, bounds_error=False)
        f_N = RegularGridInterpolator((x3D, y3D, z3D), n3D, bounds_error=False)
        # Interpolation Function (gives field in nT)
        f_B = lambda x: np.array([f_Bx(x)[0], f_By(x)[0], f_Bz(x)[0]])
        f_n = lambda x: f_N(x)[0]

        # EF_3D_fields = [np.transpose(_, (2, 0, 1)) for _ in EF_3D_fields]
        # EFx_3D, EFy_3D, EFz_3D = code_to_phys_rot(EF_3D_fields, True)*sim.get_scale('eflux')
        EFx_3D, EFy_3D, EFz_3D = EF_3D_fields
        B3Dx, B3Dy, B3Dz = B3D_fields

        selection.f_B = f_B
        selection.f_n = f_n
        # selection.tmp = EFx_3D
        # selection.tmp2 = EFy_3D
        # selection.tmp3 = EFz_3D
        selection.tmp = B3Dx
        selection.tmp2 = B3Dy
        selection.tmp3 = B3Dz

    # --- SMOOTH FIELDS

    # EF_fields = [fill_field_boundaries(EF_fields, selection, value=None, radius=[2.5, 3])]
    fill_opts = {'selection': selection, 'value': None, 'radius': [3.52, 4]}
    nan_opts = {'selection': selection, 'value': np.nan, 'radius': [3.52, 3.6]}
    EF_fields = fill_field_boundaries(EF_fields, **fill_opts)
    EF_fields = [smooth_field(_, std=1) for _ in EF_fields]
    EF_fields = fill_field_boundaries(EF_fields, **nan_opts)

    EF0_fields = fill_field_boundaries(EF0_fields, **fill_opts)
    EF0_fields = [smooth_field(_, std=3) for _ in EF0_fields]
    EF0_fields = fill_field_boundaries(EF0_fields, **nan_opts)

    E_fields = fill_field_boundaries(E_fields, **fill_opts)
    E_fields = [smooth_field(_, std=2) for _ in E_fields]
    E_fields = fill_field_boundaries(E_fields, **nan_opts)
    
    B_fields = fill_field_boundaries(B_fields, **fill_opts)
    B_fields = [smooth_field(_, std=2) for _ in B_fields]
    B_fields = fill_field_boundaries(B_fields, **nan_opts)

    j_fields = fill_field_boundaries(j_fields, **fill_opts)
    j_fields = [smooth_field(_, std=4) for _ in j_fields]
    j_fields = fill_field_boundaries(j_fields, **nan_opts)

    j0_fields = fill_field_boundaries(j0_fields, **fill_opts)
    j0_fields = [smooth_field(_, std=4) for _ in j0_fields]
    j0_fields = fill_field_boundaries(j0_fields, **nan_opts)

    rho_fields = fill_field_boundaries(rho_fields, **nan_opts)
    rho0_fields = fill_field_boundaries(rho0_fields, **nan_opts)
    # rho_fields = [smooth_field(_, std=1) for _ in rho_fields]
    # rho0_fields = [smooth_field(_, std=1) for _ in rho0_fields]
    p_fields = fill_field_boundaries(p_fields, **fill_opts)
    p_fields = [smooth_field(_, std=2) for _ in p_fields]
    p_fields = fill_field_boundaries(p_fields, **nan_opts)

    p0_fields = fill_field_boundaries(p0_fields, **fill_opts)
    p0_fields = [smooth_field(_, std=2) for _ in p0_fields]
    p0_fields = fill_field_boundaries(p0_fields, **nan_opts)

    # --- SPLIT INTO COMPONENTS
    bx, by, bz = B_fields
    ex, ey, ez = E_fields
    jx, jy, jz = j_fields
    jx0, jy0, jz0 = j0_fields
    rho, rho0 = rho_fields[0], rho0_fields[0]
    n, n0 = n_fields[0], n0_fields[0]
    EFx, EFy, EFz = EF_fields
    EFx0, EFy0, EFz0 = EF0_fields
    pxx, pyy, pzz, pxy, pxz, pyz = p_fields
    pxx0, pyy0, pzz0, pxy0, pxz0, pyz0 = p0_fields

    EF = np.sqrt(EFx**2 + EFy**2 + EFz**2)
    EF0 = np.sqrt(EFx0**2 + EFy0**2 + EFz0**2)
    b = np.sqrt(bx**2 + by**2 + bz**2)
    rho = np.abs(rho)
    rho0 = np.abs(rho0)
    rho += small
    rho0 += small
    b += small
    #gbx,gby = np.gradient(b)

    # --- CALCULATE DIPOLE FIELD
    bx_dip, by_dip, bz_dip = sim.calculate_magnetic_dipole(center=[-0.1,0.1,0])
    # print(f'shape of bz is {bz.shape}')
    # print(f'shape of bz_dip is {bz_dip.shape}')
    bz_dip = np.transpose(bz_dip, (1, 2, 0))

    # bz_dip = phys_to_code_rot(bz_dip)
    # print(f'shape of bz_dip is {bz_dip.shape}')
        # n3D = np.transpose(n3D_fields[0], (2, 0, 1))
    # bz_dip_2D = cut_field(phys_to_code_rot(bz_dip), selection, transpose=False, clip=True)
    # bz_dip_2D = cut_field(phys_to_code_rot(bz_dip), selection, transpose=True, clip=True)
    bz_dip_2D = cut_field(bz_dip, selection, transpose=True, clip=True)
    # print(f'shape of bz_dip_2D is {bz_dip_2D.shape}')
    # exit()
    # bz_dip = sim.get_scale('B', sp)
    bxpic, bypic, bzpic = code_to_phys_rot([bx, by, bz], True)*sim.get_scale('B', sp)
    # bxpic, bypic, bzpic = code_to_phys_rot([bx, by, bz], True)
    bz_sub_bzdip = bzpic - bz_dip_2D
    # bz_sub_bzdip = None

    # --- ELECTRON SCALE LENGTH IN CELLS
    dx = sim.cell_in_phys_units[0] * sim.unit_phys

    # de = sim.f_de(rho0*sim.code_n*sim.scaling[0]**3)/(dx)
    de = sim.f_de(rho0*4*np.pi*sim.code_n)*sim.scaling[0]/dx
    # de = sim.f_de(rho0*4.*np.pi, 1.)*sim.code_space*1.3/dx
    # de2 = sim.f_de(rho0*4*np.pi, 1.)*sim.code_space/dx

    # rd = np.divide(de2,de)
    # print_var(rd, name='rd')
    # exit()

    # dp = sim.f_dp(rho *4*np.pi*sim.code_n)*sim.scaling[0]/(0.25*sim.unit_phys)
    dp = sim.f_dp(rho *4*np.pi*sim.code_n)*sim.scaling[0]/dx
    cell_code = sim.size_code[0]/sim.size_cell[0] # 184/460=0.4
    # dp = sim.f_dp(rho * sim.code_n)*sim.scaling[0]/dx
    wpp_code = np.sqrt(rho*4*np.pi*1.0**2/(1.0))
    wpe_code = np.sqrt(rho0*4*np.pi*1.0**2/(1.0/sim.mrcode))
    dp_cgs = 1./wpp_code
    de_cgs = 1./wpe_code
    dp = dp_cgs/cell_code
    de = de_cgs/cell_code

    # de1 = sim.f_de(1e-2*1e6)*sim.scaling[0]
    # print(f'de for n=1e-2 is {de1/1000} km')
    # print(f'de for n=1e-2 is {de1/dx} cells in de')
    # exit()
    # dp = de

    # dp = sim.f_dp(rho *sim.code_n)*sim.code_space/(dx)
    # dp = sim.f_dp(rho)*sim.code_space/(0.1)

    # proton inertial length
    # dp = sim.f_dp(rho)
    # dp = dp * sim.code_space/sim.unit_phys
    # dp = dp/sim.unit_phys

    # --- PRESSURE
    pxx = (pxx - jx*jx / (rho+small)) /qom
    pyy = (pyy - jy*jy / (rho+small)) /qom
    pzz = (pzz - jz*jz / (rho+small)) /qom
    pxy = (pxy - jx*jy / (rho+small)) /qom
    pxz = (pxz - jx*jz / (rho+small)) /qom
    pyz = (pyz - jy*jz / (rho+small)) /qom
    p = pxx + pyy + pzz
    b2D = bx*bx + by*by + small
    b = bz*bz + b2D
    perp2x = bz*bx/np.sqrt(b*b2D)
    perp2y = bz*by/np.sqrt(b*b2D)
    perp2z = -np.sqrt(b2D/b)

    ppar = (bx*pxx*bx
            + by*pyy*by
            + bz*pzz*bz
            + 2*bx*pxy*by
            + 2*bx*pxz*bz
            + 2*by*pyz*bz
            )/b
    pperp1 = (by*pxx*by
              + bx*pyy*bx
              - 2*by*pxy*bx
              )/b2D
    pperp2 = (perp2x*pxx*perp2x
              + perp2y*pyy*perp2y
              + perp2z*pzz*perp2z
              + 2*perp2x*pxy*perp2y
              + 2*perp2x*pxz*perp2z
              + 2*perp2y*pyz*perp2z)
    # pperp2 = ppar


    pxx0 = (pxx0 - jx0*jx0 / (rho0+small)) /qom0
    pyy0 = (pyy0 - jy0*jy0 / (rho0+small)) /qom0
    pzz0 = (pzz0 - jz0*jz0 / (rho0+small)) /qom0
    pxy0 = (pxy0 - jx0*jy0 / (rho0+small)) /qom0
    pxz0 = (pxz0 - jx0*jz0 / (rho0+small)) /qom0
    pyz0 = (pyz0 - jy0*jz0 / (rho0+small)) /qom0
    p0 = pxx0 + pyy0 + pzz0

    ppar0 = (bx*pxx0*bx + 2*bx*pxy0*by + 2*bx*pxz0*bz
            +by*pyy0*by + 2*by*pyz0*bz + bz*pzz0*bz)/b
    pperp10 = (by*pxx0*by - 2*by*pxy0*bx + bx*pyy0*bx)/b2D
    pperp20 = (perp2x*pxx0*perp2x + 2*perp2x*pxy0*perp2y + 2*perp2x*pxz0*perp2z
             +perp2y*pyy0*perp2y + 2*perp2y*pyz0*perp2z
             +perp2z*pzz0*perp2z)

    # --- THERMAL VELOCITY
    uth = np.sqrt(abs(qom*pxx/rho))
    vth = np.sqrt(abs(qom*pyy/rho))
    wth = np.sqrt(abs(qom*pzz/rho))
    vth_tot = np.sqrt(uth**2 + vth**2 + wth**2)
    # T = (pxx + pyy + pzz)/3

    vthpar = np.sqrt(abs(qom*ppar/rho))
    vthperp1 = np.sqrt(abs(qom*pperp1/rho))
    vthperp2 = np.sqrt(abs(qom*pperp2/rho))
    # T_par = ppar/rho
    # vth_tot2 = np.sqrt(vthperp1**2 + vthperp2**2 + vthpar**2)
    vthpar0 = np.sqrt(abs(qom0*ppar0/rho0))
    vthperp10 = np.sqrt(abs(qom0*pperp10/rho0))
    vthperp20 = np.sqrt(abs(qom0*pperp20/rho0))

    # --- AGYROTROPY AND ANISOTROPY
    vth_agy = 2*(vthperp1-vthperp2)/(vthperp2+vthperp1+small)
    vth_agy = 1 - pperp2/pperp1
    vth_ani = 2*vthpar/(vthperp2+vthperp1+small)
    # vth_agy = smooth_field(vth_agy, std=1)
    # vth_ani = smooth_field(vth_ani, std=1)

    vth_agy0 = 2*(vthperp10-vthperp20)/(vthperp20+vthperp10+small)
    vth_agy0 = 1 - pperp20/pperp10
    vth_ani0 = 2*vthpar0/(vthperp20+vthperp10+small)
    # vth_agy0 = smooth_field(vth_agy0, std=1)
    # vth_ani0 = smooth_field(vth_ani0, std=1)
    # --- ExB VELOCITY
    # vexbx, vexby, vexbz = np.cross(E_fields, B_fields, axis=0)
    # vexbx, vexby, vexbz = vexbx/b, vexby/b, vexbz/b
    # vexb = np.sqrt(vexbx**2+vexby**2+vexbz**2)
    # test_imshow(bz, selection=selection, cmap=None, norm=None, range=[-10,10]); exit()

    # --- CURRENT DENSITY
    jtx = jx+jx0
    jty = jy+jy0
    jtz = jz+jz0
    # jtx, jty, jtz = [smooth_field(_, std=1) for _ in [jtx, jty, jtz]]
    # jt = np.sqrt(jtx**2+jty**2+jtz**2)
    # jt = np.sqrt(jx**2+jy**2+jz**2) - np.sqrt(jx0**2+jy0**2+jz0**2)

    # --- VELOCITIES
    vx = jx/rho*np.sign(sim.qoms[sp])
    vy = jy/rho*np.sign(sim.qoms[sp])
    vz = jz/rho*np.sign(sim.qoms[sp])
    vx0 = jx0/rho0*np.sign(sim.qoms[0])
    vy0 = jy0/rho0*np.sign(sim.qoms[0])
    vz0 = jz0/rho0*np.sign(sim.qoms[0])
    v = np.sqrt(vx**2 + vz**2+ vy**2) + small
    v0 = np.sqrt(vx0**2 + vz0**2+ vy0**2) + small
    # V = [vx, vy, vz]
    # V0 = [vx0, vy0, vz0]
    # V_unit = [vx/v, vy/v, vz/v]
    # V_unit = [np.divide(vx,v), np.divide(vy,v), np.divide(vz,v)]
    # V0_unit = [vx0/v0, vy0/v0, vz0/v0]
    # V0_unit = [np.divide(vx0,v0), np.divide(vy0,v0), np.divide(vz0,v0)]
    # V_unit = V/np.sqrt(vx**2+vy**2+vz**2)
    # V0_unit = V0/np.sqrt(vx0**2+vy0**2+vz0**2)

    # --- VELOCITY DIFFERENCE
    # dV = np.dot(V0, V0_unit.T)
    # dVx = vx0-vx*V0_unit[0]
    # dVy = vy0-vy*V0_unit[1]
    # dVz = vz0-vz*V0_unit[2]
    # # dV = np.sqrt(dVx**2 + dVy**2 + dVz**2)
    # dV = v0 - vx*V0_unit[0] - vy*V0_unit[1] - vz*V0_unit[2]
    # dvx = vx0-vx
    # dvy = vy0-vy
    # dvz = vz0-vz
    # dv = np.sqrt(dvx**2 + dvz**2+ dvy**2)
    # dvs = v0-v

    # v_sub_vexbx = vx - vexbx
    # v_sub_vexby = vy - vexby
    # v_sub_vexbz = vz - vexbz
    # v_sub_vexb = v - vexb
    # v_sub_vexb0 = v0 - vexb

    # --- ENTHALPY FLUX
    HFx = (1/2.)*(rho*uth*vth_tot**2)
    HFy = (1/2.)*(rho*vth*vth_tot**2)
    HFz = (1/2.)*(rho*wth*vth_tot**2)
    HF = np.sqrt(HFx**2 + HFy**2 + HFz**2)
    # HF = (1/2.)*(rho*wth**3)

    # --- KINETIC ENERGY FLUX
    KEFx = (1/2.)*(rho*vx*v**2)
    KEFy = (1/2.)*(rho*vy*v**2)
    KEFz = (1/2.)*(rho*vz*v**2)
    # KEFx, KEFy, KEFz = [smooth_field(_, std=0.5) for _ in [KEFx, KEFy, KEFz]]
    KEF = np.sqrt(KEFx**2 + KEFy**2 + KEFz**2)
    KEFx0 = (1/2.)*(rho0*vx0*v0**2)
    KEFy0 = (1/2.)*(rho0*vy0*v0**2)
    KEFz0 = (1/2.)*(rho0*vz0*v0**2)
    # KEFx0, KEFy0, KEFz0 = [smooth_field(_, std=2) for _ in [KEFx0, KEFy0, KEFz0]]
    KEF0 = np.sqrt(KEFx0**2 + KEFy0**2 + KEFz0**2)

    # --- DIFFERENCE BETWEEN TOTAL AND KINETIC ENERGY FLUX
    dEFx = EFx - KEFx
    dEFy = EFy - KEFy
    dEFz = EFz - KEFz
    dEF = np.sqrt(dEFx**2 + dEFy**2 + dEFz**2)
    dEFx0 = EFx0 - KEFx0
    dEFy0 = EFy0 - KEFy0
    dEFz0 = EFz0 - KEFz0
    # dEFx0, dEFy0, dEFz0 = [smooth_field(_, std=2) for _ in [dEFx0, dEFy0, dEFz0]]
    dEF0 = np.sqrt(dEFx0**2 + dEFy0**2 + dEFz0**2)
    HFx = dEFx
    HFy = dEFy
    HFz = dEFz
    HF = dEF
    HFx0 = dEFx0
    HFy0 = dEFy0
    HFz0 = dEFz0
    HF0 = dEF0

    # --- RATIO OF KINETIC TO TOTAL ENERGY FLUX
    # rEFx = KEFx/EF*selection.sim.momentum_corrector[sp]
    # rEFy = KEFy/EF*selection.sim.momentum_corrector[sp]
    # rEFz = KEFz/EF*selection.sim.momentum_corrector[sp]
    # rEF  = KEF/EF*selection.sim.momentum_corrector[sp]
    # rEFx, rEFy, rEFz, rEF = [smooth_field(_, std=1) for _ in [rEFx, rEFy, rEFz, rEF]]


    # --- CONVERT TO PHYSICAL COORDINATES
    # rho0 = rho0*sim.get_scale('rho', 0)/sim.scaling[0]
    # rho = rho*sim.get_scale('rho', sp)/sim.scaling[0]
    rho0 = rho0*sim.get_scale('rho', 0)
    rho = rho*sim.get_scale('rho', sp)
    ex, ey, ez = code_to_phys_rot([ex, ey, ez], True)*sim.get_scale('E', sp)
    bx, by, bz = code_to_phys_rot([bx, by, bz], True)*sim.get_scale('B', sp)
    # bz_dip = sim.get_scale('B', sp)
    # bz_sub_bzdip = sim.get_scale('B', sp)
    E  = np.sqrt(ex**2 + ey**2 + ez**2)
    b  = np.sqrt(bx**2 + by**2 + bz**2)

    EFx, EFy, EFz = code_to_phys_rot([EFx, EFy, EFz], True)*sim.get_scale('eflux', sp)
    KEFx, KEFy, KEFz = code_to_phys_rot([KEFx, KEFy, KEFz], True)*sim.get_scale('eflux', sp)
    HFx, HFy, HFz = code_to_phys_rot([HFx, HFy, HFz], True)*sim.get_scale('eflux', sp)
    EFx0, EFy0, EFz0 = code_to_phys_rot([EFx0, EFy0, EFz0], True)*sim.get_scale('eflux', 0)
    KEFx0, KEFy0, KEFz0 = code_to_phys_rot([KEFx0, KEFy0, KEFz0], True)*sim.get_scale('eflux', 0)
    HFx0, HFy0, HFz0 = code_to_phys_rot([HFx0, HFy0, HFz0], True)*sim.get_scale('eflux', 0)
    # rEFx, rEFy, rEFz = code_to_phys_rot([rEFx, rEFy, rEFz], True)*sim.get_scale('eflux', sp)
    # dEFx, dEFy, dEFz = code_to_phys_rot([dEFx, dEFy, dEFz], True)*sim.get_scale('eflux', sp)
    # dEF *= sim.get_scale('eflux', sp)
    EF *= sim.get_scale('eflux', sp)
    KEF *= sim.get_scale('eflux', sp)
    HF *= sim.get_scale('eflux', sp)
    EF0 *= sim.get_scale('eflux', 0)
    KEF0 *= sim.get_scale('eflux', 0)
    HF0 *= sim.get_scale('eflux', 0)

    jx, jy, jz = code_to_phys_rot([jx, jy, jz], True)*sim.get_scale('j', sp)
    jx0, jy0, jz0 = code_to_phys_rot([jx0, jy0, jz0], True)*sim.get_scale('j', 0)
    jtx, jty, jtz = code_to_phys_rot([jtx, jty, jtz], True)*sim.get_scale('j', sp)
    j = np.sqrt(jx**2 + jy**2 + jz**2)
    j0 = np.sqrt(jx0**2 + jy0**2 + jz0**2)
    jt = np.sqrt(jtx**2 + jty**2 + jtz**2)

    vx, vy, vz = code_to_phys_rot([vx, vy, vz], True)*sim.get_scale('v', sp)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    vx0, vy0, vz0 = code_to_phys_rot([vx0, vy0, vz0], True)*sim.get_scale('v', 0)
    v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    # vexbx, vexby, vexbz = code_to_phys_rot([vexbx, vexby, vexbz], True)*sim.get_scale('v', sp)
    # vx0, vy0, vz0 = code_to_phys_rot([vx0, vy0, vz0], True)*sim.get_scale('v', sp)
    # dvx, dvy, dvz = code_to_phys_rot([dvx, dvy, dvz], True)*sim.get_scale('v', sp)
    # v_sub_vexbx, v_sub_vexby, v_sub_vexbz = code_to_phys_rot([v_sub_vexbx, v_sub_vexby, v_sub_vexbz], True)*sim.get_scale('v', sp)
    # vthpar, vthperp1, vthperp2 = code_to_phys_rot([vthpar, vthperp1, vthperp2], True)*sim.get_scale('v', sp)
    # vexb = np.sqrt(vexbx**2 + vexby**2 + vexbz**2)
    # v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    # dv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
    # v_sub_vexb = np.sqrt(v_sub_vexbx**2 + v_sub_vexby**2 + v_sub_vexbz**2)
    # vthperp = np.sqrt(vthperp1**2 + vthperp2**2)

    ppar, pperp1, pperp2 = code_to_phys_rot([ppar, pperp1, pperp2], True)*sim.get_scale('p', sp)
    ppar0, pperp10, pperp20 = code_to_phys_rot([ppar0, pperp10, pperp20], True)*sim.get_scale('p', 0)
    pperp = np.sqrt(pperp1**2 + pperp2**2)
    pperp0 = np.sqrt(pperp10**2 + pperp20**2)
    selection.ppar = ppar
    selection.pperp = pperp
    selection.pperp1 = pperp1
    selection.pperp2 = pperp2
    selection.ppar0 = ppar0
    selection.pperp0 = pperp0
    selection.pperp10 = pperp10
    selection.pperp20 = pperp20


    selection.vth_agy = vth_agy
    selection.vth_ani = vth_ani
    selection.vth_agy0 = vth_agy0
    selection.vth_ani0 = vth_ani0
        # return self.c_phys/self.f_wpp(np)
    # --- RETURN SELECTION FIELDS
    selection.vx = vx
    selection.vy = vy
    selection.vz = vz
    selection.v = v
    selection.vx0 = vx0
    selection.vy0 = vy0
    selection.vz0 = vz0
    selection.v0 = v0
    selection.jx = jx
    selection.jy = jy
    selection.jz = jz
    selection.j = j
    selection.jx0 = jx0
    selection.jy0 = jy0
    selection.jz0 = jz0
    selection.j0 = j0
    selection.jtx = jtx
    selection.jty = jty
    selection.jtz = jtz
    selection.jt = jt
    selection.rho = rho
    selection.rho0 = rho0
    selection.de = de
    selection.dp = dp
    selection.E = E
    selection.b = b
    selection.bz = bz
    selection.bx = bx
    selection.by = by
    # print(f' shape of bz = {bz.shape}')
    # exit()
    # selection.bz_dip = bz_dip
    # selection.bz_sub_bzdip = bz_sub_bzdip
    selection.EFx = EFx
    selection.EFy = EFy
    selection.EFz = EFz
    selection.EF = EF
    selection.KEFx = KEFx
    selection.KEFy = KEFy
    selection.KEFz = KEFz
    selection.KEF = KEF
    selection.HFx = HFx
    selection.HFy = HFy
    selection.HFz = HFz
    selection.HF = HF
    selection.EFx0 = EFx0
    selection.EFy0 = EFy0
    selection.EFz0 = EFz0
    selection.EF0 = EF0
    selection.KEFx0 = KEFx0
    selection.KEFy0 = KEFy0
    selection.KEFz0 = KEFz0
    selection.KEF0 = KEF0
    selection.HFx0 = HFx0
    selection.HFy0 = HFy0
    selection.HFz0 = HFz0
    selection.HF0 = HF0

    selection.bz_sub_bzdip = bz_sub_bzdip
    selection.bz_dip = bz_dip_2D

    # --- BINS FOR PLOTTING
    # x, y = selection.xy_bins()
    # r = np.sqrt(x**2 + y**2)
    # z = np.zeros_like(x)
    # x, y, z = selection.xyz_bins()
    # r = np.sqrt(x**2 + y**2 + z**2)
    # selection.r = r
    # selection.x = x
    # selection.y = y
    # selection.z = z

    # print(f'Done calculating')
    return selection

def calculate_quick(selection, tmp):
    sim = selection.sim
    Bx = cut_field(tmp[0], selection)
    By = cut_field(tmp[1], selection)
    Bz = cut_field(tmp[2], selection)
    # EFx = cut_field(tmp[0], selection)
    # EFy = cut_field(tmp[1], selection)
    # EFz = cut_field(tmp[2], selection)
    # EFx, EFy, EFz = code_to_phys_rot([EFx, EFy, EFz], True)*sim.get_scale('eflux', selection.species)
    bx, by, bz = code_to_phys_rot([Bx, By, Bz], True)*sim.get_scale('b', selection.species)
    # EFx = np.sqrt(EFx**2 + EFy**2 + EFz**2)
    # EFx = EFy
    # EFx = smooth_field(EFx, std=1)
    # Bx = smooth_field(Bx, std=1)
    # selection.EFx = EFx
    selection.bx = bx
    selection.by = by
    selection.bz = bz
    # x, y = selection.xy_bins()
    # r = np.sqrt(x**2 + y**2)
    # x, y, z = selection.xyz_bins(field=bz)
    x, y, z = selection.xyz_bins()
    r = np.sqrt(x**2 + y**2 + z**2)
    # selection.r = r
    # print(f'shape of x = {x.shape}')
    # exit()
    selection.x = x
    selection.y = y
    selection.z = z
    return selection

def calculate_quick_3Dfields(selection):
    # print(f'Starting quick 3D calculation.')
    sp = selection.species
    sim = selection.sim
    qom = sim.qoms[sp]
    small = 1e-12*qom #1e-12

    calculate_all = True

    if calculate_all:
    # if hasattr(selection, 'f_B') and hasattr(selection, 'B3Dx'):
    #     Bx = cut_field(selection.B3Dx, selection)
    #     By = cut_field(selection.B3Dy, selection)
    #     Bz = cut_field(selection.B3Dz, selection)
    #     bx, by, bz = code_to_phys_rot([Bx, By, Bz], True)*sim.get_scale('b', selection.species)
    #     selection.bx = bx
    #     selection.by = by
    #     selection.bz = bz
    # else:
        # B_fields, _ = load_fields(selection, keys=['Bx', 'By', 'Bz'])
        B3D_fields, _ = load_fields(selection, keys=['Bx', 'By', 'Bz'], cut=False)
        # n3D_fields, _ = load_fields(selection, keys=['rho_0'], cut=False)

        # B_fields = [smooth_field(_, std=1) for _ in B_fields]
        # bx, by, bz = B_fields
        # bx, by, bz = code_to_phys_rot([bx, by, bz], True)*sim.get_scale('B', sp)

        selection.B3Dx = B3D_fields[0]
        selection.B3Dy = B3D_fields[1]
        selection.B3Dz = B3D_fields[2]
        B3D_fields = [np.transpose(_, (2, 0, 1)) for _ in B3D_fields]
        B3Dx, B3Dy, B3Dz = code_to_phys_rot(B3D_fields, True)*sim.get_scale('B')
        Bx = cut_field(selection.B3Dx, selection)
        By = cut_field(selection.B3Dy, selection)
        Bz = cut_field(selection.B3Dz, selection)
        bx, by, bz = code_to_phys_rot([Bx, By, Bz], True)*sim.get_scale('b', selection.species)

        # n3D = np.transpose(n3D_fields[0], (2, 0, 1))
        # n3D = np.abs(n3D) * sim.get_scale('n', 1) + 1e-12
        # --- Grid points for the field
        x3D = np.linspace(sim.max_phys[0], sim.min_phys[0], B3Dx.shape[0])
        y3D = np.linspace(sim.min_phys[1], sim.max_phys[1], B3Dy.shape[1])
        z3D = np.linspace(sim.min_phys[2], sim.max_phys[2], B3Dz.shape[2])
        # Linear Field Interpolators for each component
        f_Bx = RegularGridInterpolator((x3D, y3D, z3D), B3Dx, bounds_error=False)
        f_By = RegularGridInterpolator((x3D, y3D, z3D), B3Dy, bounds_error=False)
        f_Bz = RegularGridInterpolator((x3D, y3D, z3D), B3Dz, bounds_error=False)
        # f_N = RegularGridInterpolator((x3D, y3D, z3D), n3D, bounds_error=False)
        f_B = lambda x: np.array([f_Bx(x)[0], f_By(x)[0], f_Bz(x)[0]])
        # f_n = lambda x: f_N(x)[0]
        selection.f_B = f_B
        # selection.f_n = f_n
        selection.bz = bz #TODO
        # print(f'Done calculating')
    return selection


def fill_field_boundaries(fields, selection, value=None, radius=None):
    """ Mask fields inside a given radius """
    if radius is None:
        return fields
    if isinstance(fields, list):
        selection.bz = fields[0]
    else:
        selection.bz = fields
    r = getattr(selection, 'r', None)
    mask0 = r.T>radius[0]
    mask1 = r.T<radius[1]
    mask = mask0*mask1

    if r is not None:
        if isinstance(fields, list):
            for field in fields:
                avg_value = np.nanmedian(field[mask])
                if value is not None:
                    avg_value = value
                field[r.T<radius[0]] = avg_value
        else:
            avg_value = np.nanmedian(fields[mask])
            if value is not None:
                avg_value = value
            fields[r.T<radius[0]] = avg_value
    return fields

def mask_fields(fields, selection, value=None, radius=None):
    """ Mask fields inside a given radius """
    # mask_radius = 4.0   # Re
    # return fields
    if radius is None:
        return fields
        # radius = round(selection.sim.L_square / selection.sim.scaling[0], 0)
        # radius = 2.0
    r = getattr(selection, 'r', None)
    if r is not None:
        if isinstance(fields, list):
            value = 0. if value is None else value
            for field in fields:
                field[r.T<radius] = value
        else:
            value = np.nan if value is None else value
            fields[r.T<radius] = value
    return fields
