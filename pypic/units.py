def info(key, selection, species=None, coord='phys'):
    """ Get the units and range for a given key. """
    if isinstance(key, list):
        # return [info(k, selection, species, coord) for k in key]
        return info(key[0], selection, species, coord)
    if key is None or len(key) < 1:
        return '', 1., [0, 1]
    key = key.replace('_', '') # remove underscores
    key = key.lower() # make lowercase
    # remove last digit (typically for species)
    # species = None if key[-1].isdigit() else species
    key = key[:-1] if key[-1].isdigit() else key
    sim = selection.sim
    species = selection.species if species is None else species
    opt = dict(species=species, coord=coord)
    # ================== POSITION
    if key in ['xphys', 'x']:
        val_range = [selection.min_phys[0], selection.max_phys[0]]
        units, scale = sim.get_units('xphys', **opt)
    if key in ['yphys', 'y']:
        val_range = [selection.min_phys[1], selection.max_phys[1]]
        units, scale = sim.get_units('yphys', **opt)
    if key in ['zphys', 'z']:
        val_range = [selection.min_phys[2], selection.max_phys[2]]
        units, scale = sim.get_units('zphys', **opt)
    if key in ['xcode']:
        val_range = [selection.min_code[0], selection.max_code[0]]
        units, scale = sim.get_units('xcode', **opt)
    if key in ['ycode']:
        val_range = [selection.min_code[1], selection.max_code[1]]
        units, scale = sim.get_units('ycode', **opt)
    if key in ['zcode']:
        val_range = [selection.min_code[2], selection.max_code[2]]
        units, scale = sim.get_units('zcode', **opt)
    if key in ['rcode']:
        val_range = [0, 50]
        units, scale = sim.get_units('rcode', **opt)
    if key in ['rphys', 'r']:
        val_range = [0, 30]
        units, scale = sim.get_units('r', **opt)
    if key in ['theta']:
        val_range = [-np.pi/2, np.pi/2]
        units, scale = sim.get_units('theta', **opt)
    if key in ['phi']:
        val_range = [0, 2*np.pi]
        units, scale = sim.get_units('phi', **opt)
    # ================== MAGNETIC FIELD
    if key in ['bx', 'by', 'bz', 'bdip', 'bxdip', 'bydip', 'bzdip',
               'bzsubbzdip', 'bxsubbxdip', 'bysubbydip', 'bsubbdip',
               'bpar', 'bperp', 'bperp1', 'bperp2',
               'br', 'btheta', 'bphi']:
        val_range = [-20, 20]
        units, scale = sim.get_units('B', **opt)
    # ================== MAGNETIC FIELD STRENGTH
    elif key in ['b']:
        val_range =  [0, 100]
        units, scale = sim.get_units('B', **opt)
    # ================== ELECTRIC FIELD
    elif key in ['e', 'ex', 'ey', 'ez',
                 'epar', 'eperp', 'eperp1', 'eperp2',
                 'er', 'etheta', 'ephi']:
        val_range = [-100, 100]
        units, scale = sim.get_units('E', **opt)
    # ================== ExB ENERGY FLUX
    elif key in ['exb', 'exbx', 'exby', 'exbz',
                 'exbpar', 'exbperp', 'exbperp1', 'exbperp2',
                 'exbr', 'exbtheta', 'exbphi']:
        val_range = [-1, 1]
        units, scale = sim.get_units('eflux', **opt)
    # ================== J.E ENERGY DENSITY
    elif key in ['jdote', 'jdotex', 'jdotey', 'jdotez',
                 'jdotepar', 'jdoteperp', 'jdoteperp1', 'jdoteperp2',
                 'jdotebr', 'jdotetheta', 'jdotephi']:
        val_range = [0, 1]
        units, scale = sim.get_units('eden',  **opt)
    # ================== JxB FORCE DENSITY
    elif key in ['jxb', 'jxbx', 'jxby', 'jxbz',
                 'jxbpar', 'jxbperp', 'jxbperp1', 'jxbperp2',
                 'jxbbr', 'jxbtheta', 'jxbphi']:
        val_range = [0, 1]
        units, scale = sim.get_units('forcedensity',  **opt)
    # ================== rhoE FORCE DENSITY
    elif key in ['rhoe', 'rhoex', 'rhoey', 'rhoez',
                 'rhoepar', 'rhoeperp', 'rhoeperp1', 'rhoeperp2',
                 'rhoeb', 'rhoetheta', 'rhoephi']:
        val_range = [0, 1]
        units, scale = sim.get_units('forcedensity',  **opt)
    # ================== CURRENTS
    elif key in ['j', 'jt']:
        # val_range =  [0, 10]
        val_range =  [0, 4]
        units, scale = sim.get_units('j', **opt)
    # ================== CURRENTS
    elif key in ['jx', 'jy', 'jz', 'jtx', 'jty', 'jtz',
                 'jpar', 'jperp', 'jperp1', 'jperp2',
                 'jtpar', 'jtperp', 'jtperp1', 'jtperp2',
                 'jtr', 'jttheta', 'jtphi',
                 'jr', 'jtheta', 'jphi',
                 'j0', 'j1',]:
        # val_range =  [-20, 20]
        val_range =  [-4, 4]
        units, scale = sim.get_units('j', **opt)
    # ================== VELOCITIES
    elif key in ['v', 'vx', 'vy', 'vz',
                 'vpar', 'vperp', 'vperp1', 'vperp2',
                 'vr', 'vtheta', 'vphi',
                 'vexb', 'vexbx', 'vexby', 'vexbz',
                 'vexbpar', 'vexbperp', 'vexbperp1', 'vexbperp2',
                 'vexbr', 'vexbtheta', 'vexbphi',
                 'vsubvexb', 'vsubvexbx', 'vsubvexby', 'vsubvexbz']:
        val_range = [0, 1500]
        units, scale = sim.get_units('v', **opt)
    # ================== THERMAL VELOCITIES
    elif key in ['vthperp', 'vthperp1', 'vthperp2', 'vthpar', 'vthtot',
                 'vth', 'vthx', 'vthy', 'vthz',
                 'vthr', 'vththeta', 'vthphi',]:
        val_range = [0, 1500]
        units, scale = sim.get_units('v', **opt)
    # ================== PARTICLE SPEEDS
    elif key in ['speed']:
        if species == 0:
            val_range = [100, 2500]
        else:
            val_range = [100, 1900]
        units, scale = sim.get_units('v', **opt)
    # ================== DENSITIES
    elif key in ['rho', 'density']:
        val_range = [0.1, 10]
        units, scale = sim.get_units('rho', **opt)
    # ================== DENSITIES PER CELL
    elif key in ['n']:
        val_range = [1, 300]
        units, scale = sim.get_units('n', **opt)
    # ================== PRESSURE
    elif key in ['p', 'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz',
                 'pyx', 'pzx', 'pzy',
                 'ppar', 'pperp', 'pperp1', 'pperp2']:
        val_range = [0, 8]
        units, scale = sim.get_units('p', **opt)
    # ================== SCALE LENGTHS
    elif key in ['scale']:
        val_range = [0, 5]
        units, scale = sim.get_units('scale', **opt)
    # ================== AGYROTROPY
    elif key in ['agy', 'vthagy']:
        val_range = [-0.5, 0.5]
        units, scale = sim.get_units('agy', **opt)
    # ================== ANISOTROPY
    elif key in ['ani', 'vthani']:
        val_range = [0, 2]
        units, scale = sim.get_units('ani', **opt)
    # ================== ENERGY FLUX
    elif key in ['ef', 'efx', 'efy', 'efz',
                 'ef0', 'efx0', 'efy0', 'efz0',
                 'efpar', 'efperp', 'efperp1', 'efperp2',
                 'efr', 'eftheta', 'efphi']:
        if species == 0:
            val_range = [-0.01, 0.01] # [0, 0.05]
        else:
            val_range = [-2, 2] # [0, 2]
        units, scale = sim.get_units('eflux', **opt)
    # ================== ENTHALPY FLUX
    elif key in ['hf', 'hfx', 'hfy', 'hfz',
                 'hf0', 'hfx0', 'hfy0', 'hfz0',
                 'hfpar', 'hfperp', 'hfperp1', 'hfperp2',
                 'hfr', 'hftheta', 'hfphi']:
        if species == 0:
            val_range = [-0.01, 0.01]
        else:
            val_range = [-2, 2]
        units, scale = sim.get_units('eflux', **opt)
    # ================== KINETIC ENERGY FLUX
    elif key in ['kef', 'kefx', 'kefy', 'kefz',
                 'kef0', 'kefx0', 'kefy0', 'kefz0',
                 'kefpar', 'kefperp', 'kefperp1', 'kefperp2',
                 'kefr', 'keftheta', 'kefphi']:
        if species == 0:
            val_range = [-0.2, 0.2]
        else:
            val_range = [-2, 2]
        units, scale = sim.get_units('eflux', **opt)
    # ================== RATIO OF KINETIC TO TOTAL ENERGY FLUX
    elif key in ['ref', 'refx', 'refy', 'refz',
                 'refpar', 'refperp', 'refperp1', 'refperp2',
                 'refr', 'reftheta', 'refphi']:
        val_range = [0, 1.0]
        units, scale = sim.get_units('ref', **opt)
    elif key in ['energy']:
        if species == 0:
            val_range = [5, 200]
        else:
            val_range = [5, 100]
        units, scale = sim.get_units('energy', **opt)
    # ================== ENTROPY
    elif key in ['entropy', 's']:
        val_range = [0, 1]
        units, scale = sim.get_units('entropy', **opt)
    # ================== TEMPERATURE
    elif key in ['temp', 'temperature', 'T']:
        val_range = [0, 1]
        units, scale = sim.get_units('T', **opt)
    # ================== CYCLE
    elif key in ['cycle']:
        val_range = selection.cycle_limits
        units, scale = sim.get_units('cycle', **opt)
    # ================== CELL IN DI
    elif key in ['di', 'dp']:
        val_range = [0, 5]
        units, scale = '', 1.
    # ================== CELL IN DE
    elif key in ['de']:
        val_range = [0, 0.6]
        units, scale = '', 1.
    # ================== OTHER
    else:
        print(f'Warning: Unknown variable key: {key}')
        val_range = [0, 1]
        units, scale = sim.get_units(None)
    return units, scale, val_range

def var_limits(key, selection, species=None, coord='phys'):
    return info(key, selection, species=None, coord='phys')[2]

def pretty_title(key, species=None, short=False, var_name=None,
                 base=None, default=None,
                 species_list = ['Electron', 'Ion',
                                 'HE Electron', 'HE Ion'],
                 LaTeX=True
                 ):
    """ Get a pretty title for a variable name """
    if not LaTeX:
        return key
    title = ''
    if species is not None:
        title += species_list[species] + ' '
    key = key.replace('_', ' ') # remove underscores
    key = key.lower() # make lowercase
    species = None
    if key[-1].isdigit(): # Remove last digit (typically for species)
        species = key[-1]
        key = key[:-1]

    if not short:
        # Adjectives like Total, Parallel or Perpendicular go first
        if 'total' in key:
            key = key.replace('total', '')
            title += r'Total' + ' '
        if 'tot' in key:
            key = key.replace('tot', '')
            title += r'Total' + ' '
        if 'par' in key:
            key = key.replace('par', '')
            title += r'Parallel' + ' '
        elif 'perp' in key:
            key = key.replace('perp', '')
            title += r'Perpendicular' + ' '

        # Enumerate possible variable names
        if key in ['bx', 'by', 'bz', 'b']:
            title += r'Magnetic Field'
        elif key in ['bdip', 'bxdip', 'bydip', 'bzdip']:
            title += r'Dipole Magnetic Field'
        elif key in ['bzsubbzdip', 'bxsubbxdip', 'bysubbydip', 'bsubbdip']:
            title += r'Magnetic Field deviation from dipole'
        elif key in ['ex', 'ey', 'ez', 'e']:
            title += r'Electric Field'
        elif key in ['de', 'dp']:
            title += r'Scale length'
        elif key in ['dce', 'dcp']:
            title += r'Gyroradius'
        elif key in ['deincell', 'dpincell']:
            title = 'Cells per ' + title + r'Scale Length'
        elif key in ['temp', 'temperature']:
            title = 'Temparature'
        elif 'vexb' in key:
            title += r'$\mathbf{E} \times \mathbf{B}$ Velocity'
            var_name = False if var_name is None else var_name
        elif 'exb' in key:
            title += r'Poynting Flux' # mW/m^2 Energy Flux
        elif 'jdote' in key:
            title += r'$\mathbf{J} \dot \mathbf{E}$' # mW/m^3 Energy Density
        elif 'jxb' in key:
            title += r'$\mathbf{J} \times \mathbf{B}$ Force Density' # N/m^3
        elif 'rhoe' in key:
            title += r'$\rho \mathbf{E}$ Force Density' # N/m^3
            var_name = False if var_name is None else var_name
        elif 'gradp' in key:
            title += r'$-\nabla P$ Force Density'
            var_name = False if var_name is None else var_name
        elif 'hf' in key:
            title += r'Enthalpy Flux'
        elif 'kef' in key:
            title += r'Kinetic Energy Flux'
        elif 'ref' in key:
            title += r'Kinetic to Total Energy Flux Ratio'
        elif 'ef' in key:
            title += r'Energy Flux'
        elif 'agy' in key:
            eqn = r'$\frac{2(v_{th_{\perp1}}-v_{th_{\perp2}})}{v_{th_{\perp1}}+v_{th_{\perp2}}}$'
            title += r'Agyrotropy' + ', ' + eqn
            var_name = False if var_name is None else var_name
        elif 'ani' in key:
            eqn = r'$\frac{2v_{th_{||}}}{v_{th_{\perp1}}+v_{th_{\perp2}}}$'
            title += r'Anisotropy' + ', ' + eqn
            var_name = False if var_name is None else var_name
        elif 'theta' in key:
            title += r'Latitude'
        elif 'phi' in key:
            title += r'Azimuth'
        elif 'rho' in key or 'density' in key:
            title += r'Density'
        elif 'energy' in key:
            title += r'Energy'
        elif 'n' in key:
            title += r'Particles per cell'
        elif 'vth' in key:
            title += r'Thermal Speed'
        elif 'v' in key:
            title += r'Velocity'
        elif 'j' in key:
            title += r'Current Density'
        elif 'p' in key:
            title += r'Pressure'
        elif 's' in key:
            title += r'Entropy'
        elif 't' in key:
            title += r'Temperature'

    var_name = True if var_name is None else var_name
    if var_name:
        title += r', ' + pretty_name(key, base=base, default=default)
    return title

def pretty_name(key, base=None, default=None, LaTeX=True):
    """ Return pretty variable names for plotting """
    # possible_names = ('vth','th','p_yz','jx0','vthperp1',
    #  'Bperp2x','v_unit','jtx','pperp2','ppar','vexbx',
    #  'vth_agy','vth_ani','rho','Bx','b_x','vth_add_vexb',
    #  'b3d','vth_tot', 'vth_phys' ,'vth_GSM')
    # try:
    if not LaTeX:
        return key
    if isinstance(key, list):
        # return [pretty_name(k, base, default) for k in key]
        return pretty_name(key[0], base, default)
    if default is not None:
        return default
    if key is None or len(key) < 1:
        return ''
    key = key.replace('_', '') # remove underscores
    key = key.lower() # make sure all lower case
    species = None
    if key in ['dp', 'di']:
        return r"$\text{Cells in }d_i$"
    if key in ['de']:
        return r"$\text{Cells in }d_e$"
    if key in ['ef', 'efx', 'efy', 'efz']:
        axis = key[-1] if key[-1] in 'xyz' else ''
        # return r"$\text{EF}_" + f"{{{axis}}}" + r"$"
        txt = f"\\text{{EF}}_{{{axis}}}"
        # surround txt with $ if it's not already
        return txt if txt[0] == '$' else f"${txt}$"
        # return f"\\text{{EF}}_{{{axis}}}"
    if 'sub' in key:
        splits = key.split('sub')
        base0 = pretty_name(splits[0]).replace('$', '')
        base1 = pretty_name(splits[1]).replace('$', '')
        return r"${{{a}}}-{{{b}}}$".format(a=base0, b=base1)
    if 'add' in key:
        splits = key.split('add')
        base0 = pretty_name(splits[0]).replace('$', '')
        base1 = pretty_name(splits[1]).replace('$', '')
        return r"${{{a}}}+{{{b}}}$".format(a=base0, b=base1)

    if key[-1].isdigit():
        species = key[-1]
        key = key[:-1]
    if base is not None:
        base = base.replace('$', '')

    if 'exb' in key:
        key = key.replace('exb', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        if len(base) > 0:
            short_name = r"${{{a}}}_{{, E \times B}}$".format(a=base)
        else:
            short_name = r"$E \times B$"
    elif 'jdote' in key:
        key = key.replace('jdote', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        if len(base) > 0:
            short_name = r"${{{a}}}_{{, J \dot E}}$".format(a=base)
        else:
            short_name = r"$J \dot E$"
    elif 'jxb' in key:
        key = key.replace('jxb', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        if len(base) > 0:
            short_name = r"${{{a}}}_{{, j \times B}}$".format(a=base)
        else:
            short_name = r"$j \times B$"
    elif 'rhoe' in key:
        key = key.replace('rhoe', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        if len(base) > 0:
            short_name = r"${{{a}}}_{{, \rho \mathbf{{E}}}}$".format(a=base)
        else:
            short_name = r"$\rho \mathbf{E}$"
    elif 'unit' in key:
        key = key.replace('unit', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"$\hat{{{a}}}$".format(a=base)
    elif 'agy' in key:
        key = key.replace('agy', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_{{,\text{{agy}}}}$".format(a=base)
    elif 'ani' in key:
        key = key.replace('ani', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_{{,\text{{ani}}}}$".format(a=base)
    elif 'par' in key:
        key = key.replace('par', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_\parallel$".format(a=base)
    elif 'perp' in key:
        key = key.replace('perp', '')
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_\perp$".format(a=base)
    elif 'theta' in key:
        key = key.replace('theta', '')
        if len(key) > 0:
            key = pretty_name(key)
            base = key.replace('$', '') if base is None else base
            short_name = r"${{{a}}}_\theta$".format(a=base)
        else:
            short_name = r"$\theta$"
    elif 'phi' in key:
        key = key.replace('phi', '')
        if len(key) > 0:
            key = pretty_name(key)
            base = key.replace('$', '') if base is None else base
            short_name = r"${{{a}}}_\phi$".format(a=base)
        else:
            short_name = r"$\phi$"
    elif 'rho' in key:
        key = key.replace('rho', '')
        if len(key) > 0:
            key = pretty_name(key)
            base = key.replace('$', '') if base is None else base
            short_name = r"${{{a}}}_\rho$".format(a=base)
        else:
            short_name = r"$\rho$"
    elif 'kappa' in key:
        key = key.replace('kappa', '')
        if len(key) > 0:
            key = pretty_name(key)
            base = key.replace('$', '') if base is None else base
            short_name = r"${{{a}}}_\kappa$".format(a=base)
        else:
            short_name = r"$\kappa$"
    elif 'energy' in key:
        key = key.replace('energy', '')
        if len(key) > 0:
            key = pretty_name(key)
            base = key.replace('$', '') if base is None else base
            # short_name = r"${{{a}}}_E$".format(a=base)
            short_name = r"$E_{{{a}}}$".format(a=base)
        else:
            short_name = r"$E$"
    elif len(key) > 4 and key[-4:] in ('phys', 'code', 'cell'):
        last_char = key[-4:]
        key = key[:-4]
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_\text{{,{b}}}$".format(a=base, b=last_char)
    elif len(key) > 3 and key[-3:] in ('tot', 'dip', 'GSM', 'gsm', 'GSE', 'gse'):
        last_char = key[-3:]
        key = key[:-3]
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_\text{{,{b}}}$".format(a=base, b=last_char)
    elif len(key) > 2 and key[-2:] in ('th', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz',
                                       'yx', 'zx', 'zy', '2D', '3D', '2d', '3d'):
        last_char = key[-2:]
        key = key[:-2]
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_\text{{{b}}}$".format(a=base, b=last_char)
    elif key[-1] in ('x', 'y', 'z', 'r', 't'):
        last_char = key[-1]
        key = key[:-1]
        key = pretty_name(key)
        base = key.replace('$', '') if base is None else base
        short_name = r"${{{a}}}_{{{b}}}$".format(a=base, b=last_char)
    else:
        short_name = base if base is not None else r"$\text{{{a}}}$".format(a=key)

    if species is not None:
        short_name = short_name.replace('$', '')
        short_name = r"${{{a}}}_{b}$".format(a=short_name, b=species)
    return short_name
    # except Exception as e:
    #     print(f'Error in pretty_name: {e}')
    #     return key


def pretty_time(t, units='s', LaTeX=True):
    """ Convert time in seconds to minutes and seconds string """
    if units == 's':
        mins = int(t // 60)
        secs = int(t - 60*mins)
        if LaTeX:
            return f'{mins:01d}'+r'$\text{min}$$:$'+f'{secs:2d}'+r'$\text{s}$'
        else:
            return f'{mins:01d}'+r' min : '+f'{secs:2d}'+r' s'
    else:
        raise ValueError('units must be "s" or "ms"')

