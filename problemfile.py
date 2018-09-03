
filename_mesh = r'%(inputfile)s'
    
material_2 = {
    'name' : 'coef',
    'values' : {'val' : 1.0},
}

regions = {
    'Omega' : 'all', 
    'Active' : ('vertices of group 1', 'vertex'),
    'Ground' : ('vertices of group 2', 'vertex'),
}

field_1 = {
    'name' : 'temperature',
    'dtype' : 'real',
    'shape' : (1,),
    'region' : 'Omega',
    'approx_order' : 1,
}

variable_1 = {
    'name' : 't',
    'kind' : 'unknown field',
    'field' : 'temperature',
    'order' : 0, # order in the global vector of unknowns
}

variable_2 = {
    'name' : 's',
    'kind' : 'test field',
    'field' : 'temperature',
    'dual' : 't',
}

ebc_1 = {
    'name' : 't1',
    'region' : 'Active',
    'dofs' : {'t.0' : 0.0},
}

ebc_2 = {
    'name' : 't2',
    'region' : 'Ground',
    'dofs' : {'t.0' : 1.0},
}

integral_1 = {
    'name' : 'i',
    'order' : 2,
}

equations = {
    'Temperature' : """dw_laplace.i.Omega( coef.val, s, t ) = 0"""
}

solver_0 = {
    'name' : 'ls',
    'kind' : 'ls.scipy_direct',
    'method' : 'auto',
}

solver_1 = {
    'name' : 'newton',
    'kind' : 'nls.newton',

    'i_max'      : 1,
    'eps_a'      : 1e-10,
    'eps_r'      : 1.0,
    'macheps'   : 1e-16,
    'lin_red'    : 1e-2, # Linear system error < (eps_a * lin_red).
    'ls_red'     : 0.1,
    'ls_red_warp' : 0.001,
    'ls_on'      : 1.1,
    'ls_min'     : 1e-5,
    'check'     : 0,
    'delta'     : 1e-6,
}

options = {
    'nls' : 'newton',
    'ls' : 'ls',
    'output_format' : 'vtk',
    'output_dir':r'%(outdir)s',
}
