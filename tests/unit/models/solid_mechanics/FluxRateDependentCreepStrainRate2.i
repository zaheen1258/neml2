[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/phi_dot'
    input_Scalar_values = '1.5'
    input_SR2_names = 'state/internal/S'
    input_SR2_values = 'S'
    output_SR2_names = 'state/internal/Ephi_rate'
    output_SR2_values = 'Ephi_rate'
    derivative_abs_tol = 1e-6
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
 [s1]
    type = FillSR2
    values = '0 0 0 2 6 3'
  []
  [s2]
    type = FillSR2
    values = '0 0 0 -2 -6 3'
  []
  [S]
    type = LinspaceSR2
    start = 's1'
    end = 's2'
    nstep = 3
  []
  [Ephi_rate]
    type = SR2
    values = "-36.5922  -9.2676  45.8598   0.0000   0.0000   0.0000
              -18.0000   0.0000  18.0000   0.0000   0.0000   0.0000
              -36.5922  -9.2676  45.8598   0.0000   0.0000   0.0000"
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = FluxRateDependentCreepStrainRate
    stress = 'state/internal/S'
    neutron_flux_rate = 'state/internal/phi_dot'
    param_1 = 2
    param_2 = 1
    creep_strain_rate = 'state/internal/Ephi_rate'
  []
[]
