[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/phi_dot'
    input_Scalar_values = '0.0015'
    input_SR2_names = 'state/internal/S'
    input_SR2_values = 'S'
    output_SR2_names = 'state/internal/Ephi_rate'
    output_SR2_values = 'Ephi_rate'
  []
[]

[Tensors]
  [S]
    type = FillSR2
    values = '-0.3482 0.3482 0 0.087045 0.087045 0.78333'
  []
  [Ephi_rate]
    type = FillSR2
    values = '-1.7375e-06 -3.2060e-08 1.7695e-06 0.0 0.0 0.0'
  []
[]

[Models]
  [model]
    type = FluxRateDependentCreepStrainRate
    stress = 'state/internal/S'
    neutron_flux_rate = 'state/internal/phi_dot'
    param_1 = 0.001
    param_2 = 0.35
    creep_strain_rate = 'state/internal/Ephi_rate'
  []
[]
