// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/solid_mechanics/FluxRateDependentCreepStrainRate.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/Vec.h"
#include "neml2/tensors/SR2.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/tensors/functions/linalg/eigh.h"

namespace neml2
{
register_NEML2_object(FluxRateDependentCreepStrainRate);

OptionSet
FluxRateDependentCreepStrainRate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() =
      " This object calculates the rate of creep strain depending on the principal stresses and "
      "neutron flux rate following associative flow rule, i.e. "
      "\\f$ \\dot{\\boldsymbol{\\varepsilon}}_{\\phi} = \\begin{bmatrix} \\dot{\\varepsilon}_1 & 0 "
      "& 0 \\\\ "
      "0 & \\dot{\\varepsilon}_2 & 0 \\\\ "
      "0 & 0 & \\dot{\\varepsilon}_3 \\end{bmatrix} \\dot{\\phi} \\f$, "
      "where \\f$ \\dot{\\varepsilon}_i = \\alpha \\left(\\sigma_i - \\beta "
      "\\left(\\sigma_j + \\sigma_k \\right) \\right) \\f$.";

  options.set_input("stress") = VariableName(STATE, "internal", "S");
  options.set("stress").doc() = "Stress tensor calculated using any stress calculator";

  options.set_input("neutron_flux_rate") = VariableName(STATE, "internal", "phi_dot");
  options.set("neutron_flux_rate").doc() = "Neutron flux rate";

  options.set<TensorName<Scalar>>("param_1");
  options.set("param_1").doc() = "Scaling parameter 1";

  options.set<TensorName<Scalar>>("param_2");
  options.set("param_2").doc() = "Scaling parameter 2";

  options.set_output("creep_strain_rate") = VariableName(STATE, "internal", "Ephi_rate");
  options.set("creep_strain_rate").doc() = "Rate of creep strain";

  return options;
}

FluxRateDependentCreepStrainRate::FluxRateDependentCreepStrainRate(const OptionSet & options)
  : Model(options),
    _stress(declare_input_variable<SR2>("stress")),
    _phi_dot(declare_input_variable<Scalar>("neutron_flux_rate")),
    _alpha(declare_parameter<Scalar>("alpha", "param_1")),
    _beta(declare_parameter<Scalar>("beta", "param_2")),
    _Ephi_dot(declare_output_variable<SR2>("creep_strain_rate"))
{
}

void
FluxRateDependentCreepStrainRate::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
  auto [evals, evecs] = linalg::eigh(_stress);
  const auto Ephi_dot1 =
      _alpha * (evals.base_index({0}) - _beta * (evals.base_index({1}) + evals.base_index({2})));
  const auto Ephi_dot2 =
      _alpha * (evals.base_index({1}) - _beta * (evals.base_index({2}) + evals.base_index({0})));
  const auto Ephi_dot3 =
      _alpha * (evals.base_index({2}) - _beta * (evals.base_index({0}) + evals.base_index({1})));

  if (out)
    _Ephi_dot = SR2::fill(Scalar(Ephi_dot1), Scalar(Ephi_dot2), Scalar(Ephi_dot3)) * _phi_dot;

  if (dout_din)
  {
    const auto I = SR2::identity(_stress.options());
    auto evecs_tr = R2(evecs).transpose();

    if (_stress.is_dependent())
    {
      /// outer product of e_i x e_i
      auto n1xn1 = SR2::fill(1.0, 0.0, 0.0, _stress.options());
      auto n2xn2 = SR2::fill(0.0, 1.0, 0.0, _stress.options());
      auto n3xn3 = SR2::fill(0.0, 0.0, 1.0, _stress.options());

      /// outer product of eigenvectors v_i x v_i
      auto v1xv1 = R2(evecs).col(0).self_outer();
      auto v2xv2 = R2(evecs).col(1).self_outer();
      auto v3xv3 = R2(evecs).col(2).self_outer();

      auto de1_dstress = _alpha * _phi_dot * ((1 + _beta) * v1xv1 - _beta * I);
      auto de2_dstress = _alpha * _phi_dot * ((1 + _beta) * v2xv2 - _beta * I);
      auto de3_dstress = _alpha * _phi_dot * ((1 + _beta) * v3xv3 - _beta * I);
      _Ephi_dot.d(_stress) =
          n1xn1.outer(de1_dstress) + n2xn2.outer(de2_dstress) + n3xn3.outer(de3_dstress);
    }

    if (_phi_dot.is_dependent())
    {
      auto dErate_dphi = SR2::fill(Scalar(Ephi_dot1), Scalar(Ephi_dot2), Scalar(Ephi_dot3));
      _Ephi_dot.d(_phi_dot) = dErate_dphi;
    }
  }
}
} // namespace neml2
