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

#include "neml2/user_tensors/FillR2.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/misc/assertions.h"

namespace neml2
{
register_NEML2_object(FillR2);

OptionSet
FillR2::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a R2 with a vector of Scalars. The vector length must be 1, 3, 6, or "
                  "9. When vector length is 1, the Scalar value is used to fill the diagonals; "
                  "when vector length is 3, the Scalar values are used to fill the respective "
                  "diagonal entries; when vector length is 6, the Scalar values are used to fill "
                  "the tensor following the Voigt notation; when vector length is 9, the Scalar "
                  "values are used to fill the tensor in the row-major fashion.";

  options.set<std::vector<TensorName<Scalar>>>("values");
  options.set("values").doc() = "Scalars used to fill the R2";

  return options;
}

FillR2::FillR2(const OptionSet & options)
  : UserTensorBase(options),
    R2(fill(options.get<std::vector<TensorName<Scalar>>>("values")))
{
}

R2
FillR2::fill(const std::vector<TensorName<Scalar>> & values) const
{
  auto * f = factory();
  neml_assert(f, "Internal error: factory != nullptr");

  if (values.size() == 1)
    return R2::fill(values[0].resolve(f));
  if (values.size() == 3)
    return R2::fill(values[0].resolve(f), values[1].resolve(f), values[2].resolve(f));
  if (values.size() == 6)
    return R2::fill(values[0].resolve(f),
                    values[1].resolve(f),
                    values[2].resolve(f),
                    values[3].resolve(f),
                    values[4].resolve(f),
                    values[5].resolve(f));
  if (values.size() == 9)
    return R2::fill(values[0].resolve(f),
                    values[1].resolve(f),
                    values[2].resolve(f),
                    values[3].resolve(f),
                    values[4].resolve(f),
                    values[5].resolve(f),
                    values[6].resolve(f),
                    values[7].resolve(f),
                    values[8].resolve(f));

  neml_assert(false,
              "Number of values must be 1, 3, 6, or 9, but ",
              values.size(),
              " values are provided.");

  return R2();
}
} // namespace neml2
