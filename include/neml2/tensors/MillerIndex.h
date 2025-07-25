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

#pragma once

#include "neml2/tensors/PrimitiveTensor.h"

namespace neml2
{
class Vec;
class Scalar;

/// Represention of a crystal direction or plane a Miller Index
// I delibrately made this not inherit from VecBase but we could
// reconsider that later on
class MillerIndex : public PrimitiveTensor<MillerIndex, 3>
{
public:
  using PrimitiveTensor<MillerIndex, 3>::PrimitiveTensor;

  /// Accessor
  Scalar operator()(Size i) const;

  /// Most likely construction method -- make from three ints
  static MillerIndex fill(int64_t a,
                          int64_t b,
                          int64_t c,
                          const TensorOptions & options = default_integer_tensor_options());

  /// Reduce to the greatest common demoninator
  MillerIndex reduce() const;

  /// Convert back to a (real) Vec
  Vec to_vec(const TensorOptions & options = default_tensor_options()) const;

  /// Convert back to a normalized real Vec
  Vec to_normalized_vec(const TensorOptions & options = default_tensor_options()) const;
};
} // namespace neml2
