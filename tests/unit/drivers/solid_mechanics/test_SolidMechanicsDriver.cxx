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

#include <catch2/catch_test_macros.hpp>

#include "utils.h"
#include "neml2/drivers/solid_mechanics/SolidMechanicsDriver.h"

using namespace neml2;

TEST_CASE("SolidMechanicsDriver", "[SolidMechanicsDriver]")
{
  SECTION("strain control")
  {
    auto factory = load_input("drivers/solid_mechanics/test_SolidMechanicsDriver_strain.i");
    auto driver = factory->get_driver("driver");
    diagnose(*driver);
    REQUIRE(driver->run());
  }

  SECTION("stress control")
  {
    auto factory = load_input("drivers/solid_mechanics/test_SolidMechanicsDriver_stress.i");
    auto driver = factory->get_driver("driver");
    diagnose(*driver);
    REQUIRE(driver->run());
  }

  SECTION("mixed control")
  {
    auto factory = load_input("drivers/solid_mechanics/test_SolidMechanicsDriver_mixed.i");
    auto driver = factory->get_driver("driver");
    diagnose(*driver);
    REQUIRE(driver->run());
  }

  SECTION("temperature dependent")
  {
    auto factory = load_input("drivers/solid_mechanics/test_SolidMechanicsDriver_temperature.i");
    auto driver = factory->get_driver("driver");
    diagnose(*driver);
    REQUIRE(driver->run());
  }

  SECTION("large deformation incremental")
  {
    auto factory = load_input(
        "drivers/solid_mechanics/test_LargeDeformationIncrementalSolidMechanicsDriver.i");
    auto driver = factory->get_driver("driver");
    diagnose(*driver);
    REQUIRE(driver->run());
  }
}
