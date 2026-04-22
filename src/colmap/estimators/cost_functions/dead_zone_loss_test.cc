// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/cost_functions/dead_zone_loss.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(DeadZoneLoss, ZeroResidualInsideZone) {
  DeadZoneLoss loss(/*epsilon=*/1.0);
  double rho[3] = {-1.0, -1.0, -1.0};
  loss.Evaluate(0.0, rho);
  EXPECT_EQ(rho[0], 0.0);
  EXPECT_EQ(rho[1], 0.0);
  EXPECT_EQ(rho[2], 0.0);
}

TEST(DeadZoneLoss, InsideZoneNoPenalty) {
  DeadZoneLoss loss(/*epsilon=*/1.0);
  // r = 0.5 (inside zone).
  double rho[3] = {-1.0, -1.0, -1.0};
  loss.Evaluate(0.25, rho);
  EXPECT_EQ(rho[0], 0.0);
  EXPECT_EQ(rho[1], 0.0);
  EXPECT_EQ(rho[2], 0.0);
}

TEST(DeadZoneLoss, BoundaryContinuity) {
  DeadZoneLoss loss(/*epsilon=*/1.0);
  // r = 1.0 exactly on the boundary: rho and rho' are zero from both sides.
  double rho[3] = {-1.0, -1.0, -1.0};
  loss.Evaluate(1.0, rho);
  EXPECT_NEAR(rho[0], 0.0, 1e-12);
  EXPECT_NEAR(rho[1], 0.0, 1e-12);
  // rho'' is allowed to jump at the boundary (like HuberLoss).
}

TEST(DeadZoneLoss, OutsideZoneQuadratic) {
  const double epsilon = 1.0;
  DeadZoneLoss loss(epsilon);
  // r = 3, s = 9.
  const double s = 9.0;
  const double r = std::sqrt(s);
  double rho[3];
  loss.Evaluate(s, rho);
  const double expected_rho = (r - epsilon) * (r - epsilon);
  const double expected_drho = 1.0 - epsilon / r;
  const double expected_d2rho = epsilon / (2.0 * s * r);
  EXPECT_NEAR(rho[0], expected_rho, 1e-12);
  EXPECT_NEAR(rho[1], expected_drho, 1e-12);
  EXPECT_NEAR(rho[2], expected_d2rho, 1e-12);
}

TEST(DeadZoneLoss, NumericalDerivative) {
  const double epsilon = 2.0;
  DeadZoneLoss loss(epsilon);
  // Probe a few points above the threshold; rho' should match central
  // difference of rho to high accuracy.
  for (double r : {2.5, 3.0, 5.0, 10.0}) {
    const double s = r * r;
    const double h = 1e-5 * s;
    double rho_plus[3], rho_minus[3], rho_mid[3];
    loss.Evaluate(s + h, rho_plus);
    loss.Evaluate(s - h, rho_minus);
    loss.Evaluate(s, rho_mid);
    const double numerical_drho = (rho_plus[0] - rho_minus[0]) / (2.0 * h);
    EXPECT_NEAR(rho_mid[1], numerical_drho, 1e-6)
        << "rho' mismatch at r=" << r;
    const double numerical_d2rho =
        (rho_plus[1] - rho_minus[1]) / (2.0 * h);
    EXPECT_NEAR(rho_mid[2], numerical_d2rho, 1e-4)
        << "rho'' mismatch at r=" << r;
  }
}

TEST(DeadZoneLoss, ZeroEpsilonIsL2) {
  // epsilon=0 should reduce to rho(s)=s, rho'=1, rho''=0 (standard squared).
  DeadZoneLoss loss(0.0);
  for (double s : {0.25, 1.0, 4.0, 9.0}) {
    double rho[3];
    loss.Evaluate(s, rho);
    EXPECT_NEAR(rho[0], s, 1e-12);
    EXPECT_NEAR(rho[1], 1.0, 1e-12);
    EXPECT_NEAR(rho[2], 0.0, 1e-12);
  }
}

TEST(DeadZoneLoss, LargeEpsilonSuppressesMostResiduals) {
  DeadZoneLoss loss(/*epsilon=*/100.0);
  double rho[3];
  // Residuals with r=5 should be completely suppressed.
  loss.Evaluate(25.0, rho);
  EXPECT_EQ(rho[0], 0.0);
  EXPECT_EQ(rho[1], 0.0);
  EXPECT_EQ(rho[2], 0.0);
}

}  // namespace
}  // namespace colmap
