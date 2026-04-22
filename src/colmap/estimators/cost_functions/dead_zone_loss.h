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

#pragma once

#include <cmath>

#include <ceres/loss_function.h>

namespace colmap {

// Dead-zone (epsilon-insensitive squared) loss.
//
// Let r = ||residual||_2 and s = r^2. Given a threshold epsilon >= 0, the loss
// is zero inside the ball r <= epsilon and grows quadratically with r
// outside of it:
//
//   rho(s) = 0                         if r <= epsilon,
//   rho(s) = (r - epsilon)^2           if r >  epsilon.
//
// Expanded in terms of s, with r = sqrt(s):
//   rho(s)   = s - 2 * epsilon * sqrt(s) + epsilon^2
//   rho'(s)  = 1 - epsilon / sqrt(s)
//   rho''(s) = epsilon / (2 * s^{3/2})
//
// Both rho and rho' are continuous at r = epsilon (both are zero from either
// side). rho'' has a step discontinuity at the boundary, just like ceres'
// HuberLoss; ceres does not require C^2 loss kernels.
//
// Typical use case: soft priors where small deviations are expected (noise,
// sync jitter, etc.) and must not be penalized, but large deviations flag
// outliers and should be strongly pulled back. epsilon = 0 recovers standard
// squared loss. Larger epsilon means a wider "no penalty" zone.
class DeadZoneLoss : public ceres::LossFunction {
 public:
  explicit DeadZoneLoss(double epsilon)
      : epsilon_(epsilon), epsilon_sq_(epsilon * epsilon) {}

  void Evaluate(double s, double rho[3]) const final {
    if (s <= epsilon_sq_) {
      rho[0] = 0.0;
      rho[1] = 0.0;
      rho[2] = 0.0;
      return;
    }
    const double r = std::sqrt(s);
    const double d = r - epsilon_;
    rho[0] = d * d;
    rho[1] = 1.0 - epsilon_ / r;
    rho[2] = epsilon_ / (2.0 * s * r);
  }

 private:
  const double epsilon_;
  const double epsilon_sq_;
};

}  // namespace colmap
