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

#include "colmap/estimators/cost_functions/pose_prior.h"

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/util/eigen_matchers.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(AbsolutePosePositionPriorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePosePositionPriorCostFunctor::Create(Eigen::Vector3d::Zero()));

  Rigid3d sensor_from_world =
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());

  Eigen::Vector3d residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const double* parameters[1] = {sensor_from_world.params.data()};

  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));

  sensor_from_world =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  const Eigen::Vector3d position_in_world =
      Inverse(sensor_from_world).translation();
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(-position_in_world), 1e-6));

  cost_function.reset(
      AbsolutePosePositionPriorCostFunctor::Create(position_in_world));
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));
}

TEST(AbsoluteRigPosePositionPriorCostFunctor, Nominal) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsoluteRigPosePositionPriorCostFunctor::Create(Eigen::Vector3d::Zero()));

  Rigid3d sensor_from_rig(Eigen::Quaterniond::Identity(),
                          Eigen::Vector3d::Zero());
  Rigid3d rig_from_world(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d::Zero());

  Eigen::Vector3d residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  const double* parameters[2] = {sensor_from_rig.params.data(),
                                 rig_from_world.params.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));

  sensor_from_rig =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  rig_from_world =
      Rigid3d(Eigen::Quaterniond::UnitRandom(), Eigen::Vector3d::Random());
  const Rigid3d sensor_from_world = sensor_from_rig * rig_from_world;
  const Eigen::Vector3d position_in_world =
      Inverse(sensor_from_world).translation();
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals,
              EigenMatrixNear(Eigen::Vector3d(-position_in_world), 1e-6));

  cost_function.reset(
      AbsoluteRigPosePositionPriorCostFunctor::Create(position_in_world));
  residuals =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-6));
}

TEST(AbsolutePosePriorCostFunctor, Nominal) {
  const Rigid3d cam_from_world_prior;
  std::unique_ptr<ceres::CostFunction> cost_function(
      AbsolutePosePriorCostFunctor::Create(cam_from_world_prior));

  double cam_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double residuals[6];
  const double* parameters[1] = {cam_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  cam_from_world[4] = 1;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 1);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(static_cast<double*>(cam_from_world)) =
      rotation_matrix;
  cam_from_world[5] = 2;
  cam_from_world[6] = 3;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], 1, 1e-6);
  EXPECT_NEAR(residuals[4], 2, 1e-6);
  EXPECT_NEAR(residuals[5], 3, 1e-6);
}

TEST(RelativePosePriorCostFunctor, Nominal) {
  Rigid3d i_from_j_prior(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(0, 0, -1));
  std::unique_ptr<ceres::CostFunction> cost_function(
      RelativePosePriorCostFunctor::Create(i_from_j_prior));

  double i_from_world[7] = {0, 0, 0, 1, 0, 0, 0};
  double j_from_world[7] = {0, 0, 0, 1, 0, 0, 1};
  double residuals[6];
  const double* parameters[2] = {i_from_world, j_from_world};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 0);

  i_from_world[6] = 4;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], 0);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 4);

  j_from_world[4] = 2;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_EQ(residuals[0], 0);
  EXPECT_EQ(residuals[1], 0);
  EXPECT_EQ(residuals[2], 0);
  EXPECT_EQ(residuals[3], -2);
  EXPECT_EQ(residuals[4], 0);
  EXPECT_EQ(residuals[5], 4);

  // Rotation by 90 degrees around the Y axis.
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  Eigen::Map<Eigen::Quaterniond>(static_cast<double*>(j_from_world)) =
      rotation_matrix;
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0], 0, 1e-6);
  EXPECT_NEAR(residuals[1], DegToRad(-90.0), 1e-6);
  EXPECT_NEAR(residuals[2], 0, 1e-6);
  EXPECT_NEAR(residuals[3], 0, 1e-6);
  EXPECT_NEAR(residuals[4], 0, 1e-6);
  EXPECT_NEAR(residuals[5], 2, 1e-6);
}

TEST(CovarianceWeightedCostFunctor, AbsolutePosePositionPriorCostFunctor) {
  const Rigid3d cam_from_world(Eigen::Quaterniond::UnitRandom(),
                               Eigen::Vector3d::Random());
  const Rigid3d world_from_cam = Inverse(cam_from_world);

  double residuals[3];
  const double* parameters[1] = {cam_from_world.params.data()};

  std::unique_ptr<ceres::CostFunction> cost_function(
      CovarianceWeightedCostFunctor<AbsolutePosePositionPriorCostFunctor>::
          Create(2 * Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  EXPECT_NEAR(residuals[0],
              -0.5 * std::sqrt(2) * world_from_cam.translation()[0],
              1e-6);
  EXPECT_NEAR(residuals[1],
              -0.5 * std::sqrt(2) * world_from_cam.translation()[1],
              1e-6);
  EXPECT_NEAR(residuals[2],
              -0.5 * std::sqrt(2) * world_from_cam.translation()[2],
              1e-6);
}

TEST(ConstantVelocityPriorCostFunctor, ZeroOnUniformDtScrewMotion) {
  // Iterative chain: T_{k+1} = delta * T_k with fixed world-frame delta.
  // This makes T_curr * Inverse(T_prev) constant for all k, so the
  // twist-difference residual is exactly zero.
  const double dt = 0.1;
  const double sigma_rot = 0.01;
  const double sigma_trans = 0.01;

  const Eigen::Quaterniond q_step(Eigen::AngleAxisd(
      0.05, Eigen::Vector3d(0.4, -0.8, 0.3).normalized()));
  const Eigen::Vector3d t_step(0.03, -0.01, 0.005);
  const Rigid3d delta(q_step, t_step);

  const Rigid3d T_prev(
      Eigen::Quaterniond(Eigen::AngleAxisd(
          0.3, Eigen::Vector3d(0.7, -0.2, 0.1).normalized())),
      Eigen::Vector3d(1.0, 2.0, -3.0));
  const Rigid3d T_curr = delta * T_prev;
  const Rigid3d T_next = delta * T_curr;

  std::unique_ptr<ceres::CostFunction> cost_function(
      ConstantVelocityPriorCostFunctor::Create(dt, dt, sigma_rot, sigma_trans));

  const double* parameters[3] = {T_prev.params.data(),
                                 T_curr.params.data(),
                                 T_next.params.data()};
  double residuals[6];
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, 1e-10) << "component " << i;
  }
}

TEST(ConstantVelocityPriorCostFunctor, ZeroOnNonUniformDtPureTranslation) {
  // Non-uniform dt but pure translation (omega = 0) -> V = I, decoupled
  // math is exact. delta_prev and delta_next scale linearly with dt.
  const double dt_prev = 0.07;
  const double dt_next = 0.21;
  const double sigma_rot = 0.01;
  const double sigma_trans = 0.01;

  const Eigen::Vector3d vel(-0.5, 0.2, 0.0);
  const Rigid3d delta_prev(Eigen::Quaterniond::Identity(), vel * dt_prev);
  const Rigid3d delta_next(Eigen::Quaterniond::Identity(), vel * dt_next);

  const Rigid3d T_prev(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  const Rigid3d T_curr = delta_prev * T_prev;
  const Rigid3d T_next = delta_next * T_curr;

  std::unique_ptr<ceres::CostFunction> cost_function(
      ConstantVelocityPriorCostFunctor::Create(
          dt_prev, dt_next, sigma_rot, sigma_trans));

  const double* parameters[3] = {T_prev.params.data(),
                                 T_curr.params.data(),
                                 T_next.params.data()};
  double residuals[6];
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, 1e-10) << "component " << i;
  }
}

TEST(ConstantVelocityPriorCostFunctor, ZeroOnNonUniformDtPureRotation) {
  // Non-uniform dt but pure rotation (translation = 0) -> decoupled math
  // is exact. angle_axis scales linearly with dt.
  const double dt_prev = 0.07;
  const double dt_next = 0.21;
  const double sigma_rot = 0.01;
  const double sigma_trans = 0.01;

  const Eigen::Vector3d omega(0.05, -0.08, 0.12);
  const Eigen::Quaterniond q_prev(
      Eigen::AngleAxisd(omega.norm() * dt_prev, omega.normalized()));
  const Eigen::Quaterniond q_next(
      Eigen::AngleAxisd(omega.norm() * dt_next, omega.normalized()));
  const Rigid3d delta_prev(q_prev, Eigen::Vector3d::Zero());
  const Rigid3d delta_next(q_next, Eigen::Vector3d::Zero());

  const Rigid3d T_prev(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  const Rigid3d T_curr = delta_prev * T_prev;
  const Rigid3d T_next = delta_next * T_curr;

  std::unique_ptr<ceres::CostFunction> cost_function(
      ConstantVelocityPriorCostFunctor::Create(
          dt_prev, dt_next, sigma_rot, sigma_trans));

  const double* parameters[3] = {T_prev.params.data(),
                                 T_curr.params.data(),
                                 T_next.params.data()};
  double residuals[6];
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(residuals[i], 0.0, 1e-10) << "component " << i;
  }
}

TEST(ConstantVelocityPriorCostFunctor, NonzeroOnJerk) {
  // Construct three poses where T_curr deviates from the uniform-velocity
  // midpoint by a known translation. Residual must reflect that deviation,
  // scaled by 1 / sigma.
  const double dt = 0.1;
  const double sigma_rot = 0.01;
  const double sigma_trans = 0.02;

  const Rigid3d T_prev(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
  // Uniform-velocity midpoint would be T_curr = (I, (1,0,0)).
  // Perturb translation so that forward increment differs from backward by
  // (+0.04, 0, 0). With dt_ratio = 1, residual_trans_x = -0.04 / sigma_trans * 2.
  Rigid3d T_curr(Eigen::Quaterniond::Identity(),
                 Eigen::Vector3d(1.0 - 0.02, 0.0, 0.0));
  Rigid3d T_next(Eigen::Quaterniond::Identity(),
                 Eigen::Vector3d(2.0 + 0.02, 0.0, 0.0));

  std::unique_ptr<ceres::CostFunction> cost_function(
      ConstantVelocityPriorCostFunctor::Create(dt, dt, sigma_rot, sigma_trans));

  const double* parameters[3] = {T_prev.params.data(),
                                 T_curr.params.data(),
                                 T_next.params.data()};
  double residuals[6];
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, nullptr));
  // delta_back.t = (0.98, 0, 0); delta_fwd.t = (1.04, 0, 0)
  // residual_trans_x = (1.04 - 0.98) / sigma_trans = 0.06 / 0.02 = 3.0
  EXPECT_NEAR(residuals[0], 0.0, 1e-8);
  EXPECT_NEAR(residuals[1], 0.0, 1e-8);
  EXPECT_NEAR(residuals[2], 0.0, 1e-8);
  EXPECT_NEAR(residuals[3], 3.0, 1e-6);
  EXPECT_NEAR(residuals[4], 0.0, 1e-8);
  EXPECT_NEAR(residuals[5], 0.0, 1e-8);
}

}  // namespace
}  // namespace colmap
