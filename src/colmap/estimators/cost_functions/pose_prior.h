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

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

template <typename T>
inline void EigenQuaternionToAngleAxis(const T* eigen_quaternion,
                                       T* angle_axis) {
  const T quaternion[4] = {eigen_quaternion[3],
                           eigen_quaternion[0],
                           eigen_quaternion[1],
                           eigen_quaternion[2]};
  ceres::QuaternionToAngleAxis(quaternion, angle_axis);
}

// 6-DoF error on the absolute sensor pose. The residual is the log of the error
// pose, splitting SE(3) into SO(3) x R^3. The residual is computed in the
// sensor frame. Its first and last three components correspond to the rotation
// and translation errors, respectively.
struct AbsolutePosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePriorCostFunctor, 6, 7> {
 public:
  explicit AbsolutePosePriorCostFunctor(const Rigid3d& sensor_from_world_prior)
      : world_from_sensor_prior_(Inverse(sensor_from_world_prior)) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(sensor_from_world) *
        world_from_sensor_prior_.rotation().cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(sensor_from_world + 4) +
        EigenQuaternionMap<T>(sensor_from_world) *
            world_from_sensor_prior_.translation().cast<T>();

    return true;
  }

 private:
  const Rigid3d world_from_sensor_prior_;
};

// 6-DoF error on the absolute sensor pose in a rig. The residual is the log of
// the error pose, splitting SE(3) into SO(3) x R^3. The residual is computed
// in the sensor frame. Its first and last three components correspond to the
// rotation and translation errors, respectively.
struct AbsoluteRigPosePriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePriorCostFunctor, 6, 7, 7> {
 public:
  explicit AbsoluteRigPosePriorCostFunctor(
      const Rigid3d& sensor_from_world_prior)
      : world_from_sensor_prior_(Inverse(sensor_from_world_prior)) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Matrix<T, 3, 1> sensor_from_world_translation =
        EigenVector3Map<T>(sensor_from_rig + 4) +
        EigenQuaternionMap<T>(sensor_from_rig) *
            EigenVector3Map<T>(rig_from_world + 4);

    const Eigen::Quaternion<T> param_from_prior_rotation =
        sensor_from_world_rotation *
        world_from_sensor_prior_.rotation().template cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        sensor_from_world_translation +
        sensor_from_world_rotation *
            world_from_sensor_prior_.translation().template cast<T>();
    return true;
  }

 private:
  const Rigid3d world_from_sensor_prior_;
};

// 3-DoF error on the sensor position in the world coordinate frame.
struct AbsolutePosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePosePositionPriorCostFunctor, 3, 7> {
 public:
  explicit AbsolutePosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals = position_in_world_prior_.cast<T>() +
                EigenQuaternionMap<T>(sensor_from_world).inverse() *
                    EigenVector3Map<T>(sensor_from_world + 4);
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 3-DoF error on the rig sensor position in the world coordinate frame.
struct AbsoluteRigPosePositionPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPosePositionPriorCostFunctor,
                                 3,
                                 7,
                                 7> {
 public:
  explicit AbsoluteRigPosePositionPriorCostFunctor(
      const Eigen::Vector3d& position_in_world_prior)
      : position_in_world_prior_(position_in_world_prior) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    const Eigen::Matrix<T, 3, 1> sensor_from_world_translation =
        EigenVector3Map<T>(sensor_from_rig + 4) +
        EigenQuaternionMap<T>(sensor_from_rig) *
            EigenVector3Map<T>(rig_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    residuals =
        position_in_world_prior_.cast<T>() +
        sensor_from_world_rotation.inverse() * sensor_from_world_translation;
    return true;
  }

 private:
  const Eigen::Vector3d position_in_world_prior_;
};

// 6-DoF error between two absolute camera poses based on a prior on their
// relative pose, with identical scale for the translation. The residual is
// computed in the frame of camera i. Its first and last three components
// correspond to the rotation and translation errors, respectively.
//
// Derivation:
//    i_T_w = ΔT_i·i_T_j·j_T_w
//    where ΔT_i = exp(η_i) is the resjdual in SE(3) and η_i in tangent space.
//    Thus η_i = log(i_T_w·j_T_w⁻¹·j_T_i)
//    Rotation term: ΔR = log(i_R_w·j_R_w⁻¹·j_R_i)
//    Translation term: Δt = i_t_w + i_R_w·j_R_w⁻¹·(j_t_i -j_t_w)
struct RelativePosePriorCostFunctor
    : public AutoDiffCostFunctor<RelativePosePriorCostFunctor, 6, 7, 7> {
 public:
  explicit RelativePosePriorCostFunctor(const Rigid3d& i_from_j_prior)
      : j_from_i_prior_(Inverse(i_from_j_prior)) {}

  template <typename T>
  bool operator()(const T* const i_from_world,
                  const T* const j_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> i_from_j_rotation =
        EigenQuaternionMap<T>(i_from_world) *
        EigenQuaternionMap<T>(j_from_world).inverse();
    const Eigen::Quaternion<T> param_from_prior_rotation =
        i_from_j_rotation * j_from_i_prior_.rotation().template cast<T>();
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);

    const Eigen::Matrix<T, 3, 1> j_from_i_prior_translation =
        j_from_i_prior_.translation().cast<T>() -
        EigenVector3Map<T>(j_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_prior_translation(
        residuals_ptr + 3);
    param_from_prior_translation =
        EigenVector3Map<T>(i_from_world + 4) +
        i_from_j_rotation * j_from_i_prior_translation;

    return true;
  }

 private:
  const Rigid3d j_from_i_prior_;
};

// Constant-velocity / zero-acceleration prior on three temporally adjacent
// camera poses from the same sensor. The residual compares the forward
// inter-frame twist with the (dt-scaled) backward inter-frame twist; it is
// exactly zero for any uniform-screw trajectory T_k = exp(k * dt_k * xi) * T_0.
//
// Parameter blocks are Rigid3d-layout (qx, qy, qz, qw, tx, ty, tz); all three
// are cam_from_world (or rig_from_world) poses expressed in the same world
// frame. The 6D residual is split into a 3D rotation part (angle-axis) and a
// 3D translation part, each normalized by its sigma. Decoupled SO(3) x R^3 is
// used as a first-order approximation to SE(3)-log; it is exact for the
// constant-velocity case we are penalizing and accurate to O(|omega| * |t|)
// for non-constant jerk — well within SLAM inter-frame rotation regimes.
struct ConstantVelocityPriorCostFunctor
    : public AutoDiffCostFunctor<ConstantVelocityPriorCostFunctor,
                                 6,
                                 7,
                                 7,
                                 7> {
 public:
  ConstantVelocityPriorCostFunctor(double dt_prev,
                                   double dt_next,
                                   double sigma_rot_rad,
                                   double sigma_trans_m)
      : dt_ratio_(dt_next / dt_prev),
        inv_sigma_rot_(1.0 / sigma_rot_rad),
        inv_sigma_trans_(1.0 / sigma_trans_m) {}

  template <typename T>
  bool operator()(const T* const cam_from_world_prev,
                  const T* const cam_from_world_curr,
                  const T* const cam_from_world_next,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> q_prev = EigenQuaternionMap<T>(cam_from_world_prev);
    const Eigen::Quaternion<T> q_curr = EigenQuaternionMap<T>(cam_from_world_curr);
    const Eigen::Quaternion<T> q_next = EigenQuaternionMap<T>(cam_from_world_next);
    const Eigen::Matrix<T, 3, 1> t_prev = EigenVector3Map<T>(cam_from_world_prev + 4);
    const Eigen::Matrix<T, 3, 1> t_curr = EigenVector3Map<T>(cam_from_world_curr + 4);
    const Eigen::Matrix<T, 3, 1> t_next = EigenVector3Map<T>(cam_from_world_next + 4);

    // delta_back = T_curr * Inverse(T_prev):
    //   q = q_curr * q_prev.conjugate()
    //   t = t_curr - q * t_prev
    const Eigen::Quaternion<T> q_back = q_curr * q_prev.conjugate();
    const Eigen::Matrix<T, 3, 1> t_back = t_curr - (q_back * t_prev);

    const Eigen::Quaternion<T> q_fwd = q_next * q_curr.conjugate();
    const Eigen::Matrix<T, 3, 1> t_fwd = t_next - (q_fwd * t_curr);

    T aa_back[3];
    EigenQuaternionToAngleAxis(q_back.coeffs().data(), aa_back);
    T aa_fwd[3];
    EigenQuaternionToAngleAxis(q_fwd.coeffs().data(), aa_fwd);

    const T dt_ratio = T(dt_ratio_);
    const T inv_sigma_rot = T(inv_sigma_rot_);
    const T inv_sigma_trans = T(inv_sigma_trans_);

    residuals_ptr[0] = inv_sigma_rot * (aa_fwd[0] - dt_ratio * aa_back[0]);
    residuals_ptr[1] = inv_sigma_rot * (aa_fwd[1] - dt_ratio * aa_back[1]);
    residuals_ptr[2] = inv_sigma_rot * (aa_fwd[2] - dt_ratio * aa_back[2]);
    residuals_ptr[3] = inv_sigma_trans * (t_fwd.x() - dt_ratio * t_back.x());
    residuals_ptr[4] = inv_sigma_trans * (t_fwd.y() - dt_ratio * t_back.y());
    residuals_ptr[5] = inv_sigma_trans * (t_fwd.z() - dt_ratio * t_back.z());
    return true;
  }

 private:
  const double dt_ratio_;
  const double inv_sigma_rot_;
  const double inv_sigma_trans_;
};

}  // namespace colmap
