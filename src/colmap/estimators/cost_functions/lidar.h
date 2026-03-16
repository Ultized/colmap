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

// LiDAR-specific Ceres cost functors.
//
// Two variants:
//   PointToPlaneCostFunctor  – 1-residual:  dot(xyz - p_lidar, normal)
//   PointToPointCostFunctor  – 3-residuals: xyz - p_lidar
//
// Both operate on the 3-D position of an SfM Point3D (parameter block: xyz[3]).

#include "colmap/estimators/cost_functions/utils.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

// ---------------------------------------------------------------------------
// PointToPlaneCostFunctor
//
// Residual = dot(xyz - p_lidar, n_lidar)
//
// Constrains the SfM 3D point to lie on the tangent plane at the LiDAR
// surface element.  Insensitive to errors along the viewing ray, which is
// the dominant uncertainty direction of SfM triangulation.  Requires a valid
// (unit) surface normal from the LiDAR point cloud.
// ---------------------------------------------------------------------------
struct PointToPlaneCostFunctor
    : public AutoDiffCostFunctor<PointToPlaneCostFunctor, 1, 3> {
 public:
  PointToPlaneCostFunctor(const Eigen::Vector3d& lidar_xyz,
                          const Eigen::Vector3d& lidar_normal)
      : lidar_xyz_(lidar_xyz), lidar_normal_(lidar_normal) {}

  template <typename T>
  bool operator()(const T* const xyz, T* residuals) const {
    residuals[0] =
        lidar_normal_.cast<T>().dot(
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(xyz) -
            lidar_xyz_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& lidar_xyz,
                                     const Eigen::Vector3d& lidar_normal) {
    return new ceres::AutoDiffCostFunction<PointToPlaneCostFunctor, 1, 3>(
        new PointToPlaneCostFunctor(lidar_xyz, lidar_normal));
  }

 private:
  const Eigen::Vector3d lidar_xyz_;
  const Eigen::Vector3d lidar_normal_;
};

// ---------------------------------------------------------------------------
// PointToPointCostFunctor
//
// Residuals = xyz - p_lidar  (3-vector)
//
// Fallback when no surface normal is available.
// ---------------------------------------------------------------------------
struct PointToPointCostFunctor
    : public AutoDiffCostFunctor<PointToPointCostFunctor, 3, 3> {
 public:
  explicit PointToPointCostFunctor(const Eigen::Vector3d& lidar_xyz)
      : lidar_xyz_(lidar_xyz) {}

  template <typename T>
  bool operator()(const T* const xyz, T* residuals) const {
    residuals[0] = xyz[0] - T(lidar_xyz_[0]);
    residuals[1] = xyz[1] - T(lidar_xyz_[1]);
    residuals[2] = xyz[2] - T(lidar_xyz_[2]);
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& lidar_xyz) {
    return new ceres::AutoDiffCostFunction<PointToPointCostFunctor, 3, 3>(
        new PointToPointCostFunctor(lidar_xyz));
  }

 private:
  const Eigen::Vector3d lidar_xyz_;
};

}  // namespace colmap
