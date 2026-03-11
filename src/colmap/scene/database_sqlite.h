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

#include "colmap/geometry/pose_prior.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Can be used to construct temporary in-memory database.
constexpr inline char kInMemorySqliteDatabasePath[] = ":memory:";

struct SixDofPosePrior {
  data_t corr_data_id = kInvalidDataId;
  std::string image_name;
  Rigid3d cam_from_world;
  Eigen::Matrix3d rotation_covariance =
	  Eigen::Matrix3d::Constant(PosePrior::kNaN);
  Eigen::Matrix3d position_covariance =
	  Eigen::Matrix3d::Constant(PosePrior::kNaN);
  PosePrior::CoordinateSystem coordinate_system =
	  PosePrior::CoordinateSystem::UNDEFINED;
  Eigen::Vector3d gravity = Eigen::Vector3d::Constant(PosePrior::kNaN);

  inline bool HasPose() const { return cam_from_world.params.allFinite(); }
  inline bool HasRotationCov() const { return rotation_covariance.allFinite(); }
  inline bool HasPositionCov() const { return position_covariance.allFinite(); }
  inline bool HasGravity() const { return gravity.allFinite(); }
};

std::shared_ptr<Database> OpenSqliteDatabase(const std::filesystem::path& path);

std::vector<SixDofPosePrior> ReadSixDofPosePriorsFromDatabase(
	const std::filesystem::path& path,
  const std::string& table_name = "6dof_pose_priors");

}  // namespace colmap
