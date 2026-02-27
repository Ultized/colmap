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

#include "colmap/scene/lidar_point_cloud.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/ply.h"

#include <fstream>
#include <sstream>
#include <filesystem>

namespace colmap {

std::shared_ptr<LidarPointCloud> LoadLidarPointCloud(
    const std::filesystem::path& path) {
  THROW_CHECK(std::filesystem::exists(path))
      << "LiDAR point cloud file not found: " << path;

  std::vector<LidarPoint> points;
  const std::string ext_input = path.extension().string();
  // StringToLower operates in place, so copy first.
  std::string ext = ext_input;
  StringToLower(&ext);

  if (ext == ".ply") {
    // ---------------------------------------------------------------------------
    // PLY format: use COLMAP's built-in reader which handles both ASCII and
    // binary PLY, and reads x/y/z position plus optional nx/ny/nz normals.
    // ---------------------------------------------------------------------------
    const auto ply_points = ReadPly(path);
    points.reserve(ply_points.size());
    for (const auto& p : ply_points) {
      LidarPoint lp;
      lp.xyz    = Eigen::Vector3d(p.x, p.y, p.z);
      lp.normal = Eigen::Vector3d(p.nx, p.ny, p.nz);
      points.push_back(lp);
    }
  } else {
    // ---------------------------------------------------------------------------
    // ASCII text format (.txt / .xyz / .csv / …)
    //   Supported line layouts (whitespace-separated):
    //     x y z
    //     x y z nx ny nz
    //     x y z r g b          (colour ignored)
    //     x y z nx ny nz r g b (colour ignored)
    //   Lines starting with '#' or empty are skipped.
    // ---------------------------------------------------------------------------
    std::ifstream file(path);
    THROW_CHECK(file.is_open()) << "Cannot open LiDAR file: " << path;

    std::string line;
    while (std::getline(file, line)) {
      // Trim leading whitespace.
      const size_t first = line.find_first_not_of(" \t\r\n");
      if (first == std::string::npos || line[first] == '#') continue;

      std::istringstream ss(line.substr(first));
      double x, y, z;
      if (!(ss >> x >> y >> z)) continue;  // malformed line

      LidarPoint lp;
      lp.xyz = Eigen::Vector3d(x, y, z);

      double nx, ny, nz;
      if (ss >> nx >> ny >> nz) {
        lp.normal = Eigen::Vector3d(nx, ny, nz);
      }

      points.push_back(lp);
    }
  }

  LOG(INFO) << "LoadLidarPointCloud: loaded " << points.size()
            << " points from " << path;
  THROW_CHECK(!points.empty()) << "LiDAR file is empty: " << path;

  return std::make_shared<LidarPointCloud>(std::move(points));
}

}  // namespace colmap
