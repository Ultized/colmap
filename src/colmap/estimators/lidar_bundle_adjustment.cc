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

#include "colmap/estimators/lidar_bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/lidar.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ceres/ceres.h>

namespace colmap {

// ============================================================================
// LidarMatcher implementation
// ============================================================================

LidarMatcher::LidarMatcher(const LidarPointCloud& cloud,
                             const LidarMatchingOptions& options)
    : cloud_(cloud), options_(options) {}

std::vector<LidarConstraint> LidarMatcher::BuildConstraints(
    const Reconstruction& reconstruction, int phase) const {
  THROW_CHECK_GE(phase, 1);
  THROW_CHECK_LE(phase, 2);

  const double max_dist =
      (phase == 1) ? options_.phase1_max_distance : options_.phase2_max_distance;
  const double max_sq_dist = max_dist * max_dist;

  if (cloud_.Empty()) return {};

  // -------------------------------------------------------------------------
  // Phase 1: raw KNN search over all Point3Ds.
  // Phase 2: same search but with tighter threshold.
  // -------------------------------------------------------------------------
  struct RawMatch {
    point3D_t point3D_id;
    size_t lidar_idx;
    double sq_dist;
  };
  std::vector<RawMatch> raw_matches;
  raw_matches.reserve(reconstruction.NumPoints3D());

  // -----------------------------------------------------------------------
  // Optimization B: parallel KNN matching via OpenMP.
  //
  // reconstruction.Points3D() is an unordered_map and cannot be indexed
  // directly. We snapshot the (id, xyz, error, track_length) tuples into a
  // flat vector so that the OpenMP loop can use integer indices.
  // KDTree3d::KNearestNeighbors is read-only and therefore thread-safe.
  // Each thread accumulates into its own local vector; results are merged
  // after the parallel region to avoid any locking on the hot path.
  // -----------------------------------------------------------------------
  struct PointEntry {
    point3D_t id;
    Eigen::Vector3d xyz;
    double error;
    int track_length;
  };
  std::vector<PointEntry> point_entries;
  point_entries.reserve(reconstruction.NumPoints3D());
  for (const auto& [pid, pt] : reconstruction.Points3D()) {
    point_entries.push_back(
        {pid, pt.xyz, pt.error, static_cast<int>(pt.track.Length())});
  }

  const int n_points = static_cast<int>(point_entries.size());
  std::atomic<int> skipped_track{0};
  std::atomic<int> skipped_repr{0};
  std::atomic<int> skipped_dist{0};

  // Thread-local storage for partial results.
  const int n_threads =
#ifdef _OPENMP
      std::max(1, omp_get_max_threads());
#else
      1;
#endif
  std::vector<std::vector<RawMatch>> thread_matches(
      static_cast<size_t>(n_threads));
  for (auto& v : thread_matches) v.reserve(256);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int i = 0; i < n_points; ++i) {
    const PointEntry& pe = point_entries[static_cast<size_t>(i)];

    if (options_.min_track_length > 0 &&
        pe.track_length < options_.min_track_length) {
      ++skipped_track;
      continue;
    }
    if (options_.max_reprojection_error > 0.0 &&
        pe.error > options_.max_reprojection_error) {
      ++skipped_repr;
      continue;
    }

    const auto candidates =
        cloud_.KNearestNeighbors(pe.xyz, options_.k_candidates);

    double best_sq = std::numeric_limits<double>::max();
    size_t best_idx = 0;
    for (const auto& c : candidates) {
      if (c.sq_dist < best_sq && c.sq_dist <= max_sq_dist) {
        best_sq = c.sq_dist;
        best_idx = c.index;
      }
    }
    if (best_sq > max_sq_dist) { ++skipped_dist; continue; }

    const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
    thread_matches[static_cast<size_t>(tid)].push_back(
        {pe.id, best_idx, best_sq});
  }

  // Merge thread-local results.
  for (auto& v : thread_matches) {
    raw_matches.insert(raw_matches.end(), v.begin(), v.end());
  }

  LOG(INFO) << "[LidarMatcher] Phase " << phase
            << " quality pre-filter: skipped_track_length=" << skipped_track.load()
            << " skipped_reprojection_error=" << skipped_repr.load()
            << " skipped_no_lidar_coverage=" << skipped_dist.load()
            << " candidates_passed=" << raw_matches.size();

  if (raw_matches.empty()) return {};

  // -------------------------------------------------------------------------
  // Statistical outlier rejection: remove matches with distance
  //   > mean + stat_sigma * std_dev
  // -------------------------------------------------------------------------
  if (options_.stat_sigma > 0.0 && raw_matches.size() > 3) {
    std::vector<double> dists;
    dists.reserve(raw_matches.size());
    for (const auto& m : raw_matches) dists.push_back(std::sqrt(m.sq_dist));

    const double mean =
        std::accumulate(dists.begin(), dists.end(), 0.0) /
        static_cast<double>(dists.size());

    double variance = 0.0;
    for (double d : dists) variance += (d - mean) * (d - mean);
    variance /= static_cast<double>(dists.size());
    const double sigma = std::sqrt(variance);

    const double threshold = mean + options_.stat_sigma * sigma;
    const double threshold_sq = threshold * threshold;

    std::vector<RawMatch> filtered;
    filtered.reserve(raw_matches.size());
    for (const auto& m : raw_matches) {
      if (m.sq_dist <= threshold_sq) filtered.push_back(m);
    }
    raw_matches = std::move(filtered);
  }

  // -------------------------------------------------------------------------
  // Build final LidarConstraint objects.
  // Phase 2: apply optional normal alignment filter.
  // -------------------------------------------------------------------------
  std::vector<LidarConstraint> constraints;
  constraints.reserve(raw_matches.size());

  for (const auto& m : raw_matches) {
    const LidarPoint& lp = cloud_.Point(m.lidar_idx);

    // Phase 2 normal alignment check.
    if (phase == 2 && options_.phase2_max_normal_alignment > 0.0 &&
        lp.HasNormal() &&
        !NormalAlignmentOk(lp.normal, reconstruction, m.point3D_id)) {
      continue;
    }

    LidarConstraint c;
    c.point3D_id = m.point3D_id;
    c.xyz_lidar  = lp.xyz;
    c.normal     = lp.normal;
    c.sq_dist    = m.sq_dist;
    c.use_plane  = lp.HasNormal();

    constraints.push_back(c);
  }

  LOG(INFO) << "[LidarMatcher] Phase " << phase
            << ": built " << constraints.size()
            << " constraints (raw=" << raw_matches.size() << ").";

  return constraints;
}

bool LidarMatcher::NormalAlignmentOk(const Eigen::Vector3d& lidar_normal,
                                      const Reconstruction& reconstruction,
                                      point3D_t point3d_id) const {
  // Compute approximate mean viewing direction for this 3D point by looking
  // at all images that observe it and averaging their optical axis directions.
  if (!reconstruction.ExistsPoint3D(point3d_id)) return false;
  const Point3D& pt = reconstruction.Point3D(point3d_id);
  if (pt.track.Length() == 0) return false;

  Eigen::Vector3d mean_dir = Eigen::Vector3d::Zero();
  int count = 0;
  for (const auto& te : pt.track.Elements()) {
    if (!reconstruction.ExistsImage(te.image_id)) continue;
    const Image& img = reconstruction.Image(te.image_id);
    if (!img.HasPose()) continue;
    // Camera centre → point direction.
    // ProjectionCenter() = -R^T * t = camera centre in world frame.
    const Eigen::Vector3d img_center = img.ProjectionCenter();
    const Eigen::Vector3d ray = (pt.xyz - img_center).normalized();
    mean_dir += ray;
    ++count;
  }
  if (count == 0) return true;  // cannot decide, allow
  mean_dir.normalize();

  // |dot(normal, ray)| close to 1 → grazing: avoid.  Close to 0 → frontal.
  // Reject when the normal is too aligned with the viewing ray.
  const double alignment = std::abs(lidar_normal.dot(mean_dir));
  return alignment <= options_.phase2_max_normal_alignment;
}

// ============================================================================
// LidarBundleAdjusterImpl
//
// Wraps a DefaultCeresBundleAdjuster and injects LiDAR residuals into its
// already-constructed ceres::Problem before solving.
// ============================================================================
namespace {

class LidarBundleAdjusterImpl : public BundleAdjuster {
 public:
  LidarBundleAdjusterImpl(
      const BundleAdjustmentOptions& options,
      const LidarBundleAdjustmentOptions& lidar_options,
      const BundleAdjustmentConfig& config,
      const std::vector<LidarConstraint>& constraints,
      Reconstruction& reconstruction)
      : BundleAdjuster(options, config) {
    // 1. Build the standard BA adjuster (problem is constructed in ctor).
    inner_ = CreateDefaultCeresBundleAdjuster(options, config, reconstruction);

    auto* ceres_ba =
        dynamic_cast<CeresBundleAdjuster*>(inner_.get());
    THROW_CHECK(ceres_ba != nullptr)
        << "CreateDefaultCeresBundleAdjuster must return a CeresBundleAdjuster";

    std::shared_ptr<ceres::Problem>& problem = ceres_ba->Problem();

    // 2. Create loss function for LiDAR residuals.
    ceres::LossFunction* raw_loss = nullptr;
    switch (lidar_options.loss_function_type) {
      case CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL:
        raw_loss = nullptr;
        break;
      case CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1:
        raw_loss = new ceres::SoftLOneLoss(lidar_options.loss_scale);
        break;
      case CeresBundleAdjustmentOptions::LossFunctionType::CAUCHY:
        raw_loss = new ceres::CauchyLoss(lidar_options.loss_scale);
        break;
      case CeresBundleAdjustmentOptions::LossFunctionType::HUBER:
        raw_loss = new ceres::HuberLoss(lidar_options.loss_scale);
        break;
    }
    // Wrap with ScaledLoss to apply the weight schedule.
    ceres::LossFunction* scaled_loss =
        new ceres::ScaledLoss(raw_loss,
                              lidar_options.weight,
                              ceres::DO_NOT_TAKE_OWNERSHIP);
    // Store in a shared_ptr so ownership is clear.
    loss_function_.reset(scaled_loss);
    if (raw_loss) raw_loss_.reset(raw_loss);

    // 3. Add LiDAR residuals for every constraint whose point3D is a variable
    //    parameter block in the problem.
    int added = 0;
    for (const auto& c : constraints) {
      if (!reconstruction.ExistsPoint3D(c.point3D_id)) continue;
      Point3D& point3D = reconstruction.Point3D(c.point3D_id);
      double* xyz = point3D.xyz.data();

      // Only add a residual if the point is already a (variable) parameter
      // block in the problem, to avoid introducing unconstrained blocks.
      if (!problem->HasParameterBlock(xyz)) continue;
      if (problem->IsParameterBlockConstant(xyz)) continue;

      ceres::CostFunction* cost = nullptr;
      if (c.use_plane && lidar_options.use_point_to_plane) {
        cost = PointToPlaneCostFunctor::Create(c.xyz_lidar, c.normal);
      } else {
        cost = PointToPointCostFunctor::Create(c.xyz_lidar);
      }

      problem->AddResidualBlock(cost, scaled_loss, xyz);
      ++added;
    }

    LOG(INFO) << "[LidarBA] Added " << added
              << " LiDAR residuals (weight=" << lidar_options.weight
              << ", constraints=" << constraints.size() << ").";
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    return inner_->Solve();
  }

 private:
  std::unique_ptr<BundleAdjuster> inner_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> raw_loss_;
};

}  // namespace

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<BundleAdjuster> CreateLidarBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const LidarBundleAdjustmentOptions& lidar_options,
    const BundleAdjustmentConfig& config,
    const std::vector<LidarConstraint>& constraints,
    Reconstruction& reconstruction) {
  return std::make_unique<LidarBundleAdjusterImpl>(
      options, lidar_options, config, constraints, reconstruction);
}

}  // namespace colmap
