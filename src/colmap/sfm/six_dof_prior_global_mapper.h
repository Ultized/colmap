// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include "colmap/estimators/lidar_bundle_adjustment.h"
#include "colmap/scene/lidar_point_cloud.h"
#include "colmap/scene/database_sqlite.h"
#include "colmap/sfm/prior_global_mapper.h"

#include <unordered_map>

namespace colmap {

class SixDofPriorGlobalMapper : public PriorGlobalMapper {
 public:
  explicit SixDofPriorGlobalMapper(
      std::shared_ptr<const DatabaseCache> database_cache,
      std::vector<SixDofPosePrior> six_dof_pose_priors,
      std::shared_ptr<const LidarPointCloud> lidar_cloud = nullptr,
      LidarMatchingOptions lidar_matching_options = LidarMatchingOptions(),
      LidarBundleAdjustmentOptions lidar_ba_options =
          LidarBundleAdjustmentOptions(),
      double lidar_phase1_weight = 0.05,
      double lidar_phase2_weight = 1.0,
      bool fix_poses_in_lidar_ba = true);
  ~SixDofPriorGlobalMapper() override = default;

  bool Solve(const PriorGlobalMapperOptions& options);

 protected:
  bool ShouldUseSixDofPosePriors(const PriorGlobalMapperOptions& options) const;

  size_t FilterPoseGraphEdgesWithSixDofPriors(
      double max_rotation_error_deg,
      double max_translation_direction_error_deg);

  size_t SeedRotationsFromSixDofPriors(bool reset_translations);

  size_t EnforceMaxRotationPriorDeviation(double max_rotation_deviation_deg,
                                          const char* stage_name);

  void RestorePosesFromSixDofPriors(const char* stage_name);

    size_t DeRegisterFramesWithTooFewObservations(
            int min_observations_per_registered_image,
            const char* stage_name);

  bool ValidateReconstructionQuality(double min_mean_reprojection_error,
                                     int min_observations_per_registered_image,
                                     const char* stage_name);

  bool RunSixDofBundleAdjustment(
      const BundleAdjustmentOptions& ba_options,
      const PriorGlobalMapperOptions& mapper_options,
      const char* stage_name,
      const PosePriorBundleAdjustmentOptions& prior_options,
      double prior_rotation_fallback_stddev_rad,
      bool populate_temporal_triplets = false);

  // Populate temporal_triplets_ from six_dof_pose_priors_. Called once in
  // the constructor; grouped by sensor_id, sorted by timestamp.
  void BuildTemporalTriplets();

  // Populate rig_pairs_ by splitting priors on image-name prefix, sorting
  // each side by timestamp and matching by nearest-time. Reconstruction runs
  // that do not enable use_rig_pair_prior still pay only the O(n log n)
  // construction cost. Called on demand from Solve() when the rig pair prior
  // is enabled; the resulting pairs are later forwarded into each
  // BundleAdjustmentConfig via SetRigPairs.
  void BuildRigPairs(const std::string& i_prefix,
                     const std::string& j_prefix,
                     double max_dt_seconds);

  // Estimate rig_pair_baseline_ (i_from_j) by robustly averaging the
  // corresponding relative pose across all built rig_pairs_. Called only
  // when BuildRigPairs produced at least one pair and the user has not
  // supplied an explicit override.
  void EstimateRigBaseline();

  bool EstimateRobustMetricAlignmentTransform(
      const PriorGlobalMapperOptions& mapper_options,
      const char* stage_name,
      Sim3d* metric_from_current,
      std::string* alignment_source,
      size_t* num_correspondences) override;

  bool IterativeSixDofBundleAdjustment(
      const BundleAdjustmentOptions& options,
      const PriorGlobalMapperOptions& mapper_options,
      const PosePriorBundleAdjustmentOptions& prior_options,
      double prior_rotation_fallback_stddev_rad,
      double max_normalized_reproj_error,
      double min_tri_angle_deg,
      int num_iterations,
      bool skip_fixed_rotation_stage,
      bool skip_joint_optimization_stage);

  bool IterativeRetriangulateAndRefineWithSixDofPriors(
      const IncrementalTriangulator::Options& tri_options,
      const BundleAdjustmentOptions& ba_options,
      const PriorGlobalMapperOptions& mapper_options,
      const PosePriorBundleAdjustmentOptions& prior_options,
      double prior_rotation_fallback_stddev_rad,
      double max_normalized_reproj_error,
      double min_tri_angle_deg);

  bool IterativeRetriangulateAndRefineVisualOnlyWithSnapshots(
      const IncrementalTriangulator::Options& tri_options,
      const BundleAdjustmentOptions& ba_options,
      double max_normalized_reproj_error,
      double min_tri_angle_deg);

  bool ShouldUseLidarPointToPlaneInRetriangulation(
      const PriorGlobalMapperOptions& options) const;

  bool RunLidarPointToPlaneRetriangulationAlignment(
      const BundleAdjustmentOptions& ba_options,
      const PriorGlobalMapperOptions& mapper_options,
      const PosePriorBundleAdjustmentOptions& prior_options,
      double prior_rotation_fallback_stddev_rad,
      const std::unordered_set<point3D_t>& recent_point3D_ids,
      const char* stage_name);

  bool ApplyEarlyRetriangulationObservationCleanup(
      class ObservationManager& obs_manager,
      const std::unordered_set<point3D_t>& point3D_ids,
      double max_reprojection_error_px,
      const char* stage_name);

  bool RunPostEnforcementCleanup(
      const PriorGlobalMapperOptions& mapper_options,
      const PosePriorBundleAdjustmentOptions& prior_options,
      bool use_prior,
      bool use_6dof,
      double prior_rotation_fallback_stddev_rad);

    bool ApplyPostEnforcementObservationResidualFilter(
      double max_reprojection_error_px);

 private:
  std::vector<SixDofPosePrior> six_dof_pose_priors_;
  std::vector<AbsolutePosePriorConstraint> absolute_pose_priors_;
  std::unordered_map<image_t, const SixDofPosePrior*> image_to_six_dof_prior_;
  std::vector<TemporalSmoothnessTriplet> temporal_triplets_;
  std::vector<RigPairCorrespondence> rig_pairs_;
  Rigid3d rig_pair_baseline_;
  bool rig_pair_baseline_ready_ = false;
  std::shared_ptr<const LidarPointCloud> lidar_cloud_;
  LidarMatchingOptions lidar_matching_options_;
  LidarBundleAdjustmentOptions lidar_ba_options_;
    double lidar_phase1_weight_ = 0.05;
  double lidar_phase2_weight_ = 1.0;
  bool fix_poses_in_lidar_ba_ = true;
};

}  // namespace colmap