# 大纲

## 目标
- 新增 `python\util\import_lidar_priors_to_db.py`，把激光雷达稀疏模型中的 6DoF 位姿先验写入 COLMAP 数据库
- 在 GLOMAP `global_mapper` 中接入 6DoF 位姿先验，不只在 bundle adjustment 中使用，也要参与主要全局求解阶段
- 让 retriangulation 之前的 refinement 不再是纯视觉流程，而是继续保留 6DoF 位姿约束
- retriangulation 添加开关，默认为纯视觉，不添加其他任何约束
- 对最终解算结果与 `sparse_lidar` 做对比，验证 6DoF 约束是否有效提升重建稳定性和精度
- 添加 激光的点到平面约束，在retriangle 环节，同时设置参数，只有当重投影误差小于 `3 px` 时才进行稀疏点到激光点的约束，开启后就不会进行关闭，然后这个阈值可以通过cli 修改。

补充原则：
- 优先保证视觉内部一致性，6DoF 位姿先验和激光先验的角色是增强困难场景下的鲁棒性，而不是替代视觉几何本身
- 因此推荐配置应优先选择“不破坏视觉重建自洽性”的方案，而不是单纯追求更贴近 prior 的数值结果

## 验收标准
- 平均相机 center 误差小于 `0.2 m`
- 最大相机 center 误差小于 `0.5 m`
- 重投影误差小于 `3 px`
- `mean_reprojection_error == 0` 视为解算失败
- 不允许出现 `zero-observation image`

# 测试数据

添加了6dof 位姿先验的数据库 

"D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\db_lidar_priors_test.db"

原始图像数据 

"D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\images_raw"

激光雷达先验的位姿稀疏模型数据 

"D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\sparse_lidar"

测试结果写到这个文件夹下创建新的文件夹

"D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\results"

降采样后的激光数据

## 已完成的能力

### 1. 数据库与先验导入
- 已新增 `python\util\import_lidar_priors_to_db.py`
- 已支持把 `sparse_lidar` 的 6DoF 位姿先验写入数据库表 `6dof_pose_priors`
- 已写入完整先验字段：`qvec / tvec / position / rotation_covariance / position_covariance / gravity / coordinate_system`

### 2. 6DoF 主流程接入
- 已新增 `SixDofPriorGlobalMapper`
- 当数据库存在 `6dof_pose_priors` 时，`global_mapper` 默认自动启用 6DoF 路径
- 6DoF 位姿先验已经接入：
	- pose graph filtering
	- rotation averaging
	- global positioning
	- iterative bundle adjustment
	- retriangulation refinement
- 已修复 global positioning 中 seeded center 非有限导致的求解失败问题

### 3. LiDAR 结构约束接入
- 已支持在 6DoF retriangulation refinement 中引入 LiDAR 点到平面约束
- 已支持 `GlobalMapper.lidar_retriangulation_max_reprojection_error`
	- 仅当稀疏点重投影误差不超过阈值时，才参与 LiDAR 约束
- 已支持“早期但保守”的 LiDAR 稳定性门控与早期清理：
	- `GlobalMapper.lidar_retriangulation_early_min_track_length`
	- `GlobalMapper.lidar_retriangulation_early_max_mean_reprojection_error`
	- `GlobalMapper.lidar_retriangulation_early_post_alignment_max_reprojection_error_px`
	- 作用是让早期 LiDAR 只作用在视觉上已经相对稳定的 anchor 上，并在每轮 refinement 后尽早裁掉坏 observation，抑制错误局部结构扩散
- 已支持保守模式 `GlobalMapper.use_lidar_point_to_plane_only_in_final_retriangulation=1`
	- LiDAR 约束只进入 final pass，避免在前几轮 refinement 中把局部结构拉偏
- 已支持 final joint `6DoF + LiDAR` BA
	- 通过 `GlobalMapper.lidar_fix_poses_in_lidar_ba=0` 启用

### 4. Final enforcement 与 clean-final 清理
- 已支持 `GlobalMapper.max_position_prior_deviation`
- 已支持 `GlobalMapper.clamp_positions_to_prior_after_optimization`
- 已支持 `GlobalMapper.delete_frames_with_position_prior_deviation_after_optimization`
	- 可在优化结束后直接删除偏离先验过大的 frame，而不是强行保留
- 已支持 `GlobalMapper.post_enforcement_max_reprojection_error_px`
	- 在 final enforcement 之后，按像素残差裁剪高残差观测
- 当前 post-enforcement filter 已显式改成 observation-level 语义：
	- 先删坏观测
	- 仅当 track 长度不足时才删点

### 5. 质量校验与稳定性保护
- 已修复最终 observation 统计误判问题，不再把正常结果误判为 `0 observation`
- 已支持在最终写出前退注册观测不足的 frame，避免模型写出失败
- 已把 `mean_reprojection_error == 0` 明确视为异常失败信号
- 当前流程默认保证：
	- 不接受 `zero-observation image`
	- 不接受零重投影误差退化解

## 当前推荐配置

### 推荐目标
- 当前推荐的是“以视觉内部一致性为主、6DoF + LiDAR 为辅”的 clean-final 策略
- 目标不是单纯把 pose 拉回 prior，而是在尽量不破坏视觉几何自洽性的前提下，用 6DoF 和 LiDAR 提升困难场景下的稳定性，并同时满足姿态误差和重投影误差门槛

### 当前最优 clean-final 配置
- 结果目录：`results/global_mapper_6dof_auto_default_ba_lidar_retri_early_gate_cleanup_v2/0`
- 这也是当前代码层面推荐保留的 clean-final 方案
- 当前代码默认值已经对齐到这套方案的算法参数；唯一保留手动控制的是 `GlobalMapper.num_threads`

对应命令行：

```powershell
$env:QT_QPA_PLATFORM_PLUGIN_PATH='D:\Code\Cpp\colmap\build\vcpkg_installed\x64-windows\Qt6\plugins\platforms'
$env:QT_PLUGIN_PATH='D:\Code\Cpp\colmap\build\vcpkg_installed\x64-windows\Qt6\plugins'
$env:PATH='C:\Program Files\NVIDIA cuDSS\v0.7\bin\12;D:\Code\Cpp\colmap\build\_deps\onnxruntime-build\lib;D:\Code\Cpp\colmap\build\vcpkg_installed\x64-windows\bin;D:\Code\Cpp\colmap\build\vcpkg_installed\x64-windows\Qt6\plugins\platforms;' + $env:PATH
& 'D:\Code\Cpp\colmap\build\src\colmap\exe\RelWithDebInfo\colmap.exe' global_mapper \
	--database_path 'D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\db_lidar_priors_test.db' \
	--image_path 'D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\images_raw' \
	--output_path 'D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\results\global_mapper_6dof_auto_default_ba_lidar_retri_early_gate_cleanup_v2' \
	--GlobalMapper.use_prior_position 0 \
	--GlobalMapper.use_6dof_pose_priors 1 \
	--GlobalMapper.use_6dof_retriangulation_refinement 1 \
	--GlobalMapper.lidar_point_cloud_path 'D:\Data\HP300\2025-12-20_qiantai\mapper\2025-12-20_qiantai_num_1\splat_data\downsample_with_normals.ply' \
	--GlobalMapper.use_lidar_point_to_plane_in_retriangulation 1 \
	--GlobalMapper.use_lidar_point_to_plane_only_in_final_retriangulation 0 \
	--GlobalMapper.lidar_fix_poses_in_lidar_ba 0 \
	--GlobalMapper.lidar_phase2_weight 0.5 \
	--GlobalMapper.lidar_retriangulation_max_reprojection_error 3 \
	--GlobalMapper.lidar_retriangulation_early_min_track_length 4 \
	--GlobalMapper.lidar_retriangulation_early_max_mean_reprojection_error 2 \
	--GlobalMapper.lidar_retriangulation_early_post_alignment_max_reprojection_error_px 2 \
	--GlobalMapper.max_position_prior_deviation 0.5 \
	--GlobalMapper.clamp_positions_to_prior_after_optimization 1 \
	--GlobalMapper.delete_frames_with_position_prior_deviation_after_optimization 0.5 \
	--GlobalMapper.post_enforcement_max_reprojection_error_px 3 \
	--GlobalMapper.min_mean_reprojection_error 1e-6 \
	--GlobalMapper.min_observations_per_registered_image 1 \
	--GlobalMapper.random_seed 42 \
	--GlobalMapper.num_threads 1
```

建议配置如下：
- `GlobalMapper.use_prior_position=0`
- `GlobalMapper.use_6dof_pose_priors=1`
- `GlobalMapper.use_6dof_retriangulation_refinement=1`
- `GlobalMapper.lidar_point_cloud_path=downsample_with_normals.ply`
- `GlobalMapper.use_lidar_point_to_plane_in_retriangulation=1`
- `GlobalMapper.use_lidar_point_to_plane_only_in_final_retriangulation=0`
- `GlobalMapper.lidar_fix_poses_in_lidar_ba=0`
- `GlobalMapper.lidar_phase2_weight=0.5`
- `GlobalMapper.lidar_retriangulation_max_reprojection_error=3`
- `GlobalMapper.lidar_retriangulation_early_min_track_length=4`
- `GlobalMapper.lidar_retriangulation_early_max_mean_reprojection_error=2`
- `GlobalMapper.lidar_retriangulation_early_post_alignment_max_reprojection_error_px=2`
- `GlobalMapper.max_position_prior_deviation=0.5`
- `GlobalMapper.clamp_positions_to_prior_after_optimization=1`
- `GlobalMapper.delete_frames_with_position_prior_deviation_after_optimization=0.5`
- `GlobalMapper.post_enforcement_max_reprojection_error_px=3`
- `GlobalMapper.min_mean_reprojection_error=1e-6`
- `GlobalMapper.min_observations_per_registered_image=1`
- `GlobalMapper.num_threads` 仍建议显式手动传入，不作为固定默认推荐值

说明：
- 虽然命令里保留了 `GlobalMapper.clamp_positions_to_prior_after_optimization=1`
- 但由于删除阈值与偏离阈值同为 `0.5 m`，在当前推荐配置里，超阈值 frame 会被直接删除，因此最终语义是 delete-only

### 当前推荐配置的结果
- `registered_images: 1063`
- `points3D: 50925`
- `mean_reprojection_error_px: 0.895035`
- `mean_observations_per_image: 314.492004`
- `zero_observation_images: 0`
- `common_images: 1061`
- `mean_center_error_m: 0.012495`
- `median_center_error_m: 0.011186`
- `p95_center_error_m: 0.024605`
- `max_center_error_m: 0.081422`
- `mean_rotation_error_deg: 0.322881`
- `max_rotation_error_deg: 1.179115`

### 最新 phase-split sweep 结论
- 已实现并验证：6DoF retriangulation 的 early refinement 使用 `phase=1`，final alignment 使用 `phase=2`
- 在此基础上测试：
	- `GlobalMapper.lidar_retriangulation_early_max_mean_reprojection_error=1.5`
	- `GlobalMapper.lidar_phase2_max_distance=0.10 / 0.08 / 0.05`
- 三组结果目录：
	- `results/global_mapper_6dof_auto_default_ba_lidar_retri_phase1early_er15_p210_v1/0`
	- `results/global_mapper_6dof_auto_default_ba_lidar_retri_phase1early_er15_p208_v1/0`
	- `results/global_mapper_6dof_auto_default_ba_lidar_retri_phase1early_er15_p205_v1/0`
- 结果摘要：
	- 三组都保留 `1063` 张注册图像，`common_images=1061`
	- `p210` / `p208` / `p205` 的最终指标完全一致：
		- `points3D: 50939`
		- `mean_reprojection_error_px: 0.895255`
		- `mean_center_error_m: 0.012535`
		- `p95_center_error_m: 0.024713`
		- `max_center_error_m: 0.074429`
		- `mean_rotation_error_deg: 0.323111`
		- `max_rotation_error_deg: 1.176580`
- 与当前推荐配置相比：
	- `points3D` 略多：`50939 > 50925`
	- 但 `mean_reprojection_error_px` 小幅变差：`0.895255 > 0.895035`
	- `mean_center_error_m` 也小幅变差：`0.012535 > 0.012495`
	- `max_center_error_m` 略好：`0.074429 < 0.081422`
- 结论：
	- `phase2_max_distance` 从 `0.10` 收紧到 `0.08` 或 `0.05` 只减少了 final phase-2 LiDAR 约束数，没有改变最终解
	- 这轮 phase-split + `early_max_mean_reprojection_error=1.5` 没有超越当前推荐配置，因此当前推荐不变

### 最新多线程复跑结论
- 已对当前最优配置做“只改线程数”的复跑：
	- 结果目录：`results/global_mapper_6dof_auto_default_ba_lidar_retri_early_gate_cleanup_mt8_v1/0`
	- 唯一改动：`GlobalMapper.num_threads=8`
- 结果：
	- 总时长从约 `947.991 s` 降到 `745.242 s`
	- 但注册图像从 `1063` 降到 `1058`
	- `mean_reprojection_error_px` 从 `0.895035` 变成 `0.906540`
	- `max_center_error_m` 从 `0.081422` 变成 `0.109140`
- 结论：
	- 多线程确实能提速
	- 但 `num_threads=8` 会把结果带到另一条略差的解上，不满足“速度提升且不降低指标”
	- 因此当前最优方案仍然是上面的 `early_gate_cleanup_v2 + num_threads=1`

### 默认值调整结论
- 当前代码默认值已改为贴近 `early_gate_cleanup_v2` 的算法配置，包括：
	- `GlobalMapper.use_prior_position=0`
	- `GlobalMapper.use_lidar_point_to_plane_in_retriangulation=1`
	- `GlobalMapper.lidar_fix_poses_in_lidar_ba=0`
	- `GlobalMapper.lidar_phase2_weight=0.5`
	- `GlobalMapper.max_position_prior_deviation=0.5`
	- `GlobalMapper.clamp_positions_to_prior_after_optimization=1`
	- `GlobalMapper.delete_frames_with_position_prior_deviation_after_optimization=0.5`
- 未改成默认的仍包括：
	- `GlobalMapper.num_threads`
	- 数据路径类参数，例如 `database_path` / `image_path` / `lidar_point_cloud_path` / `output_path`

### 为什么当前推荐这一组
- 6DoF 先验已经贯穿主要求解链路，不再只是 BA 里的弱约束
- LiDAR 不再被完全推迟到 final pass，而是通过“稳定 anchor 门控 + 早期 observation cleanup”更早参与 refinement
- 这使 LiDAR 可以更早抑制错误局部结构，同时避免把视觉上还不稳定的稀疏点拿去强行贴 LiDAR
- final 阶段不再保留明显偏离先验的坏 frame，意味着先验只作为鲁棒性兜底，而不是强行主导最终结构
- post-enforcement `3 px` 残差过滤能继续降低真实后验重投影误差，同时不会打穿 `zero-observation image` 约束
- 这组配置同时满足：
	- 平均 center 误差小于 `0.2 m`
	- 最大 center 误差小于 `0.5 m`
	- 重投影误差小于 `3 px`
	- `zero-observation image = 0`

和此前的 final-only clean-final 基线相比，这组新配置表现为：
- 保留了更多注册图像：`1063 > 1058`
- 降低了平均重投影误差：`0.895 px < 0.981 px`
- 明显降低了最大 center 误差：`0.081 m < 0.119 m`
- 同时没有破坏“视觉内部一致性优先”的原则

从设计原则上看，这组配置的优势是：
- 它没有把 6DoF 或 LiDAR 提升成“压过视觉几何”的硬主导项
- 它更接近“视觉先站稳一部分结构，再让 LiDAR 只介入稳定部分，并及时清理异常结构”的策略
- 因此更符合“最优的还是视觉内部一致性”这一目标

## 备选配置

### 如果优先追求最强绝对姿态精度
- 可使用 `joint final 6DoF+LiDAR BA + final clamp` 方案
- 结果目录：`results/global_mapper_6dof_auto_default_ba_lidar_retri_pose_final_joint_clamp_v1/0`
- 该方案指标更强：
	- `mean_center_error_m: 0.012019`
	- `max_center_error_m: 0.133345`
	- `mean_reprojection_error_px: 0.000970`
- 但这个方案会保留一部分需要靠 final clamp 才能回到 prior 附近的 frame，因此从“视觉内部一致性优先”的角度，它不如当前 clean-final 推荐方案自然

## 当前结论
- 当前 6DoF + LiDAR 路径已经完整可用，并且已经形成稳定推荐配置
- 从“视觉内部一致性优先，同时仍满足全部验收门槛”的角度，当前最佳方案是：
	- `早期 gated LiDAR refinement + delete-only + post_enforcement_max_reprojection_error_px=3`
- 从“绝对姿态误差尽可能低，即使更依赖 final prior enforcement”的角度，当前最佳方案是：
	- `joint final 6DoF+LiDAR BA + final clamp`
- 当前代码默认更偏向 clean-final 路径：
	- `GlobalMapper.post_enforcement_max_reprojection_error_px` 默认已切到 `3.0`

## 还有什么可以优化

### 1. 可重复性还需要在“完整目标配置”下补验证
- 已经确认固定 seed 和单线程可以让简化配置稳定复现
- 但还没有用“完整 LiDAR clean-final 配置”做完严格的 `3 px / 5 px` 多次重复实验
- 下一步应使用完全相同的 LiDAR 参数、删除参数和 final enforcement 参数，再加：
	- `GlobalMapper.random_seed=固定值`
	- `GlobalMapper.num_threads=1`
	- 验证 `3 px` 与 `5 px` 的真实稳定性差异
	- 重点确认当前推荐方案是否能在不破坏视觉内部一致性的前提下稳定复现

### 2. 继续减少 final delete-only 的覆盖率损失
- 当前 `3 px` clean-final 已经很好，但仍会删除一部分 frame
- 后续仍可继续优化的是：
	- 减少最终被删 frame 数量
	- 同时不放大 `max_center_error_m`
	- 同时不回升 `mean_reprojection_error_px`
	- 同时不引入需要依赖更强 prior clamp 才能维持的结构

### 3. 让 LiDAR 约束更早发挥正作用，而不是只在 final pass 稳妥生效
- 当前已经验证：LiDAR 可以更早进入 refinement，但前提是必须配套稳定性门控与早期 observation 清理
- 这说明问题不在于“LiDAR 不能早用”，而在于“不能让 LiDAR过早绑定视觉上不稳定的结构”
- 后续可以继续优化：
	- 继续细化早期 stable-anchor 判定
	- 调整早期 cleanup 阈值，减少不必要的 observation 丢失
	- 分阶段调度 LiDAR residual 权重与 loss
	- 核心目标不是“更早加更多 LiDAR”，而是“只在不破坏视觉内部一致性时再让 LiDAR 发力”

### 4. 把 post-enforcement filter 做成更强的确定性 clean-final 组件
- 现在已经明确改成 observation-level filter
- 后续还可以继续完善：
	- 在完整目标配置下验证稳定性
	- 进一步减少由遍历顺序、线程数、边界 track 长度带来的细微波动
	- 明确区分“删坏观测”和“删整点”的统计输出
	- 让 clean-final 更像视觉结果清理，而不是额外的人为几何重塑

### 5. 针对局部高风险 frame 做更早期的结构抑制
- 当前 delete-only 能把最终坏 frame 清掉，但这是 final stage 的止损
- 后续若要进一步减少覆盖率损失，最有价值的是把问题前移：
	- 更早拒绝错误局部结构
	- 避免错误三角化进入 BA
	- 减少最后必须删除的 frame 数量
	- 优先提高视觉内部自洽结构的保留率，而不是事后依赖更强的先验纠偏

## 当前建议
- `task.md` 后续建议继续保持“结论文档”风格，不再记录完整实验流水账
- 新实验若要补充，优先只追加：
	- 采用的配置
	- 核心结果指标
	- 是否优于当前推荐配置
	- 是否改变推荐结论

补充实现决策：
- 已将 `GlobalMapper.post_enforcement_max_reprojection_error_px` 的默认值切换为 `3.0`
- 因此当前代码里的 clean-final 默认策略，已经优先采用 `delete-only + 3 px pixel residual filter`
- 这一默认策略也更符合“视觉内部一致性优先，先验负责鲁棒性兜底”的当前原则
- 若后续需要回退到旧行为，可显式传：
	- `GlobalMapper.post_enforcement_max_reprojection_error_px=-1`
