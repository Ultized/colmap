# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 优先阅读

本仓库的权威说明文档是根目录下的 **`AGENTS.md`**。里面已经包含：

- 完整的目录结构和模块依赖分层（util → math → geometry → sensor → feature → optim → scene → estimators → sfm/mvs → controllers → exe / ui）
- 所有关键类/文件的位置索引（`Reconstruction`、`Rigid3d`、`IncrementalMapper`、`BundleAdjuster` 等）
- 通用的 Linux/Ninja 构建、ctest、pycolmap 构建命令
- 命名约定、坐标约定（`target_from_source`、`x_in_y`）、特殊标识符类型（`image_t`、`point3D_t` …）
- 依赖列表（Eigen / Ceres / Boost / SQLite / OpenImageIO / PoseLib / FAISS + CUDA / ONNX / Qt / CGAL 可选项）

处理本仓库相关任务时，**先读 `AGENTS.md`**；本文件只补充它没覆盖的内容。

## 本地环境（Windows + vcpkg）

开发机为 Windows 11 + bash（Git Bash / MSYS），路径形式 `D:/Code/Cpp/colmap`。依赖走 **vcpkg manifest**（`vcpkg.json` + `vcpkg-configuration.json`），CMake 构建类型以 `Release` 为主，开启 `IPO_ENABLED=ON` 时链接耗时显著。

`build/` 目录一直保留已配置过的 CMake 缓存；除非用户明确要求"删除缓存重新构建"，**优先走增量 ninja**，不要重新跑 `cmake ..`：

```bash
cmake --build build --config Release -j 32        # 增量构建，默认 -j 32
cmake --build build --target colmap -j 32         # 单目标
```

需要干净重建时：

```bash
rm -rf build && mkdir build
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32
```

### Windows 构建的两个硬性前提

**1. CUDA_PATH 必须设置**（vcpkg `cuda` 端口在 CMake 子进程里用 `find_program(NVCC)` 检测，不读 shell PATH）：
```bash
export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
```
或在 Windows 系统环境变量里永久添加 `CUDA_PATH`。

**2. CUDA 编译必须在 VS Dev Shell 里跑**（Git Bash 缺 `%INCLUDE%`/`%LIB%`，nvcc 的 host compiler cl.exe 会找不到 MSVC 标准库头文件）。从 PowerShell 进入 VS 环境后再构建：
```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -SkipAutomaticLocation
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
Set-Location 'D:\Code\Cpp\colmap'
cmake --build build --config Release -j 32
```

如果 cmake configure 被 cmake 文件变更触发了软重配（vcpkg 重跑），同样需要在此环境下执行，否则 cuda 端口会 BUILD_FAILED。

### CUDA 架构 / 最低版本

`cmake/FindDependencies.cmake` 把 `CMAKE_CUDA_ARCHITECTURES` 固定成 `{75,80,86,89,90}`（Turing/Ampere/Ada/Hopper），`CUDA_MIN_VERSION` 提到 10.0。不再用 `native`（CMake 偶发误检本机 GPU）或 `all-major`（CI 产物膨胀）。要支持新卡或砍掉旧卡，改这一处即可。

### CASPAR CUDA BA 后端（默认编入）

`CASPAR_ENABLED=ON` + `CASPAR_USE_DOUBLE=ON` 是默认值，所以 colmap.exe 一律内置 SymForce/CASPAR 生成的 GPU BA 代码路径。**默认运行时仍走 Ceres**，需要 GPU BA 时显式选：

```
--Mapper.ba_local_backend caspar
--Mapper.ba_global_backend caspar
--BundleAdjustment.backend caspar       # 单独 bundle_adjuster 子命令
```

Windows + MSVC 上有一个 force-include 兼容层 `src/thirdparty/symforce_caspar_compat.h`，由 `src/thirdparty/CMakeLists.txt` 通过 `--pre-include` / `/FI` 注入到 `caspar_lib_core` 的编译选项里。它解决两个 SymForce 生成代码的硬伤：
- 生成的 CUDA 头用 POSIX `uint` typedef（glibc 隐式带，MSVC 没有）。必须用 `typedef unsigned int uint;`，**不能用 `#define`**——宏会污染 CUDA 自己 `<vector_types.h>` 里 `uint1`/`uint2` 的 token-paste 展开，破坏 `tuple_size` 特化。
- 生成的 `solver.cc` 不显式 include `<string>` / `<stdexcept>` 就用 `std::to_string` / `std::runtime_error`（libstdc++ 隐式带，MSVC 不带）。

不要去改 `src/thirdparty/symforce_*` 下面 SymForce 自动生成的源文件，所有 Windows 兼容补丁都集中在那个 compat header + 注入它的 CMake 逻辑里。

运行单个 C++ 测试（AGENTS.md 里有完整语法，这里是常用速记）：

```bash
ctest --test-dir build -R "^scene/reconstruction_test$" --output-on-failure
```

## 分支状态

- 默认分支：`main`（用于给上游 PR，不含自研内容）
- `6dof`：与 `main` 同步的 upstream 镜像（`180ce092`，2026-04-16）
- `6dof-lidar`：**目前的主力开发分支**，基于 `6dof` 追加自研 commit —— 6DoF prior + LiDAR global mapping 集成 + Python 工具脚本
- `re_dev` / `develop-dev` / `develop2` / `pose-prior` 等：旧探索分支，已被 `6dof-lidar` 取代，仅留作备档

本仓库是 COLMAP 上游的 fork / 镜像。

## 6dof-lidar 自研模块地图

只在 `6dof-lidar` 分支存在、AGENTS.md 未覆盖的新增文件：

| 文件 | 作用 |
|------|------|
| `src/colmap/geometry/kdtree3d.h` | header-only KD-tree（LiDAR 最近点查询） |
| `src/colmap/scene/lidar_point_cloud.{h,cc}` | LiDAR 点云容器 + 加载/查询封装 |
| `src/colmap/sfm/prior_global_mapper.{h,cc}` | 6DoF 位姿先验的 global mapping（旋转/位置分离优化） |
| `src/colmap/sfm/six_dof_prior_global_mapper.{h,cc}` | 上面加强版，支持 constant-velocity SE(3) 时序平滑先验 |
| `src/colmap/sfm/lidar_global_mapper.{h,cc}` | 在 prior global mapping 之上融合 LiDAR 点云约束的 global mapper |
| `src/colmap/estimators/cost_functions/pose_prior.h` | Ceres cost functor：6DoF pose prior（位置+姿态） |
| `src/colmap/estimators/cost_functions/dead_zone_loss.h` | Ceres `LossFunction`：死区损失（rig-extrinsic 软 prior，在容差内零代价） |
| `src/colmap/estimators/cost_functions/lidar.h` | Ceres cost functors：`PointToPlaneCostFunctor`（1残差）和 `PointToPointCostFunctor`（3残差）用于 LiDAR 点云约束 BA |
| `src/colmap/scene/database_sqlite.{h,cc}` 增量 | 新增 `pose_priors` 表读写 |
| `src/colmap/controllers/global_pipeline.{h,cc}` 增量 | `GlobalPipelineOptions::mapper` 类型替换为 `LidarGlobalMapperOptions`，`global_mapper` 子命令内部走 6DoF prior + LiDAR 通路 |
| `src/colmap/controllers/option_manager.cc` 增量 | 注册 6DoF prior / LiDAR / rig-extrinsic dead-zone / temporal-smoothness / `generate_scales` 等 CLI 选项 |

CLI 子命令两个相关入口（dispatcher 在 `src/colmap/exe/colmap.cc`，实现在 `src/colmap/exe/sfm.cc`）：

- `colmap global_mapper` —— 全局 SfM；6dof-lidar 分支上已整合 6DoF prior + LiDAR BA，**不是新增子命令**
- `colmap pose_prior_mapper` —— 增量式 pose-prior mapper（上游已有，与 6dof-lidar 自研 global 通路并存）

## Python 预处理工具（`python/util/`）

这条 LiDAR pipeline 依赖的一组预/后处理脚本，名字即功能：

| 脚本 | 作用 |
|------|------|
| `imgpose_to_sparse_lidar.py` | SLAM 输出的 `ImgPose.txt` → 初始 sparse model |
| `import_lidar_priors_to_db.py` | 把 6DoF prior 写进 COLMAP SQLite `pose_priors` 表 |
| `downsample_las_to_ply_with_normals.py` / `downsample_lidar_ply_with_plane_filter.py` | LAS/PLY 点云降采样（带法线 / 平面过滤） |
| `open3d_gicp_sim3_alignment.py` | GICP Sim3 对齐（sparse model → LiDAR） |
| `compare_sparse_pose_models.py` / `evaluate_reconstruction_result.py` | 位姿/重建结果对比评估 |
| `rename_sparse_model_from_database_ids.py` | 按 DB image_id 重命名 sparse model 的 image 名 |
| `triangulate_then_optimize_poses.py` | 先三角化再固定 3D 点只优化 pose 的独立工具 |

## 语言约定（与全局 CLAUDE.md 一致）

- 对话、解释性回答、散文说明：**中文**
- 代码、注释、commit message、PR 标题与描述、分支名、文件名、变量名：**英文**（匹配仓库已有风格，走 Google C++ Style + `.clang-format`）

## 格式化

改完 C++ / Python 后在提交前过一遍格式化脚本：

```bash
scripts/format/c++.sh         # clang-format, 仅处理变动文件
scripts/format/python.sh      # ruff format + check
```

`.clang-format` 和 `ruff.toml` 是权威配置，不要手写对齐。

## Claude 工作守则（本仓库专属）

- **不要替用户重新跑 cmake 配置步骤**。CMake 重配会触发 vcpkg 重新拉依赖，耗时十几分钟到几十分钟。除非 `CMakeCache.txt` 明显坏了或用户要求，永远走增量构建。
- **多线程默认 `-j 32`**。用户机器核心多，单 job 会严重拖慢构建。
- CLI 子命令入口在 `src/colmap/exe/colmap.cc`（dispatcher），各领域具体实现在同目录下的 `*.cc` 文件。添加新子命令时两处都要改。
- UI / GUI 相关改动（`src/colmap/ui/`）只能手动目视验证，无法自动化——如果改了 UI 代码，明确告知用户"此项仅构建通过，未跑交互测试"，不要声称"已验证"。
- 跨 session 接手任务时，开头先 `git status` + `git log -5` 确认起点状态，不要假设上个 session 留下的中间状态还在。
- **自研 BA 选项的命名规律**：`BundleAdjustment.*` 是底层 `BundleAdjustmentOptions` 字段；`GlobalMapper.ba_*` 是将同一字段透传到 `SixDofPriorGlobalMapper` 各阶段 BA 的别名，两套入口并存。新增 CLI 选项时两处都要注册（`option_manager.cc`）。
- **改动被广泛 include 的头文件后必须强制重编全部依赖**。Windows + ninja + MSVC 的 `/showIncludes` 对传递依赖不可靠（例如改 `observation_manager.h`，ninja 可能只重编直接 include 它的 `.cc`，不重编间接 include 的单元）。stale `.obj` 会表现为运行时 access violation / 逻辑错乱，极难归因。改 `scene/reconstruction.h`、`sfm/*.h`、`scene/database*.h` 这类广泛头后，要 `ninja -t clean <target>` 或删该模块 `CMakeFiles/*.dir/` 再重编。
