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

## 本地环境（双开发环境：Windows 主构建 + Linux 并行）

依赖走 **vcpkg manifest**（`vcpkg.json` + `vcpkg-configuration.json` + `.vcpkg-overlay/ports/`），CMake 构建类型以 `Release` 为主，generator 是 `Ninja Multi-Config`，开启 `IPO_ENABLED=ON` 时链接耗时显著。

两个并列的 checkout，各自维护独立 `build/`：

| 环境 | 路径 | 用途 |
|------|------|------|
| Windows 11 + Git Bash | `D:/Code/Cpp/colmap` | 主构建/调试，CUDA 走 vcpkg `cuda` 端口 |
| Linux + zsh | `/data/wangzhaolong/Code/Cpp/colmap`、`/home/wangzhaolong/Code/Cpp/colmap` | 并行开发、阅读代码、Linux 上跑 ctest |

⚠️ **Linux checkout 的 `build/` 可能是从 Windows 同步过来的镜像**：`build/CMakeCache.txt` 里 `VCPKG_INSTALLED_DIR` / `VCPKG_MANIFEST_DIR` 会硬编码 `D:/Code/Cpp/colmap/...`。**从 Linux 直接 `cmake --build build` 会立刻触发重配并失败**。Linux 上要么用独立的 `build-linux/`，要么先 `rm -rf build && cmake -S . -B build ...` 重新生成 Linux 自己的缓存。两个 OS 不要共享同一个 `build/`。

### Linux 上跑通的实战配方（验证日期 2026-05-08）

由于代理对 github HTTPS 长传输不稳定，**vcpkg 跑不了 manifest mode**（baseline fetch 持续超时）。本机 Linux 走的路径：

1. **vcpkg classic mode**：用本地 `/data/wangzhaolong/Code/Cpp/vcpkg` 直接逐个 `./vcpkg install <port>:x64-linux`。
   - 必装：`metis cgal gtest poselib faiss glew openimageio` 加 `'ceres[lapack,schur,suitesparse]'`（默认 ceres 缺 suitesparse，COLMAP 的 `find_package(CHOLMOD REQUIRED)` 会失败）。
   - **OpenImageIO 的 source tarball 必须走 gh-proxy.com 镜像下到 `/data/wangzhaolong/Code/Cpp/vcpkg/downloads/AcademySoftwareFoundation-OpenImageIO-v3.0.9.1.tar.gz`**（github 直连 + proxy 都会被截断）：
     ```bash
     unset https_proxy http_proxy
     wget -O AcademySoftwareFoundation-OpenImageIO-v3.0.9.1.tar.gz \
       "https://gh-proxy.com/https://github.com/AcademySoftwareFoundation/OpenImageIO/archive/v3.0.9.1.tar.gz"
     ```

2. **`cmake/FindCHOLMOD.cmake` 已修**：vcpkg 的 CHOLMOD target 名是 `SuiteSparse::CHOLMOD_static`（不是 `CHOLMOD::CHOLMOD`），原版 find module 不识别就 fallback 到 raw `find_library(cholmod)` 丢传递依赖。已加 alias 路径，Windows 上无影响。

3. **vcpkg `lapack-reference` 缺 BLAS 传递依赖**：本地 hack 在 `/data/wangzhaolong/Code/Cpp/vcpkg/installed/x64-linux/share/lapack-reference/lapack-targets-release.cmake` 把 `IMPORTED_LOCATION_RELEASE` 改指向 `libopenblas.a`（openblas 已含 LAPACK）。如果 vcpkg 重装 lapack-reference 此 hack 会被覆盖。

4. **链接 group flag**：cmake configure 时必须加 `CMAKE_CXX_STANDARD_LIBRARIES="-Wl,--start-group .../liblapack.a .../libopenblas.a -Wl,--end-group -lgfortran"`，否则 faiss → lapack → blas 单 pass 链接失败。

5. **6dof-lidar 分支的 `src/colmap/exe/sfm.cc` 缺一行 `#include "colmap/exe/gui.h"`**（commit 8838ed64 误删；GUI=ON 时不暴露，GUI=OFF 立刻报 `QApplication 未定义`）。

6. **build 目录命名**：Linux 用 `build-linux-cpu/`（`-DCUDA_ENABLED=OFF`）和 `build-linux-cuda/`（`-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89`，CUDA 走系统 `/usr/local/cuda-12.8`，gcc-13 兼容无需 `CUDAHOSTCXX` hack）。Linux 一律 `-DGUI_ENABLED=OFF -DONNX_ENABLED=OFF`（ONNX 的 ALIKED/onnx_matchers 测试段错误）。

7. **CUDA tree 的 ctest exclusion**（与 CI 一致）：`ctest -E "(feature/sift_test)|(mvs/gpu_mat_test)"`。

完整可复用的环境/cmake 命令模板见 `/tmp/colmap-linux-build-helper.sh`（session 内）。

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

运行单个 C++ 测试（AGENTS.md 里有完整语法，这里是常用速记）：

```bash
# 跑整个测试 binary
ctest --test-dir build -R "^scene/reconstruction_test$" --output-on-failure

# 只跑该 binary 内的某些 GTest case
./build/Release/src/colmap/scene/reconstruction_test --gtest_filter='ReconstructionTest.AddPoint*'
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
| `src/colmap/exe/sfm.cc` 增量 | `RunGlobalMapper` 重写：通过 `--GlobalMapper.use_6dof_pose_priors 1` 路由到 `SixDofPriorGlobalMapper`。**CLI verb 仍是 `global_mapper`**（注册在 `exe/colmap.cc:107`），不是新增子命令。option 入口在 `controllers/option_manager.cc`。 |

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
- CLI 子命令分两层：dispatcher 在 `src/colmap/exe/colmap.cc`（注册 name → entry 函数），实现按领域分散在 `src/colmap/exe/{sfm,mvs,feature,model,...}.cc`。`6dof-lidar` 的 `six_dof_prior_global_mapper` 子命令实现在 `src/colmap/exe/sfm.cc`，option binding 在 `controllers/option_manager.cc`。添加新子命令时三处都要改：dispatcher 注册 + 领域 .cc 实现 + option_manager 注册。
- UI / GUI 相关改动（`src/colmap/ui/`）只能手动目视验证，无法自动化——如果改了 UI 代码，明确告知用户"此项仅构建通过，未跑交互测试"，不要声称"已验证"。
- 跨 session 接手任务时，开头先 `git status` + `git log -5` 确认起点状态，不要假设上个 session 留下的中间状态还在。
- **自研 BA 选项的命名规律**：`BundleAdjustment.*` 是底层 `BundleAdjustmentOptions` 字段；`GlobalMapper.ba_*` 是将同一字段透传到 `SixDofPriorGlobalMapper` 各阶段 BA 的别名，两套入口并存。新增 CLI 选项时两处都要注册（`option_manager.cc`）。
- **改动被广泛 include 的头文件后必须强制重编全部依赖**。Windows + ninja + MSVC 的 `/showIncludes` 对传递依赖不可靠（例如改 `observation_manager.h`，ninja 可能只重编直接 include 它的 `.cc`，不重编间接 include 的单元）。stale `.obj` 会表现为运行时 access violation / 逻辑错乱，极难归因。改 `scene/reconstruction.h`、`sfm/*.h`、`scene/database*.h` 这类广泛头后，要 `ninja -t clean <target>` 或删该模块 `CMakeFiles/*.dir/` 再重编。
