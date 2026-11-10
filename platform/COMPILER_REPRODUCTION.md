# gfx936 编译链复现材料

`compiler-reproduction-manifest.json` 记录 Torch、Triton、NineToothed 版本、wheel SHA-256、四个编译器宿主源文件 SHA-256 及补丁 SHA-256。当前清单对应已通过的八卡组件作业 `c602bc31fbeb`；完整 FL2VA 视频与参考对照另行记录。请从比赛提供的离线依赖准备这些精确版本，再应用补丁；模型权重通过原始只读目录提供。

- Torch：2.9.0+das.opt1.dtk2604，CPython 3.11。
- Triton：3.3.0+das.opt1.dtk2604.torch290，官方 HCU 后端，安装在隔离的 vendor compiler 目录。公开官方分发来源为 https://pypi.sourcefind.cn/ 。上游不识别 gfx936 的发行版不能替代。
- NineToothed：0.26.0。

安装官方 Triton wheel 到隔离目录后运行 `python platform/patch_vendor_triton33_native_abi.py /path/to/vendor-compiler-33`。该脚本只修复宿主头文件搜索及 HIP 属性 ABI，保留原文件并生成 host-header-search.patch、host-properties-abi.patch；使用同一 DTK 的 `hipDeviceProp_t` 和 `hipGetDeviceProperties_v2`，不猜结构体偏移、不新增设备计算内核。本目录保留了本次实际生成的这两个补丁。

在本次 InfiniFabric 八卡环境中，继续运行 `python platform/apply_hip_scalar_properties_patch.py --site-packages /path/to/venv/lib/python3.11/site-packages`。设置 `TRITON_HIP_SCALAR_PROPERTIES=1` 后，宿主辅助模块改用带类型的 `hipDeviceGetAttribute` 读取容量、线程束、计算单元、时钟等标量，架构名称从 Torch 的实际设备属性读取。这样避开了本环境中旧版和 v2 属性结构体布局不兼容的问题。该补丁保留关闭开关时的原属性路径，校验修改前后源码和补丁哈希，保存原文件，并支持重复执行。本次 C 源码已用比赛 SDK 编译检查，八卡组件验证使用的源码哈希与清单一致。

安装官方 Nine wheel 后运行 `python platform/apply_nine_generation_host_patch.py --site-packages /path/to/venv/lib/python3.11/site-packages`，给生成的宿主启动参数增加可控的 `enable_fp_fusion=False`。脚本验证原始和修改后源文件哈希，保留原文件，重复执行不重写已匹配的文件。

InfiniFabric 的项目存储和个人 AFS 实测会拒绝子目录到父目录的 `rename`（即使设备号相同），返回 `EXDEV`。运行 `python platform/apply_triton_cache_publication_patch.py --site-packages /path/to/venv/lib/python3.11/site-packages`，将缓存的临时文件放在目标文件的同一目录，再原子替换。补丁保留唯一临时文件、完整写入后发布和失败清理，不改变设备计算。补丁前后源码哈希记录在清单；二进制、文本、文件组、16 路并发写入、注入发布失败的清理检查已在两个存储位置通过，八卡组件作业也已通过。`host-cache-publication.patch` 为对应源码差异。

不同节点读取私有缓存目录可能受文件权限影响。运行入口为每个分配创建自己的临时缓存，只复制已核验的宿主辅助模块归档；结束时删除本次创建的临时目录。原版 Transformers 和 Diffusers 从固定 wheel 校验后解压到节点内存目录，依赖文件内容保持一致，其 wheel 哈希被绑定到组件报告和完整生成入口的源码身份。

运行时必须指定配套环境：

若使用隔离的 `H3_VENDOR_COMPILER` 目录，Triton 的三个补丁应全部应用到该目录；
不要只修改虚拟环境中的另一份 Triton。NineToothed 的补丁则应用到实际安装 Nine
的 Python `purelib`。下面的命令对应这一布局，所有输入 wheel 的哈希仍需与清单一致：

```bash
export H3_VENDOR_COMPILER=/absolute/path/to/verified-vendor-compiler
PY="$H3_PYTHON"
C="$H3_VENDOR_COMPILER"
"$PY" platform/patch_vendor_triton33_native_abi.py "$C"
"$PY" platform/apply_hip_scalar_properties_patch.py --site-packages "$C"
"$PY" platform/apply_triton_cache_publication_patch.py --site-packages "$C"
SITE=$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
"$PY" platform/apply_nine_generation_host_patch.py --site-packages "$SITE"
```

```bash
export PYTHONPATH="$H3_VENDOR_COMPILER${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_HIP_INCLUDE_PATH="$DTK_ROOT/include"
export TRITON_HIP_PROPERTIES_V2=1
export TRITON_HIP_SCALAR_PROPERTIES=1
export LD_LIBRARY_PATH="$DTK_ROOT/llvm/gcvm/lib:$LD_LIBRARY_PATH"
export NINETOOTHED_DISABLE_FP_FUSION=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
```

八卡入口通过预检从全部实际设备的 `gcnArchName` 获取架构，再设置官方后端支持的 `TRITON_FORCE_HIP_ARCH`。共享内存取自带类型的 HIP 标量查询；只有 Torch 暴露对应字段时才执行与 Torch 的精确比较。线程束和计算单元数始终与 Torch 逐卡比较。

基础环境还需要 Torch 原生库、InfiniCore 用户前缀和配套 Python 库的搜索路径；本次 InfiniFabric 环境记录随实验脚本提供。DTK/GCC/驱动应由比赛环境提供，不能仅用软件版本字符串代替真实 DCU 运行验证。

Core 退出生命周期补丁是独立的 InfiniCore-op-cache-teardown.patch，由 scripts/build_streaming.sh 自动应用。无卡登录节点构建时可以设置 `H3_BUILD_DEVICE_IMPORT_CHECK=0`；之后必须在平台分配的实际 DCU 上导入、运行组件并验证正常退出。完整生成要求资格报告匹配当前计算源码、编译器和四个 native 库的实际 SHA-256，不能跳过准入。
