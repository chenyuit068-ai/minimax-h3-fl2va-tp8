# 从实测源码重建八卡版本

本工程保留原生构建R10实际使用的26份源码改动，以及固定版本的源码、xmake、离线包索引和Boost构建依赖。新建目录后按原始快照加实际改动重建，不依赖编译检查点或旧机器上的目标文件。源码和依赖准备已经在Linux离线检查通过，记录见 `evidence/offline-build-preparation.json`；该检查没有执行DTK编译或设备推理。已完成八卡完整视频验证的是相同推理源码及实际R10原生构建产物，不能将本地准备检查当作新环境设备验收。

## 环境

使用比赛提供的CPython 3.11、Torch `2.9.0+das.opt1.dtk2604`、DTK 26.04和匹配的驱动。不要替换厂商Torch。需要GCC/G++、GNU make、Git、libzstd、Python开发头文件以及pybind11 2.13.6。源码版本和所有离线包SHA-256在 `prerequisites/source-manifest.json` 与构建脚本中固定。原始权重和官方首帧数据不在代码压缩包中。

NineToothed和编译器的确切版本、wheel哈希、补丁顺序与开关见 `platform/COMPILER_REPRODUCTION.md`。HCU Triton必须使用官方 `3.3.0+das.opt1.dtk2604.torch290`，不能以普通上游Triton替代。Transformers 5.19.0与Diffusers 0.41.0的原始wheel哈希分别为 `afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518`、`ea8918b7dfd92ce793b6db689a551b8cd25d52c6c0fcd3ade0706a5fd2a25990`；离线依赖由比赛环境提供。完整运行凭据记录实际导入的环境身份。

## 构建

以下变量都填写为本人实际路径。构建目录必须不存在，需至少保留2GiB本地临时空间；源码、日志和产物均保留。构建命令不申请设备或启动模型。

```bash
export H3_PACKAGE_ROOT=/absolute/path/to/submission
export H3_PYTHON=/absolute/path/to/verified-python-3.11/bin/python
export DTK_ROOT=/opt/dtk-26.04
export H3_BUILD_ROOT=/absolute/path/to/new-native-build

# 可在无卡Linux节点执行，校验并展开固定源码和构建依赖。
# 使用此项检查时选择独立目录，不能随后复用同名目录执行正式构建。
"$H3_PYTHON" "$H3_PACKAGE_ROOT/scripts/rebuild_actual_native.py" \
  --build-root "${H3_BUILD_ROOT}-prepare-check" --prepare-only

# 从新的目录编译安装。若不指定--xmake，会从随包的3.0.5源码构建xmake。
"$H3_PYTHON" "$H3_PACKAGE_ROOT/scripts/rebuild_actual_native.py" \
  --build-root "$H3_BUILD_ROOT" --dtk-root "$DTK_ROOT" --jobs 4

export H3_CORE_ROOT="$H3_BUILD_ROOT/native/source/Projects/InfiniCore"
export H3_LM_ROOT="$H3_BUILD_ROOT/native/source/Projects/InfiniLM"
export INFINI_ROOT="$H3_BUILD_ROOT/infini"
export PYTHONPATH="$H3_LM_ROOT/python:$H3_CORE_ROOT/python:$H3_PACKAGE_ROOT:$H3_PACKAGE_ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

构建器按厂商Torch实际C++ ABI配置全部桥接单元，按顺序构建/安装infinirt、infiniop、infiniccl、infinicore_cpp_api、_infinicore和_infinilm。失败保留日志并停止，不自动重试，不修改受保护的原版入口或比较脚本。实际源码已包含宿主桥接与缓存生命周期改动，不重复应用旧的中间补丁。

## 运行

先按编译器说明准备隔离的厂商编译器和Nine宿主补丁，再创建只读模型链接视图。公开用例使用原模板中的小狗图片和 `A dog falls asleep`。模型和首帧数据从出题方给定位置读取。

```bash
export H3_VENDOR_COMPILER=/absolute/path/to/patched-vendor-compiler
export H3_MODEL_VIEW=/absolute/path/to/new-verified-model-view
export H3_FIRST_FRAME=/absolute/path/to/official-uniform-first-frame.png
export H3_RUN_DIR=/absolute/path/to/new-generation-evidence
"$H3_PYTHON" "$H3_PACKAGE_ROOT/scripts/prepare_model_view.py" \
  --model-path /public/read-only/MiniMax-H3 --output "$H3_MODEL_VIEW"

# 只在已分配的整机八卡作业内执行，申请时覆盖资格验证和完整生成的总耗时。
bash "$H3_PACKAGE_ROOT/scripts/run_validated_sample.sh"
```

本脚本先核验真实八卡环境，再依次完成四项组件资格和一次完整生成。它使用新的本次缓存，匹配实际架构，生成配置固定为50步、124帧、seed42；不提交作业、不循环重跑。视频入口本身为R2实测的原样文件；本次新增的是可移植的构建/启动材料，该新包装脚本尚未在新分配上单独复测。任意输入、输出路径和官方指定配置可通过 `run_eightcard.py fl2va --help` 对应参数传入，不减少评测工作量。

## 原版比较

独立原版参考与参与者生成必须使用相同输入、配置和权重。保留受保护的原版比较脚本，不自行设定官方阈值。

```bash
"$H3_PYTHON" "$H3_LM_ROOT/examples/minimax_h3_compare.py" \
  --tasks fl2va --model-path "$H3_MODEL_VIEW" \
  --reference-fl2va-video /absolute/path/to/independent-reference.mp4 \
  --infinilm-fl2va-video "$H3_RUN_DIR/generated.mp4" \
  --output-dir "$H3_RUN_DIR/comparison"
```

R2实测返回Size PASS、Length PASS、平均SSIM 97.394%。这一自测不能代替官方验收。生成耗时读取真实生成日志，不能把仅比较现有视频的计时当作模型性能。
