# InfiniFabric 八卡推理复现

2026 年 11 月 9 日，八卡作业 `9735e4fd51c3` 正常退出并释放资源。统一小狗图片、提示词 `A dog falls asleep`、seed 42、请求 50 步、124 帧完整运行。输出 768×928、24 fps、H.264，含 AAC 32 kHz 双声道音频；全部视频帧和音频可解码。视频约 5.1667 秒。

以独立原版参考作业 `c37023974619` 的同配置视频为对照，未修改的 `InfiniLM/examples/minimax_h3_compare.py` 返回 Size PASS、Length PASS、平均 SSIM 97.394%。这属于本地原版参考自测，官方相似度验收仍待评测，不能据此保证正式通过或获奖。

本次原生入口端到端 874.538 秒，原版参考 600.491 秒，当前尚无相对原版的端到端加速。计时口径与各阶段耗时见 `evidence/r2-public-result-summary.json`；读取现有视频的比较脚本计时不能作为生成耗时。


## 已实现与待验证

- 新入口 `run_eightcard.py fl2va` 保留原参考入口的条件处理、迭代去噪、视频/音频输出，使用原生InfiniLM常驻张量并行加载。单宿主进程管理8个原生设备工作线程；不要用torchrun启动8份完整生成入口。
- 不使用旧的单卡流式加载，也不设置延迟参数存储。指定权重仍按原始格式及哈希使用；模型本身没有训练、微调或量化。
- Nine工作区按设备分别计容量，签名绑定设备/形状/步长；热启动也进入对应设备与流。TP路径固定保留缓存，前向中超预算直接报错，避免在持锁回调中同步旧流，不能静默回退到未经验证的外部内核。
- Video-VAE解码保留已有Nine注意力适配，并在结束前清理；先保留原生模型对象，再清理其流上工作区，最后释放模型。
- 对Qwen文本层使用原生TP；当前参考视觉编码器在各rank复制，不能把维度可整除检查当成视觉权重已被八分片。
- 新增真实设备测试：RCCL/NCCL八进程归约与聚集、八卡注意力与CPU FP64对照、相同合成权重的H3 TP1/TP8对照、Qwen文本和图像TP1/TP8对照。小型合成模型只用于组件验证，不能替代规定的完整模型生成。
- 资格报告绑定源码、模型Python facade、厂商编译器、native库、实际设备和四项结果哈希。CPU检查不能提供设备资格。

## 1. 平台授权与路径

本题按比赛通知申请整机8卡。先确认资源池、远程执行、8卡并发、运行时限和已认证环境；脚本不会创建作业、申请卡数或自动重试。

个人AFS为4GiB软配额，完整模型不要复制进AFS。共享模型只读，实际H3位置、revision、依赖环境和架构必须先检查。所有下列路径都是需要现场设置的变量，不是已确认的远端目录：

```bash
export H3_PYTHON=/实际已验证环境/bin/python
export H3_LM_ROOT=/本人源码目录/InfiniLM
export H3_CORE_ROOT=/本人源码目录/InfiniCore
export INFINI_ROOT=/本人安装前缀/infini
export H3_PACKAGE_ROOT="$IFAB_AFS_HOME/h3-eightcard/submission"
export H3_RUN_DIR="$IFAB_AFS_HOME/h3-results/本次唯一运行目录"
export H3_MODEL_VIEW="$IFAB_AFS_HOME/h3-model-view-本次校验标识"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
```

项目库、AFS、运行项目复制和临时归档是四种不同容量限制。平台项目中只放启动脚本；依赖和长期源码放AFS，并监测用量。默认4GiB不足时需扩容或复用公共已验证环境，不要盲装另一份Torch。

## 2. CPU登录节点准备

先读取实际Python/Torch/DTK、xmake、Nine及Triton/FlagTree版本、公共环境路径和共享H3目录，不能把平台手册第39节建设示例当作已安装环境。

基线源码版本：InfiniCore f32e2fbb2a9b2ca912b5711f4f3836c403196a74，InfiniLM 65deb91f840ebaef752d028fad2f3217ce0950f4。原始源码离线快照保留在候选工作区同级 `prerequisites/`，其原始清单单独保存。厂商Torch/编译器不在代码包中。

现有 `scripts/build_native.sh` 负责已构建InfiniCore前缀上的InfiniLM桥接。新前缀还需依次构建/安装infinirt、infiniop、infiniccl、infinicore_cpp_api、_infinicore。使用独立参与者源码，不修改受保护baseline。新平台xmake工具链配置和编译器补丁需检查实际版本后再确定，旧gfx936配方不能直接证明新节点兼容。

在CPU阶段执行完整原始文件哈希校验并创建链接视图：

```bash
mkdir -p "$IFAB_AFS_HOME/h3"
"$H3_PYTHON" "$H3_PACKAGE_ROOT/scripts/prepare_model_view.py" \
  --model-path /实际共享H3模型目录 --output "$H3_MODEL_VIEW"
```

这会读取原始文件、逐一计算SHA-256，只在个人空间创建链接与校验清单，不复制大模型。失败就停止并核对模型版本，不替换权重。登录区只准备和编译，不运行DCU负载。

## 3. 整机八卡短测

将变量写入本人启动脚本，在平台八卡Bash或批作业内执行。先做环境probe，再依次做组件资格；每个阶段只执行一次，失败先读取日志，不自动重跑。

```bash
bash "$H3_PACKAGE_ROOT/scripts/ifab_eightcard.sh" probe
bash "$H3_PACKAGE_ROOT/scripts/ifab_eightcard.sh" qualify
```

四项组件每项默认超时600秒；初次JIT耗时需现场测量。资格阶段总上限加上环境启动和清理需纳入平台申请时限。失败子进程组会终止，整机回收仍需查看平台确认。

本次实际环境为 `scnet-dcu` 版本 `524820f26b72`、CPython 3.11.9、Torch 2.9.0+das.opt1.dtk2604、DTK 26.04、八张 gfx936。编译器宿主适配及其开关见 `platform/COMPILER_REPRODUCTION.md`。原版 Transformers 5.19.0 和 Diffusers 0.41.0 从已校验 wheel 临时解压到节点内存，3781 个文件内容逐一核验一致，导入成功；组件报告记录两个 wheel 的原始 SHA-256。

平台项目只保留小型启动脚本，实际源码与安装结果长期存于 AFS。这样避免每次提交重复传送开发 ZIP。宿主辅助模块由 CPU 节点构建并校验，复制到本次分配的私有缓存后使用；每次分配的临时依赖和缓存都在退出时清理。

本平台实测存在设备检测路径差异：一次整机探测已在全部八卡完成 HIP 初始化，随后新的生成进程却在默认 `torch.cuda.is_available()` 检查处返回不可用。当前启动脚本设置 PyTorch 提供的 `PYTORCH_NVML_BASED_CUDA_CHECK=1`，使用设备枚举路径，并在生成前独立检查八张卡的实际张量传输、回读和同步。不能只凭枚举到卡数跳过真实设备检查。此项不改变模型精度或计算算子。

探针成功仅证明环境可访问。全部四项通过才会生成 `components/eight-card-qualification.json`，也仍不表示完整模型或相似度通过。

## 4. 正式统一输入生成

官方模板白色小狗原图由运行者设置到 `H3_FIRST_FRAME`，SHA-256为b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54，提示词 `A dog falls asleep`。官方原图不重复打包进源码。seed42、50步、124帧；不显式覆盖分辨率，由参考pipeline按原图解析。已检查本地固定版本的默认解析为高928、宽768、24fps、124帧约5.1667秒；新节点仍需核对实际默认配置与输出。

```bash
export H3_EIGHT_CARD_QUALIFICATION="$H3_RUN_DIR/components/eight-card-qualification.json"
export H3_FIRST_FRAME=/实际官方小狗原图路径.png
bash "$H3_PACKAGE_ROOT/scripts/ifab_eightcard.sh" generate
"$H3_PYTHON" "$H3_PACKAGE_ROOT/inspect_output.py" \
  --output "$H3_RUN_DIR/generated.mp4" \
  --audit "$H3_RUN_DIR/generated.mp4.run.json" \
  --report "$H3_RUN_DIR/media-spec.json" \
  --width 768 --height 928 --frames 124 --fps 24
```

完整生成调用正常返回后仍需独立媒体与相似度检查。原生8卡都必须记录实际Nine attention调用；输出与审计均要求新路径，不复用历史视频。

## 5. 参考与比较

需在同一统一图片、prompt、seed、帧数、步数和画布配置下完成原始PyTorch参考视频；不能拿旧餐桌视频对比小狗用例。原参考命令使用受保护基线副本，不改评测程序。

完成两份真实视频后，原比较脚本支持只读取视频而不再次生成：

```bash
"$H3_PYTHON" "$H3_LM_ROOT/examples/minimax_h3_compare.py" \
  --tasks fl2va \
  --model-path "$H3_MODEL_VIEW" \
  --reference-fl2va-video /本次真实PyTorch参考视频.mp4 \
  --infinilm-fl2va-video "$H3_RUN_DIR/generated.mp4" \
  --output-dir "$H3_RUN_DIR/comparison"
```

报告填写未修改比较脚本的完整实测输出，不自行设定官方通过阈值。比较视频的媒体规格、完整解码和音频也应检查。R2 新版完整视频和原版参考自测已经完成；最终公开源码 commit、干净重建材料和模板报告仍在整理，官方验收待评测。

## 6. 结果与资源

重要日志通过pipefail+tee写AFS；页面日志有截断限制。正常完成、超时、失败均核对退出码和资源释放，Bash需主动停止并释放；关闭网页不释放卡。保存视频、日志、资格报告、源码哈希和媒体结果后再结束，不能只依赖临时工作区归档。

`hpc-stage-source-files.json`、历史gfx936文档和性能数字属于965984单卡历史。新候选以新源码清单、当前组件报告和新运行审计为准，不沿用旧证据宣称八卡通过。
