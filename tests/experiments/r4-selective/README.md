# R4 选择性 BF16 实验

默认提交入口仍为 R2。R4 完成八卡组件验证和 50 步、124 帧完整生成：
本地原版参考 SSIM 96.499%，入口端到端 863.173 秒；
R2 对应 97.394% 和 874.538 秒。R4 只缩短 1.30%，相似度降低
0.895 个百分点，因此未选为正式后端。这是单个公开样例的自测，官方验收未完成。

`h3_attention_matrix.py` 是实测的选择器。默认每四次 BF16 调用中一次走
原始 BF16 Q/K 路径，其余沿用 R2 融合路径；FP16/FP32 保持原路径。
两个支持文件沿用 `../r3-runtime/h3_attention_native_qk.py` 和
`../r3-runtime/h3_nine_runtime.py`，数值计算仍经过 NineToothed 编译链。

`prepare_experiment.py --output /new/path` 只复制并校验实验源码，不生成视频。
生成后仍须按主工程的整机八卡入口、固定权重和完整生成配置运行。
实验源码身份与完整运行记录见 `evidence/r4-full-video-experiment-summary.json`。
