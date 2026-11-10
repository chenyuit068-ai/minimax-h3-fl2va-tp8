# 报告数据与证据

| 报告内容 | 包内凭据 | 解读边界 |
| --- | --- | --- |
| R2八卡、50步、124帧与输出规格 | `r2-public-result-summary.json` 的 `generation` | 仅已完成的统一公开样例 |
| R2平均SSIM 97.394%、Size/Length PASS | `unmodified-comparator-result.txt`、R2摘要的`comparison` | 独立原版参考自测，官方验收未完成 |
| R2入口874.538秒、原版600.491秒 | R2摘要的`timings` | 真实生成日志口径，未实现相对原版加速 |
| 包装1033.158秒、平台1198.442秒、9588卡秒 | R2摘要的`generation` | 分别为包装、平台与计费口径，不混作模型耗时 |
| H3去噪733.851秒、视频VAE解码64.683秒 | R2摘要的`timings.native` | 对应本次生成的分段记录 |
| 原生九齿工作区合计峰值4,267,868,160字节 | R2摘要的`runtime_workspace_counter` | 不含权重、VAE工作区和其他张量，非整机显存峰值 |
| R3两次耗时740.365/733.288秒，SSIM95.816%/94.628% | `r3-full-video-experiment-summary.json`、`r3-run-*-comparator-result.txt` | 两次均更快，但相似度下降，未选为默认后端 |
| R4耗时863.173秒，SSIM96.499% | `r4-full-video-experiment-summary.json`、`r4-comparator-result.txt` | 仅缩短1.30%，未选为默认后端 |
| 推理源码与实测R2一致 | `r2-inference-source-identity.json`、`../content-manifest.json` | 38个文件逐一SHA-256绑定 |
| 完整视频可查看 | `../examples/r2-generated.mp4` | 仅示例，正式入口不读取或复用 |

相似度比较复用同一独立原版参考，不重复生成参考视频。
比较已有视频的时间不能作为模型推理性能。
完整原始运行记录保留在本地与个人AFS，包内摘要列出原始数据与视频哈希。
干净环境重建及新产物验证应查看对应的新构建凭据，不能用R10已有结果替代。
