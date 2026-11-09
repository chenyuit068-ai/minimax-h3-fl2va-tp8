"""Summarize actual evidence; never infer a full-model or official pass."""
import csv
import json
from pathlib import Path


def main():
    workspace=Path(__file__).resolve().parents[2]
    evidence=workspace/'outputs/native-evidence-current'
    results=evidence/'native-results-20261019-v8'
    correctness=json.loads((results/'native-dispatch.json').read_text())
    benchmark=json.loads((results/'native-benchmark.json').read_text())
    host=json.loads((results/'host-tools.json').read_text())
    versions=json.loads((evidence/'software-versions.json').read_text())
    assert correctness['status']==benchmark['status']==host['status']=='PASS'
    assert not correctness['full_fl2va_tested'] and not benchmark['full_fl2va_tested']
    rows=[]
    for c in benchmark['cases']:
        rows.append(dict(dtype=c['dtype'],tokens=c['tokens'],hidden=c['hidden'],op=c['op'],
                         original_ms=c['baseline']['median_ms'],optimized_ms=c['fused']['median_ms'],
                         speedup=c['speedup'],fusion_applied=c['fusion_applied'],exact=c['exact']))
    package=workspace/'submission'
    with (package/'native-dispatch-results.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    table=['| 精度 | token 数 | 算子 | 原生参考 ms | 优化调度 ms | 比值 | 路径 |',
           '| --- | ---: | --- | ---: | ---: | ---: | --- |']
    for row in rows:
        table.append(f"| {row['dtype'].replace('torch.','')} | {row['tokens']} | {row['op']} | {row['original_ms']:.5f} | {row['optimized_ms']:.5f} | {row['speedup']:.3f} | {'九齿融合' if row['fusion_applied'] else '原算子回退'} |")
    text=f'''# MiniMax-H3 单卡原生接入与优化进展报告

队伍：从容应对；队长：徐晨宇。日期：2026-10-19。

## 结论与完成边界

本轮完成 AdaLN、门控残差的原生调用接入、宿主调度优化、规模回退策略及单卡验证。代码候选包尚未完成正式 FL2VA 视频生成与验收，不能作为“初赛已通过”的证明。用户已确认目前没有收到另行提供的正式首帧、标准视频或相似度通过阈值。

模型注意力候选在 gfx906 上失败，已禁止在 HIP 生产桥接中启用。完整模型的分层卸载没有实现，当前 16GB 单卡不能按参考实现常驻全部模型权重。

## 基线环境与代码位置

- 设备：{correctness['device']}，单卡，16368 MiB 显存，HIP {correctness['hip']}。
- 实际 Python 软件版本：`{json.dumps(versions,ensure_ascii=False)}`。
- 原始 InfiniCore 提交 f32e2fbb2a9b2ca912b5711f4f3836c403196a74；InfiniLM 提交 65deb91f840ebaef752d028fad2f3217ce0950f4。源代码清单见 source-manifest.json。
- 新原生宿主代码位于 submission/native/；设备算子全部在 submission/runtime/，通过九齿生成；补丁及安装器分别在 native/InfiniLM-native-integration.patch、scripts/apply_native_patch.py。
- 原始参考源码和评测比较程序保持不变。原生参考计时是在相同二进制中关闭融合回调后，调用原有 Core 数值算子。

## 优化方案

AdaLN 将原有乘法与两次加法合并为一次九齿启动，显式保留低精度中间结果舍入。门控残差合并乘法与加法。FP32 沿用原实现，避免改变 Core 的定向舍入行为。

首次用 DLPack 建立零复制视图并准备九齿内核。后续由 C++ 验证设备、类型、形状和步长签名，向已编译的九齿内核提供当前输入指针及原生流。缓存不保留输入数值或生成结果，所有调用仍执行计算。

宿主剖析发现视图包装、设备查询和流上下文是小输入的固定成本。直接启动减少该成本；最终在 HYGON 对 AdaLN 设置 2,752,512 元素门槛、残差设置 5,505,024 元素门槛，更小输入执行原算子。该策略来自公开规模的性能测量，不依据测试身份或输入内容。

## 正确性与工程检查

- 最终原生调度测试：{correctness['total_output_checks']} 次输出检查通过，覆盖 FP16/BF16/FP32、非连续调制输入、非整齐尾块及新输入地址。FP16/BF16 全部与独立测试计算逐元素一致；FP32 首次输出与原 Core 路径一致，其余新输入检查有限性。
- DLPack 地址共享、原生句柄删除后的存储生命周期、重复运行时拒绝、回调错误传播均通过检查。
- 独立宿主工具 {len(host['checks'])} 项通过：缺失权重、错误权重索引被拒绝；视频帧数、帧率、尺寸错误时，即使放宽 SSIM 也判失败。合成视频只验证工具，不属于 H3 输出。
- 连续两例缺失资产测试均正确失败，分别保留输入、标准输出、错误输出与审计文件，批处理最终返回非零。
- 本地 NVIDIA 上的缩小 H3 结构测试通过；使用随机合成权重，且两组共用同一九齿注意力后端，只验证点算子接入差异，不能代替 DCU 或正式模型验证。

## 单算子性能

下面为最终 DCU 原生调用结果，包含宿主调度、输出分配及设备工作。预热 10 次后，测量 5 组、每组 100 次，以组均值的中位数报告；每组结束同步设备。输入隐藏维度 5376 与已下载的官方公共配置一致，输入值为随机合成测试数据。

{chr(10).join(table)}

全部计时用例输出正确。标为“原算子回退”的行并未执行九齿融合，其比值只是调度差异；不能宣传为融合收益。以上均为单算子结果，没有完整视频生成的加速数据。

## 失败实验与限制

1. 原生缩小模型初次测试在 DCU 上异常退出；本地原始注意力路径明确报 FlashAttention 未启用。当前构建不能据此宣称 H3 模型可运行。
2. 九齿 dot 注意力候选在 gfx906 的 33-token 样例与 CPU FP64 参考相差约 0.187，超过既定容差；改为向量归约后，单 token 样例仍失败。没有放宽容差来通过，HIP 运行时显式禁止该注意力候选。
3. 宿主调用初版在小输入上明显变慢。保留原始测量，并通过签名调度与原算子回退修正；不同平台和未覆盖规模仍需重新测量。
4. 不支持 GPU Graph capture，未验证多卡、跨设备输入、完整权重加载、正式首帧约束或音视频质量。

## 模型资产与后续验收

官方公共修订 42ed227ee7df40d41602854ae760620d6eb651fe 的模块权重文件约 144.02 GB，其中 Transformer 66.28 GB、文本编码器 66.71 GB。当前只下载配置与索引并记录 SHA256，未下载权重分片。文件大小来自公共仓库元数据，不等于加载实测。

后续需完成 DCU 注意力及显存方案，取得足够资源或实现并验证保持语义的分层加载，再使用正式输入、规定配置和标准视频完成端到端验收。不得量化、训练、降低规定分辨率/帧数/步数或复用生成结果。

## 复现方式

按 README 激活固定比赛环境，保留参考目录并构建参赛副本；执行 scripts/build_native.sh 后，使用 scripts/run_checks.sh 指定全新结果目录。当前 HIP 检查结束返回 2，表示点运算检查完成但模型验证仍被阻塞，详情见 model-blocker.json。

完整执行日志、各版结果及实际执行源码在 evidence/current/native-evidence-current/。归档源文件哈希应与 execution-hashes.json 核对。历史源代码候选保存在 evidence/experiment-source/，用于复核失败实验。

第三方组件用途和许可证见 THIRD_PARTY.md。完整离线工程提供依赖 wheel、xmake 与源码快照；它尚未在第二个干净 Pod 重新执行完整重建。

公开来源：[官方模型仓库](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe)、[官方模型结构说明](https://github.com/MiniMax-AI/MiniMax-H3)。
'''
    (package/'TECHNICAL_REPORT.md').write_text(text,encoding='utf-8')
    print('EVIDENCE_BASED_PROGRESS_REPORT_WRITTEN')


if __name__=='__main__':main()
