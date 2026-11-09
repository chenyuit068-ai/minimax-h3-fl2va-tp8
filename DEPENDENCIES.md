# 依赖及计算边界

本次实测环境为CPython 3.11.9、Torch 2.9.0+das.opt1.dtk2604、HIP 6.3.26093、DTK 26.04、八张gfx936设备。NineToothed 0.26.0使用官方HCU Triton 3.3.0+das.opt1.dtk2604.torch290。Transformers 5.19.0和Diffusers 0.41.0原始wheel在运行前校验并解压，文件内容逐一核验不变。编译器wheel、宿主源码和补丁哈希见 `platform/compiler-reproduction-manifest.json`，原生构建需要pybind11 2.13.6。

InfiniCore、InfiniLM、NineToothed、ntops的原始源码和各自许可保留在固定源码快照中。NineToothed与ntops的表达用于新增算子，实际设备代码由NineToothed编译链生成。正式参与者路径不调用外部FlashAttention成品内核；独立PyTorch原版参考的依赖单独隔离，不能混同为参与者算子实现。

Torch的设备、流和张量桥接接口用于宿主调度与数据交换；InfiniCore的既有基础能力负责原生模型和通信。新增或重写的Attention、AdaLN、Residual及VAE注意力计算均由NineToothed表达。神经网络视频解码仍执行模型计算，不以媒体封装替代。

PyAV、imageio、imageio-ffmpeg、FFmpeg和NumPy用于媒体读写、输出检查或独立对照。分词与图片预处理沿用参考入口的Transformers处理器和torchvision。Boost 1.74仅为C++构建依赖，随包Debian归档中保留版权文件；xmake源码带原始许可。厂商Torch、DTK、Triton以及原始模型由比赛环境提供，不重复打包。模型权重不训练、不微调、不量化。

`native/h3_dlpack.h`保留官方发行包中的DLPack ABI声明和版权，许可副本见 `licenses/Apache-2.0.txt`。原始第三方源代码的许可和声明必须一同保留。本文件只陈述实际依赖及用途，最终允许的基础能力范围以比赛规则为准。
