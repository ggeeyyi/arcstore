# ARCStore

ARCStore 是跨代码库复用的统一存储 IO 库。它用一套以**路径字符串**驱动的 API,
统一覆盖三种访问形态:

1. **直连 S3**(`s3://bucket/key`,s5cmd 优先 → aws CLI → boto3)
2. **本地文件系统**(`/local-ssd`、`/efs`、`/tmp`)
3. **FUSE mount 的 S3 桶**(mountpoint-s3,如 `/threed-code`、`/asset`)

核心规则是:**读可以利用挂载,写永远走 S3 API**(本地写 + 推送)。数据集读取
(scatter / wds)默认 `auto`:桶已挂载时读挂载的本地路径(如 `/threed-code/...`),
否则走直连 S3;需要强制直连时设 `read_policy="direct_s3"`。模型权重
(safetensors)默认仍走 direct S3,避免 run:ai streamer 在 mountpoint-s3 上死锁。
由于 mountpoint-s3 拒绝覆盖写且 rename 受限,ARCStore 的写原语在结构上不接触
mount 路径。

## 代码库结构

```text
arcstore/
├── README.md                       # 项目总览、安装方式、常用 API 速查
├── pyproject.toml                  # 包元数据、extras、console scripts、pytest 配置
├── docs/                           # 使用手册与 Koala/AWS 存储约定
├── scripts/                        # Koala 存储 smoke test 与 watchdog
├── src/
│   ├── arcstore.py                 # 顶层公共 API re-export 与包兼容入口
│   ├── location.py                 # s3://、本地路径、FUSE mount 的统一解析
│   ├── io.py / uploads.py / s3cli.py
│   │                               # 通用读写原语、S3 CLI/boto3 传输、后台上传
│   ├── staging.py / workspace.py   # 本地缓存、S3 workdir 到本地 scratch 的映射
│   ├── contents.py / logtee.py     # 路径型第三方 API 适配、训练日志写回
│   ├── data/                       # open_dataset、格式分发、访问策略、DataLoader 包装
│   ├── checkpoint/                 # save_checkpoint/load_checkpoint 统一分发
│   └── torch/                      # torch 扩展:数据后端、DCP、safetensors、manager、runtime
└── tests/
    ├── conftest.py                 # fake s5cmd、环境隔离、S3 测试夹具
    ├── test_*.py                   # 核心 IO / staging / sync / checkpoint 分发单测
    └── test_torch/                 # torch 数据集、checkpoint、safetensors、runtime 单测
```

核心包按依赖层次拆分:顶层 `arcstore` 只依赖轻量运行时;`arcstore.data` 和
`arcstore.checkpoint` 负责统一入口与分发;`arcstore.torch` 承载需要 `[torch]`
extra 的训练功能。测试默认离线运行,通过 fake `s5cmd` 模拟 S3;真实 Koala/FUSE
行为由 `scripts/koala_storage_smoke.py` 覆盖。

## 快速开始

### 安装

arcstore 是独立 pip 包(公开 remote `github.com/ggeeyyi/arcstore`)。核心层只依赖 `boto3`;
`[torch]` 扩展层另需 `s3torchconnector` / `runai-model-streamer[-s3]` / `safetensors` /
`webdataset`(由训练镜像或 `[torch]` extra 提供)。传输优先 `s5cmd`(PATH 上有即 ~1 GiB/s)
→ `aws` CLI → boto3,都没有也不会崩。

**uv 项目(推荐)** —— 声明为 git 依赖,`uv.lock` pin commit,`uv sync` / `uv run` 自动从
GitHub clone+build:

```toml
# pyproject.toml
[project]
dependencies = ["arcstore[torch]"]   # 仅核心层用 "arcstore"

[tool.uv.sources]
# remote 目前只有 main、未打 tag,pin 一个 commit 即可(打 tag 后可改 tag = "vX.Y.Z")
arcstore = { git = "https://github.com/ggeeyyi/arcstore.git", rev = "<commit>" }
```

**pip / 临时环境**:

```bash
pip install "arcstore[torch] @ git+https://github.com/ggeeyyi/arcstore.git@<commit-or-tag>"
```

**开发 arcstore 本身(editable)**:

```bash
uv pip install -e /path/to/arcstore   # 或 export PYTHONPATH=/path/to/arcstore/src:$PYTHONPATH
```

### 最小用法

所有 API 以**路径字符串**驱动:`s3://...` 走直连 S3,本地路径走文件系统(读可选 mount,写永远走 S3 API)。

```python
import arcstore

# 写:本地写完推 S3
arcstore.upload_file("/local/run/metrics.json", "s3://bkt/run/metrics.json")
arcstore.upload_dir_async("/local/run/ckpt", "s3://bkt/run/ckpt")    # 后台推
arcstore.wait_for_uploads()                                          # 退出前 flush(失败在此抛出)

# 读:本地 / mount / S3 一致语义
data = arcstore.read_bytes("s3://bkt/run/metrics.json")

# torch 扩展(需 [torch]):DCP 全量态 / scatter 数据集等底层能力
from arcstore.torch import save_full_state, load_full_state, ScatterPtDataset
```

完整接口与分场景用法见下:[接口层级](#接口层级) · [五类需求速查](#五类需求速查) · [通用读写原语](#通用读写原语)。

## 接口层级

ARCStore 的接口按抽象层级组织。实现功能时优先从高层接口查起;只有高层接口
无法表达需要的控制粒度时,再向下使用底层能力。

| 层级 | 定位 | 主要接口 | 适合场景 |
|---|---|---|---|
| L4 | 训练/运行编排层 | `CheckpointManager`、`RunStorage`、`LogTee`、`ContentsManager` | 完整训练流程、run 目录约定、日志写回、第三方库路径适配 |
| L3 | 统一任务入口层 | `open_dataset`、`build_dataloader`、`save_checkpoint`、`load_checkpoint`、`put`、`open_read/open_write` | dataset、checkpoint、小文件/目录读写的推荐入口 |
| L2 | 专用底层能力层 | `load_ckpt`、`save_full_state/load_full_state`、`load_safetensors_auto`、`ScatterPtDataset`、`build_wds_dataset` | 需要绕过统一分发、直接控制某个后端实现 |
| L1 | 通用 IO 与本地化层 | `upload_file/upload_dir`、`download_file/download_dir`、`stage_to_local`、`ensure_local_file`、`split_workdir`、`list_prefix/glob_files` | 只关心文件/目录传输、本地缓存、S3 workdir 映射 |
| L0 | 路径解析与基础设施 | `resolve`、`Location`、`is_s3`、`split_s3`、`refresh_mounts`、`s3cli.*`、`_env.*` | 调试路径分发、扩展底层传输、库内部基础设施 |

常用查询顺序:

1. 训练中要自动保存、找最新、续训、清理旧 checkpoint:用 `CheckpointManager`。
2. 一次性 checkpoint 读写:用 `save_checkpoint` / `load_checkpoint`,并显式传 `kind`。
3. dataset 读取:先用 `build_dataloader`,需要自己组 loader 时用 `open_dataset`。
4. 第三方库只接受本地路径:用 `ContentsManager`。
5. 日志写回:用 `LogTee` 或 `arcstore-tee`。
6. run 目录组织:用 `RunStorage`。
7. 普通 bytes/file/dir 读写:用 `read_bytes`、`open_read`、`write_bytes`、`open_write`、`put`。
8. 需要本地化或传输细节:用 `stage_to_local`、`ensure_local_file`、`upload_*`、`download_*`。
9. 调试路径/mount/S3 分发:用 `resolve` / `Location`。

## 五类需求速查

### 1. dataset 读取

**统一入口(推荐)**:`open_dataset` 按 `detect_format` 自动分发,集中处理
本地 / mount S3 / 直连 S3 三种访问形态,统一返回 `IterableDataset`(WebDataset
风格 dict 样本 + 单一 `decode(sample)->Any`)。已支持 `scatter` / `wds` / `mds`
(Mosaic StreamingDataset,需 `arcstore[mosaic]`)/ `synthetic`(零存储算力基线);
`jsonl` / `lmdb` 仍为预留接口(调用时给出明确报错与引导)。

```python
import arcstore

# scatter .pt:默认 auto——挂载存在则读挂载本地路径(/threed-code/...),否则 s3torchconnector
ds = arcstore.open_dataset("s3://bkt/latents/", decode=my_decode)
# WebDataset(.tar / shards / glob):同样默认 auto;无挂载时 s3 → pipe:s5cmd cat
ds = arcstore.open_dataset("s3://bkt/shards/shard-{000..099}.tar", decode=my_decode)
# Mosaic MDS / 算力基线(显式 format):
ds = arcstore.open_dataset("s3://bkt/mds/", format="mds", decode=my_decode)
ds = arcstore.open_dataset("", format="synthetic", length=10000)

# 想强制直连 S3(忽略挂载):
ds = arcstore.open_dataset("s3://bkt/latents/", decode=my_decode, read_policy="direct_s3")

# 一行 DataLoader(训练默认值:iterable 自分片、map 式 shuffle/drop_last):
dl = arcstore.build_dataloader("s3://bkt/latents/", decode=my_decode,
                               batch_size=4, num_workers=8)

# 路径分发决策可单独取用:
acc = arcstore.resolve_dataset_access("s3://bkt/latents")   # 默认 auto
# -> DatasetAccess(mode="local"|"mount"|"direct_s3", local_dir=..., s3_uri=...)
```

**底层接口**(`open_dataset` 之下,需细粒度控制时直接用):

```python
from arcstore.torch import ScatterPtDataset, expand_urls, tar_url

fmt = arcstore.detect_format(path)        # "jsonl" | "wds" | "scatter" | "lmdb"
ds = ScatterPtDataset("s3://bkt/latents/", transform=my_decode)
url = tar_url("s3://bkt/shards", "clip-000.tar")
urls = expand_urls("s3://bkt/shards/shard-*.tar")

# jsonl manifest 等小文件:本地化(mount 直接短路,零拷贝)
local = arcstore.ensure_local_file("s3://bkt/meta/manifest.jsonl")
```

### 2. checkpoint 读取

**统一入口(推荐)**:`load_checkpoint(path, kind, **kw)`,`kind` 必须显式给定
(不做自动推断),内部按 `kind` 分发到下面的底层实现。

```python
import arcstore

# kind="blob":单 .pt 对象(先 stage 到本地 NVMe 再 mmap 加载)
blob = arcstore.load_checkpoint(
    "s3://bkt/run/checkpoints/checkpoint_model_010000/model.pt",
    "blob", siblings=("model_ema.pt",))

# kind="safetensors":默认 direct S3 → run:ai streamer;read_policy="mount" → mmap
sd = arcstore.load_checkpoint("s3://bkt/models/Wan2.2-TI2V-5B/", "safetensors")

# kind="full_state":FSDP DCP 全量恢复(需 models/optimizers;要求 .metadata 存在)
step = arcstore.load_checkpoint("s3://.../dcp", "full_state",
                                models=model, optimizers=optimizer,
                                scheduler=sched, ema=ema)

# kind="accelerate"(含 Accelerate 的 DeepSpeed plugin):需 accelerator;s3 来源可给 local_dir 做 stage
step = arcstore.load_checkpoint("s3://bkt/run/checkpoint-12", "accelerate",
                                accelerator=accelerator, local_dir="/local-ssd/stage")
# 原生 DeepSpeed engine 的存取、以及"找最新 + resume 续训"的高层编排,用 CheckpointManager(见“训练编排层”)。
```

**底层接口**(`load_checkpoint` 之下,需细粒度控制时直接用):

```python
from arcstore.torch import load_ckpt, load_safetensors_auto, load_full_state

blob = load_ckpt("s3://.../model.pt", siblings=("model_ema.pt",))
sd = load_safetensors_auto("s3://bkt/models/Wan2.2-TI2V-5B/")
step = load_full_state("s3://.../dcp", model, optimizer, scheduler=sched, ema=ema)
```

### 3. checkpoint 写回

**统一入口(推荐)**:`save_checkpoint(path, kind, **kw)`,`kind` 必须显式给定。
写永远走 S3 API(s3 目的地先本地写再推送,不触碰 mount 路径)。

```python
import arcstore

# kind="blob":单对象(s3 → 临时本地 torch.save 再 upload_file;本地 → 直写)
arcstore.save_checkpoint("s3://bkt/run/ckpt/model.pt", "blob",
                         obj={"step": 100, "model": model.state_dict()})

# kind="full_state":FSDP DCP 全量态(s3torchconnector 直接流式)
arcstore.save_checkpoint("s3://.../dcp", "full_state",
                         models=model, optimizers=optimizer,
                         step=100, scheduler=sched, ema=ema)

# kind="accelerate"(含 Accelerate 的 DeepSpeed plugin):s3 目的地需 local_dir(本地分片目录)
arcstore.save_checkpoint("s3://bkt/run/checkpoint-100", "accelerate",
                         accelerator=accelerator, local_dir="/local-ssd/checkpoint-100")

# kind="safetensors":导出全量(unsharded)权重为 safetensors(DCP 聚合,rank0 写+可选上传);
# ZeRO-3/accelerate 传 state_dict=accelerator.get_state_dict(model)
arcstore.save_checkpoint("s3://bkt/models/run1/", "safetensors", model=model)
```

**底层接口**:

```python
local_dir, s3_dir = arcstore.split_workdir("s3://bkt/user/ckpts/run1")
arcstore.upload_dir_async(f"{local_dir}/checkpoints/checkpoint_model_000100",
                          f"{s3_dir}/checkpoints/checkpoint_model_000100")
arcstore.wait_for_uploads()   # 退出前 flush,失败在此抛出

from arcstore.torch import save_full_state
save_full_state(f"{s3_dir}/checkpoints/.../dcp",
                model, optimizer, step=100, scheduler=sched, ema=ema)
# DeepSpeed/Accelerate 用 save_accelerate_state/load_accelerate_state。
```

### 4. 其他小文件写回

**统一入口(推荐)**:`put` 收编 file/dir × sync/async;`write_bytes` / `open_write`
是 `read_bytes` / `open_read` 的写侧对称接口(内存字节/流直接落 s3,无需手动建临时文件)。

```python
import arcstore

# 内存字节/流直接写(s3 → 临时本地文件 + 上传;本地 → 直写,自动建父目录)
arcstore.write_bytes("s3://bkt/run/metrics.json", json.dumps(metrics).encode())
with arcstore.open_write("s3://bkt/run/config.yaml", "w") as f:
    yaml.safe_dump(cfg, f)        # with 块正常退出才上传;块内异常则不发布半截对象

# put:文件/目录自动判定(recursive=None),async_=True 走后台池
arcstore.put("/local-ssd/run/metrics.json", "s3://bkt/run/metrics.json")
arcstore.put("/local-ssd/run/samples", "s3://bkt/run/samples")            # 目录→递归
arcstore.put("/local-ssd/run/snap.png", "s3://bkt/run/snap/x.png", async_=True)
arcstore.wait_for_uploads()       # 异步退出前 flush
```

**底层接口**(`put` 之下,需固定语义时直接用):

```python
arcstore.upload_file("/local-ssd/run/metrics.json", "s3://bkt/run/metrics.json")        # 同步
arcstore.upload_file_async("/local-ssd/run/snapshot.png", "s3://bkt/run/snap/x.png")    # 异步
arcstore.upload_dir("/local-ssd/run/ckpt", "s3://bkt/run/ckpt")
```

### 5. 训练 log 写回

```bash
# Koala normal 任务优先用 koala submit --s3-log。
# shell 形态用于脚本内高级/兼容场景:
exec > >(arcstore-tee "s3://bkt/run/logs/$(hostname).log" \
         --local "$LOCAL_EXPDIR/logs/run.log" --interval 15) 2>&1

# s3tee 风格分片:
python -u train.py 2>&1 | arcstore-tee s3://bkt/run/logs/rank-0 \
  --chunked --local /local-ssd/run/log-chunks --interval 15
```

```python
# 进程内形态:
tee = arcstore.LogTee("/local-ssd/run/logs/run.log", "s3://bkt/run/logs/run.log").install()
...
tee.close()
```

## 训练编排层

以下为参考内部库 `arc_toolkit` 合并进 arcstore 的训练支撑能力,均位于
`arcstore.torch.*`(需 `[torch]`),核心 IO 层不依赖 torch。

### CheckpointManager —— 找最新 + 续训编排

架在底层 DCP / DeepSpeed 之上的高层管理器:按**模型对象**自动分发(DeepSpeed
引擎走原生 collective + s5cmd 上传,其余 FSDP/DDP/`nn.Module` 走 DCP);写完整性
标记(`_ARC_COMPLETE`),`keep_last` 回收旧 ckpt,`extras` 按 rank 落盘(RNG 等
per-rank 状态不串台),`load_latest` 跳过被抢占截断的半截 ckpt 自动续训。

```python
from arcstore.torch import CheckpointManager, RNGState

ckpt = CheckpointManager(local_dir="/local-ssd/run/ckpts",
                         s3_prefix="s3://bkt/run/ckpts",
                         async_save=True, keep_last=3)
start = ckpt.load_latest(model, optimizer, extras={"rng": RNGState()})  # 0 表示从头
for step in range(start, max_steps):
    ...
    if step % 1000 == 0:
        ckpt.save(step, model, optimizer, extras={"rng": RNGState()})
ckpt.wait()   # 退出前 flush 在途异步保存/上传
```

### runtime —— 分布式协调原语

`get_rank/get_local_rank/get_world_size/is_main/is_local_main/barrier`(无
`torch.distributed` 时回落到 env 并降级为单进程语义),`RNGState`(可 checkpoint
的进程级 RNG),`cache_dir`(`$ARCSTORE_CACHE_DIR` > `/local-ssd/arcstore` > tmp)。

### observability —— EMA / 性能埋点

`EMA`(分组 `_foreach_lerp_`,可 CPU 影子),`PerfTracker`(CUDA event 惰性读取的
吞吐 / IO / 算力分解统计),`StageTimer`,`get_gpu_memory_stats`。

```python
from arcstore.torch import EMA, PerfTracker, get_gpu_memory_stats
```

### 权重导出 / 加载

```python
import arcstore

# 推荐统一 checkpoint 接口:导出 / 读取 safetensors 权重
arcstore.save_checkpoint("s3://bkt/models/run1/", "safetensors", model=model)
state_dict = arcstore.load_checkpoint("s3://bkt/models/run1/", "safetensors")
model.load_state_dict(state_dict, strict=False)
```

`save_safetensors_weights` / `load_pretrained` 仍保留为底层/便利函数;README 主路径统一使用
`save_checkpoint` / `load_checkpoint(kind="safetensors")`。

### ContentsManager —— “给我个本地路径,退出自动上传”

`open_read`/`open_write` 给的是**文件句柄**;当第三方 API 只吃**路径**
(`save_pretrained(dir)` / `cv2.imwrite` 等)时用它:

```python
import arcstore, torch
cm = arcstore.ContentsManager()
with cm.open("s3://bkt/run/model.pt", "wb") as path:   # 写:退出后后台上传
    torch.save(state, path)
with cm.open("s3://bkt/run/model.pt", "rb") as path:   # 读:先下载再给本地路径
    state = torch.load(path)
arcstore.wait_for_uploads()
```

### arcstore-sync —— 代码 ⇄ S3 同步 CLI

koala `--code` 工作流的代码镜像(只同步代码,排除 `.git`/缓存/产物):

```bash
arcstore-sync push      # 本地 → s3://<bucket>/<user>/code/<project>/(镜像,删除前预览确认)
arcstore-sync pull      # S3 → 本地(非破坏性)
arcstore-sync status    # 打印解析出的 code 前缀 + 本地 git 状态
# 配置:ARCSTORE_SYNC_BUCKET / ARCSTORE_CODE_S3 / ARCSTORE_SYNC_PROJECT / ARCSTORE_SYNC_EXCLUDE
```

### accelerate 集成(可选,需 `arcstore[accelerate]`)

```python
from arcstore.torch.integrations.accelerate import init_distributed, build_fsdp_plugin
ctx = init_distributed(backend="fsdp", mixed_precision="bf16")  # -> RunContext
```

## 通用读写原语

读侧(本地 / mount / S3 一致语义):

```python
arcstore.exists(path)            # 本地 / mount / S3 一致语义
arcstore.read_bytes(path)
arcstore.open_read(path, "r")
arcstore.list_prefix(path)       # 直接子项;目录带尾随 "/"
arcstore.glob_files(prefix, ".safetensors")  # 有 mount 返回本地路径,否则 s3:// URI
arcstore.download_file(uri, local)           # 大块传输永远直连 S3(s5cmd 多 part)
arcstore.download_dir(uri, local_dir)
loc = arcstore.resolve(path)     # Location: scheme/bucket/key/read_path()
```

写侧(对称接口,写永远走 S3 API、不碰 mount):

```python
arcstore.write_bytes(path, data)             # read_bytes 的写侧对称
arcstore.open_write(path, "wb")              # open_read 的写侧对称(s3 干净退出才上传)
arcstore.put(local, s3_uri, *, recursive=None, async_=False)  # 收编 upload_file/dir × sync/async
arcstore.upload_file(local, uri) / upload_dir(local, uri)     # 底层固定语义
arcstore.upload_file_async(...) / upload_dir_async(...); arcstore.wait_for_uploads()
```

## 注意事项

- **mount 列表缓存可能 stale**:mountpoint-s3 缓存目录列表,刚写入的对象经
  mount 可能短暂不可见。`arcstore.exists` 已做兜底(mount miss 时回查直连 S3);
  写 API 只返回 s3:// URI,天然规避写后读 mount 的问题。
- `wait_for_uploads()` 是大声失败路径,训练退出前必须调用;atexit 钩子只兜底
  log,不抛异常。
- LMDB 无法从 S3 流式读取;`detect_format` 仅在 bucket 已 mount 时将
  s3 上的 LMDB 判定为合法(FUSE 只读 mmap)。

## 测试

```bash
pip install -e ".[test]"
pytest tests/            # 单测全部离线:fake-s5cmd PATH shim,无需 AWS / GPU
```
