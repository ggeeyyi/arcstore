# arcstore

跨代码库统一存储 IO:一套以**路径字符串**驱动的 API,覆盖三种访问形态——

1. **直连 S3**(`s3://bucket/key`,s5cmd 优先 → aws CLI → boto3)
2. **本地文件系统**(`/local-ssd`、`/efs`、`/tmp`)
3. **FUSE mount 的 S3 桶**(mountpoint-s3,如 `/threed-code`、`/asset`)

核心规则:**训练热路径默认 direct S3,写永远走 S3 API**(本地写 + 推送)。
mountpoint-s3 拒绝覆盖写、rename 受限,所以任何写原语在结构上都不接触
mount 路径。需要兼容老任务时,读路径可显式设置 `read_policy="mount"`。

## 安装

```bash
pip install "arcstore @ git+ssh://<repo-url>"            # 核心层(仅依赖 boto3)
pip install "arcstore[torch] @ git+ssh://<repo-url>"     # + torch 扩展层
```

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `ARCSTORE_S3_MOUNTS` | 未设 | mount 表:`bucket=/mountdir,bucket2=/dir2` |
| `ARCSTORE_USE_MOUNTS` | `1` | mount 翻译总开关 |
| `ARCSTORE_READ_POLICY` | `auto` | 通用读策略:`direct_s3`/`mount`/`auto` |
| `ARCSTORE_DATA_READ_POLICY` | `direct_s3` | scatter 数据集默认读策略 |
| `ARCSTORE_WDS_READ_POLICY` | `direct_s3` | WebDataset 默认读策略 |
| `ARCSTORE_MODEL_READ_POLICY` | `direct_s3` | safetensors 默认读策略 |
| `ARCSTORE_STREAMER_CONCURRENCY` | `32` | run:ai streamer 并发 |
| `ARCSTORE_STREAMER_MEMORY_LIMIT` | `34359738368` | run:ai streamer 有界内存 |
| `ARCSTORE_LOCAL_ROOT` | `/local-ssd/arcstore/workdirs` | `split_workdir` 的本地镜像根 |
| `ARCSTORE_CACHE_DIR` | `/local-ssd/arcstore/cache`(无 `/local-ssd` 时回 `/tmp/arcstore-cache`) | `stage_to_local` 缓存根 |
| `ARCSTORE_CACHE_ENABLE` | `1` | staging 开关 |
| `ARCSTORE_CACHE_BUDGET_GIB` | `200` | 缓存 LRU 预算 |
| `ARCSTORE_STAGE_PREFIXES` | 未设(=全部) | staging 白名单前缀(逗号分隔) |
| `ARCSTORE_S5CMD_WORKERS` | `32` | s5cmd `--numworkers` |
| `ARCSTORE_DCP_STAGE_DIR` | `/local-ssd/arcstore/dcp_load` | DCP S3 读取的节点级预取目录 |
| `ARCSTORE_DCP_SAVE_STAGE_DIR` | `/local-ssd/arcstore/dcp_save`(无 `/local-ssd` 时回 `/tmp/arcstore/dcp_save`) | DCP S3 写入 fallback 的本地暂存目录 |

`AWS_REGION` / `AWS_DEFAULT_REGION` 照常生效(默认 `us-west-2`)。

典型 pod 配置:

```bash
# Koala v1.4+ 新任务推荐不依赖 mount,直接用 s3:// API。
# 如需兼容显式挂载的老任务,再配置 mount 表:
export ARCSTORE_S3_MOUNTS="arcwm-code-us-west-2=/threed-code,arcwm-asset-us-west-2=/asset"
# 训练数据/权重默认 direct S3;如需全局禁用 mount 翻译:
export ARCSTORE_READ_POLICY=direct_s3
```

## 五类需求速查

### 1. dataset 读取

```python
import arcstore
from arcstore.torch import ScatterPtDataset, expand_urls, tar_url

fmt = arcstore.detect_format(path)        # "jsonl" | "wds" | "scatter" | "lmdb"

# scatter .pt:默认 direct S3 用 s3torchconnector;显式 mount 才走本地 glob
ds = ScatterPtDataset("s3://bkt/latents/", transform=my_decode)
ds_mnt = ScatterPtDataset("s3://bkt/latents/", read_policy="mount")

# WebDataset:默认 s3 → pipe:s5cmd cat;显式 mount 才返回普通文件路径
url = tar_url("s3://bkt/shards", "clip-000.tar")
urls = expand_urls("s3://bkt/shards/shard-*.tar")

# jsonl manifest 等小文件:本地化(mount 直接短路,零拷贝)
local = arcstore.ensure_local_file("s3://bkt/meta/manifest.jsonl")
```

### 2. checkpoint 读取

```python
from arcstore.torch import load_ckpt, load_safetensors_auto, load_full_state

blob = load_ckpt("s3://bkt/run/checkpoints/checkpoint_model_010000/model.pt",
                 siblings=("model_ema.pt",))   # 先 stage 到本地 NVMe 再 mmap 加载

sd = load_safetensors_auto("s3://bkt/models/Wan2.2-TI2V-5B/")
# 默认 direct S3 → run:ai streamer;read_policy="mount" → mmap mount 路径

step = load_full_state("s3://.../dcp", model, optimizer, scheduler=sched, ema=ema)
# DCP 全量恢复;S3 读取会先 stage 到 local SSD,并要求 .metadata 存在

hit = arcstore.find_latest_ckpt("s3://bkt/run/checkpoints")  # 续训发现
# -> ("s3://.../checkpoint_model_010000/model.pt", 10000) | None
```

### 3. checkpoint 写回

```python
local_dir, s3_dir = arcstore.split_workdir("s3://bkt/user/ckpts/run1")
# 训练写 local_dir(/local-ssd 镜像),后台推 S3:
arcstore.upload_dir_async(f"{local_dir}/checkpoints/checkpoint_model_000100",
                          f"{s3_dir}/checkpoints/checkpoint_model_000100")
...
arcstore.wait_for_uploads()   # 退出前 flush,失败在此抛出

from arcstore.torch import save_full_state
save_full_state(f"{s3_dir}/checkpoints/checkpoint_model_000100/dcp",
                model, optimizer, step=100, scheduler=sched, ema=ema)
# FSDP DCP:s3torchconnector 直接流式;DeepSpeed/Accelerate 用
# save_accelerate_state/load_accelerate_state。
```

### 4. 其他小文件写回

```python
arcstore.upload_file("/local-ssd/run/metrics.json", "s3://bkt/run/metrics.json")        # 同步
arcstore.upload_file_async("/local-ssd/run/snapshot.png", "s3://bkt/run/snap/x.png")    # 异步
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

## 通用读原语

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
