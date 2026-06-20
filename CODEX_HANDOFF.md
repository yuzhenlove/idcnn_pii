# NAS-IDCNN Codex 项目交接说明

## 项目目标

本项目针对中文 PII 命名实体识别实现 IDCNN，并在此基础上进行多目标神经架构搜索（NAS）。

当前 NAS 的两个优化目标是：

1. 最大化验证集（Dev）实体级 Micro-F1；
2. 最小化长度为 128 时的模型前向 FLOPs。

搜索阶段只能使用 Train 和 Dev。所有搜索结束后，才对最终 Pareto 解集使用 Test；不重新训练 Pareto 模型，直接评估搜索过程中保存的最佳 checkpoint。

## 建议阅读顺序

请先阅读以下文件，不要一开始遍历所有历史文档：

1. `README.md`：项目运行方式、数据和输出位置；
2. `docs/superpowers/specs/2026-06-20-nas-idcnn-search-design.md`：NAS 的完整设计约束；
3. `src/nas_encoding.py`：12 维个体编码、三个实验空间和 Identity 规范化；
4. `src/model_nas_idcnn.py`：可搜索 IDCNN Cell 和 CascadePointer 模型；
5. `src/nsga2.py`：非支配排序、拥挤距离和选择；
6. `src/nas_archive.py`：Archive 更新、权重来源选择和部分权重迁移；
7. `src/nas_train.py`：单个候选的 Train/Dev 训练、BF16 和 FLOPs；
8. `scripts/run_nas_search.py`：初始种群、迭代、并发、缓存、断点续跑和产物；
9. `scripts/evaluate_nas_pareto.py`：搜索结束后的 Test 评估；
10. 对应的 `tests/test_nas_*.py` 和 `tests/test_nsga2.py`。

## 已完成的主要工作

- 已实现 Softmax、CRF、CascadePointer 和 Efficient GlobalPointer 四种 NER 输出头。
- 已实现完整 NAS-IDCNN 搜索空间：
  - 12 维整数编码；
  - Conv、DWConv、SepConv、Identity；
  - 搜索通道数、ratio、Cell 数、kernel 和 dilation；
  - Identity 的 kernel/dilation 会规范化，避免重复架构。
- 已实现三个相互独立的搜索实验：
  - 实验1刻意固定 `Conv + k=3 + dilation=[1,2,4]`；
  - 实验2固定 Conv，搜索其余结构参数；
  - 实验3搜索全部变量。
- 已实现 NSGA-II、Pareto 排序、拥挤距离、交叉、变异和环境选择。
- 已实现容量为 30 的多样性 Archive，以及 Dev F1 Top-3 保留。
- Archive 初始为空；初始种群随机初始化，第一代子代开始使用 Archive 权重迁移。
- 已实现相同 `C/ratio` 架构之间的 Embedding 和首个兼容卷积模块权重迁移。
- 已实现候选结果缓存、规范化去重、搜索状态保存和中断续跑。
- 已实现候选并行子进程、进度、耗时和 ETA 日志。
- NAS 候选在 CUDA 上使用 BF16 autocast，checkpoint 和结果中记录 `precision=bf16`。
- 每个候选固定随机种子 42，搜索选择指标明确为 Dev 实体级 Micro-F1。
- 初始种群不计入代数：种群为 10、运行 5 代时最多评估 `10 + 5×10 = 60` 个候选。

当前 GitHub 代码基线提交为：

```text
386f3c8 use bf16 for nas candidate evaluation
```

## 必须保留的实验约束

- 不要缩减 NAS 的主要实现或搜索空间。
- 完整实验仍计划运行 50 代；先运行 5 代只是流程验证。
- 一代 10 个模型允许全部并行，不改成串行方案。
- 搜索期间禁止读取 Test。
- 搜索结束后不重新训练 Pareto 模型。
- 实验1的 dilation `[1,2,4]` 是刻意设计，不要“修正”为其他配置。
- Archive 更新使用两轮距离筛选：先要求 Hamming 距离 `>1`，容量不足时放宽到 `>0`。
- 不要覆盖或删除现有输出；同一命令应优先利用 `search_state.json` 和候选缓存续跑。

## 当前最重要的下一步

目标机器是 5 张 RTX 5090（每张 32 GB），计划每代 10 个候选按每张卡 2 个进程运行。

`scripts/run_nas_search.py` 已增加多 GPU 候选调度：

1. 使用 `--gpus 0,1,2,3,4` 指定物理 GPU；
2. 使用 `--workers-per-gpu 2` 为每张卡创建两个候选槽位；
3. 10 个候选按 `0,1,2,3,4,0,1,2,3,4` 分配；
4. 每个候选子进程设置独立的 `CUDA_VISIBLE_DEVICES`；
5. `--workers 10` 继续作为全局并发上限。

正式搜索前仍应先做 dry-run 和极小训练验证，再运行实验3的 5 代流程。

## 新机器启动前检查

GitHub 不包含隐私数据和生成产物。确认以下文件已经传到新机器：

```text
data/processed/train.jsonl
data/processed/dev.jsonl
data/processed/test.jsonl
data/processed/char2id.json
data/processed/label2id.json
```

检查环境和五张 GPU：

```bash
cd /datadisk/idcnn_pii
git log -1 --oneline
uv sync
nvidia-smi
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
uv run python -m unittest discover -s tests
```

当前 uv 环境使用 PyTorch `2.11.0+cu128`，与目标机器的 NVIDIA 570.124.04
驱动兼容。无卡模式下 `torch.cuda.is_available()` 为 `False` 属正常现象；
挂载 GPU 后必须重新检查。

## 现有运行命令与产物

只查看初始候选：

```bash
uv run python scripts/run_nas_search.py \
  --experiment 3 \
  --generations 5 \
  --dry-run
```

当前单 GPU/默认 GPU 的 5 代命令：

```bash
uv run python scripts/run_nas_search.py \
  --experiment 3 \
  --generations 5 \
  --population-size 10 \
  --workers 10 \
  --gpus 0,1,2,3,4 \
  --workers-per-gpu 2
```

该命令会从已有 `search_state.json` 和候选缓存继续运行，不覆盖已完成候选。

主要产物位于：

```text
outputs/nas/experiment_N/
├── candidates/          # 每个候选的 best.pt、candidate.json、train.log、process.log
├── generations/         # 初始种群及每代记录
├── search_state.json    # 断点续跑状态
├── archive.json
├── pareto.json
└── pareto.csv
```

三个实验搜索全部结束后，才执行：

```bash
uv run python scripts/evaluate_nas_pareto.py --experiment all
```

该命令生成 `pareto_test.json` 和 `pareto_test.csv`。

## 给后续 Codex 的工作原则

- 先读设计和现有测试，再修改代码；
- 修改范围保持小，不进行无关重构；
- 不假设五卡调度已经存在；
- 每次修改后运行相关单测，再运行完整测试；
- 不主动 push；需要提交时先让用户确认。
