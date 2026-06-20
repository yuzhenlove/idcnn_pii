# IDCNN PII

中文 PII 命名实体识别实验代码。项目支持四种 IDCNN 输出头：

- `softmax`: IDCNN + Softmax token 分类
- `crf`: IDCNN + CRF 结构化解码
- `cascade`: IDCNN + 类型起点分类 + 64 字符条件终点指针
- `egp`: IDCNN + Efficient GlobalPointer span 预测

IDCNN 编码器保留中文字符级输入：每个字符使用 100 维可训练 embedding，不加载作者的英文预训练词向量，也不使用 shape 特征。初始卷积将 100 维字符特征映射为 300 维隐藏特征，重复 block 的 dilation 配置为 `[1, 2, 1]`。训练时对每个 block 使用同一个输出头计算损失并求和；Softmax 和 CRF 使用 expectation-linear dropout regularization，Cascade 和 EGP 不使用该面向 token logits 的未归一化正则；预测时只使用最后一个 block 的输出。

## 仓库缺少什么

GitHub 版本只保留代码和目录结构，不上传实验数据和生成产物。

缺少的本地内容包括：

- `data/raw/pii.json`: 原始 PII 数据，涉及隐私，必须自行放入
- `data/processed/*`: 预处理后的 train/dev/test 和词表，可由脚本重新生成
- `logs/*`: 训练日志
- `outputs/*`: 模型权重、预测结果、指标汇总和图片
- `.venv/`: 本地 Python 虚拟环境

目录中的 `.gitkeep` 只是占位文件，用来让 GitHub 保留目录结构。

## 快速配置环境

需要 Python `>=3.10` 和 `uv`。

```bash
cd idcnn_pii
uv sync
```

如果没有安装 `uv`：

```bash
python -m pip install uv
uv sync
```

## 准备数据

将原始数据放到：

```text
data/raw/pii.json
```

格式为 JSON list，每条样本至少包含：

```json
{
  "text": "张三的手机号是13800000000",
  "entities": [
    {"text": "张三", "type": "NAME", "start": 0, "end": 2}
  ]
}
```

然后执行：

```bash
uv run python scripts/prepare_data.py
```

该步骤会生成：

- `data/processed/train.jsonl`
- `data/processed/dev.jsonl`
- `data/processed/test.jsonl`
- `data/processed/char2id.json`
- `data/processed/label2id.json`

无效 span 会记录到 `logs/invalid_spans.jsonl`。

## 快速跑完全部实验

原三头基线实验为 `3 heads x 4 blocks x 3 seeds = 36` 组。

```bash
uv run python scripts/prepare_data.py
uv run python scripts/run_experiments.py --heads softmax crf egp --num_blocks 1 2 3 4 --seeds 42 43 44
uv run python scripts/summarize_results.py
uv run python scripts/plot_results.py
```

Cascade Pointer 可单独试跑：

```bash
uv run python src/train.py --head cascade --num_blocks 4 --seed 42
```

默认训练参数来自 `configs.yaml`：

- `epochs=100`
- `batch_size=128`
- `max_len=512`
- `lr=0.0005`
- `Adam beta1=0.9, beta2=0.9, epsilon=1e-6`
- `input_dropout=0.35, hidden_dropout=0.15`
- `token_dropout=0.15`
- `drop_penalty=1e-4`
- `grad_clip=5.0`
- `early_stop_patience=100`

如果机器较慢，可以先跑 CPU smoke test：

```bash
uv run python scripts/prepare_data.py
uv run python src/train.py --head softmax --num_blocks 1 --seed 42 --epochs 1 --cpu
```

## 常用命令

单次训练：

```bash
uv run python src/train.py --head crf --num_blocks 2 --seed 42
```

跳过已完成实验是默认行为；强制重跑：

```bash
uv run python scripts/run_experiments.py --heads softmax crf egp --num_blocks 1 2 3 4 --seeds 42 43 44 --force
```

只汇总已有结果：

```bash
uv run python scripts/summarize_results.py \
  --heads softmax crf egp \
  --tag baseline

uv run python scripts/summarize_results.py \
  --heads softmax crf egp cascade \
  --tag all_heads
```

生成对应图表：

```bash
uv run python scripts/plot_results.py \
  --heads softmax crf egp \
  --tag baseline

uv run python scripts/plot_results.py \
  --heads softmax crf egp cascade \
  --tag all_heads
```

## NAS-IDCNN 多目标架构搜索

NAS 搜索使用 CascadePointer，只在 Train/Dev 上训练和选择架构。优化目标是
Dev 实体级 Micro-F1 最大化与长度128前向 FLOPs 最小化。每个候选固定
`seed=42`，在 CUDA/A100 上使用 BF16 混合精度；初始种群不计入代数。

三个实验分别为：

- 实验1：固定 Conv、`k=3`、`d=[1,2,4]`，搜索宽度、ratio 和 Cell 数；
- 实验2：固定 Conv，搜索宽度、ratio、Cell 数、kernel 和 dilation；
- 实验3：搜索全部变量，包括 Conv、DWConv、SepConv 和 Identity。

先检查将要启动的10个初始候选，不执行训练：

```bash
uv run python scripts/run_nas_search.py \
  --experiment 3 \
  --generations 5 \
  --dry-run
```

先跑实验3的5代流程验证。该命令评估10个初始个体，再运行5代，每代10个：

```bash
uv run python scripts/run_nas_search.py \
  --experiment 3 \
  --generations 5 \
  --population-size 10 \
  --workers 10 \
  --gpus 0,1,2,3,4 \
  --workers-per-gpu 2
```

正式按 `3 → 1 → 2` 顺序运行三个实验，每个实验50代：

```bash
uv run python scripts/run_nas_search.py \
  --experiment all \
  --generations 50 \
  --population-size 10 \
  --workers 10 \
  --gpus 0,1,2,3,4 \
  --workers-per-gpu 2
```

搜索状态保存在 `outputs/nas/experiment_{1,2,3}/search_state.json`。相同命令
会从已完成代数继续运行；已评估的规范化重复架构直接复用结果。

多 GPU 模式会为每张物理 GPU 创建指定数量的候选槽位，并为每个候选子进程
单独设置 `CUDA_VISIBLE_DEVICES`。调度器优先参考相同 `C/ratio` 的历史候选
训练耗时；历史不足时根据通道、Cell 数和卷积结构估算成本，再将较慢和较快
候选组合分配，使各 GPU 的预计总负载尽量接近。每个子进程内部只看到逻辑
`cuda:0`，`--workers` 仍控制全局并发上限。

全部搜索完成后，才使用 Test 集评估最终 Pareto 解集：

```bash
uv run python scripts/evaluate_nas_pareto.py --experiment all
```

主要输出：

- 候选结果：`outputs/nas/experiment_N/candidates/`
- 每代记录：`outputs/nas/experiment_N/generations/`
- Archive：`outputs/nas/experiment_N/archive.json`
- Dev Pareto：`outputs/nas/experiment_N/pareto.csv`
- 最终 Test：`outputs/nas/experiment_N/pareto_test.csv`

## 输出位置

- 单次实验目录：`outputs/{head}_b{num_blocks}_seed{seed}/`
- 最优模型：`best.pt`
- 指标：`metrics.json`
- 测试集预测：`test_predictions.jsonl`
- 三头基线报告：`outputs/reports/baseline/`
- 四头完整报告：`outputs/reports/all_heads/`
- 报告内总表：`summary.csv`
- 报告内均值方差：`summary_mean_std.csv`
- 报告内图表：`figures/`
- 训练日志：`logs/{head}_b{num_blocks}_seed{seed}/train.log`
