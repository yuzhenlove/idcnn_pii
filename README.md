# IDCNN PII

中文 PII 命名实体识别实验代码。项目比较三种 IDCNN 输出头：

- `softmax`: IDCNN + Softmax token 分类
- `crf`: IDCNN + CRF 结构化解码
- `egp`: IDCNN + Efficient GlobalPointer span 预测

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

完整实验为 `3 heads x 4 blocks x 3 seeds = 36` 组。

```bash
uv run python scripts/prepare_data.py
uv run python scripts/run_experiments.py --heads softmax crf egp --num_blocks 1 2 3 4 --seeds 42 43 44
uv run python scripts/summarize_results.py
uv run python scripts/plot_results.py
```

默认训练参数来自 `configs.yaml`：

- `epochs=30`
- `batch_size=64`
- `max_len=256`
- `lr=0.001`
- `early_stop_patience=5`

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
uv run python scripts/summarize_results.py
```

只重新画图：

```bash
uv run python scripts/plot_results.py
```

## 输出位置

- 单次实验目录：`outputs/{head}_b{num_blocks}_seed{seed}/`
- 最优模型：`best.pt`
- 指标：`metrics.json`
- 测试集预测：`test_predictions.jsonl`
- 总表：`outputs/summary.csv`
- 均值方差：`outputs/summary_mean_std.csv`
- 图表：`outputs/figures/`
- 训练日志：`logs/{head}_b{num_blocks}_seed{seed}/train.log`
