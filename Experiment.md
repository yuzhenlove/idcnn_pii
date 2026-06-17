# Experiment Summary

检查时间：2026-06-17

## 实验完成度

当前实验主体已经完成。项目已经实现并完成以下组合实验：

- 输出层类型：`softmax`、`crf`、`egp`
- IDCNN block 数量：`num_block=1,2,3,4`
- 随机种子：`42,43,44`
- 总实验数：`3 x 4 x 3 = 36`

每个组合均有 3 个 seed 的结果，因此 `softmax/crf/egp` 三个类型与 `num_block=1,2,3,4` 的 12 个组合均已完成。

## 实现情况

代码中已经实现三类输出层：

- `SoftmaxHead`：`src/heads.py`
- `CRFHead` / `LinearChainCRF`：`src/heads.py`
- `EfficientGlobalPointerHead`：`src/heads.py`

训练入口 `src/train.py` 支持：

```bash
uv run python src/train.py --head softmax --num_blocks 1 --seed 42
uv run python src/train.py --head crf --num_blocks 1 --seed 42
uv run python src/train.py --head egp --num_blocks 1 --seed 42
```

完整批量实验命令：

```bash
uv run python scripts/run_experiments.py --heads softmax crf egp --num_blocks 1 2 3 4 --seeds 42 43 44
```

## 实验结果表

完整明细和均值表：

- `outputs/summary.csv`
- `outputs/summary_mean_std.csv`
- `logs/experiments.csv`

按 3 个 seed 汇总后的核心结果如下：

| head | num_block | runs | dev F1 mean±std | test F1 mean±std |
|---|---:|---:|---:|---:|
| softmax | 1 | 3 | 0.8617±0.0019 | 0.8568±0.0009 |
| softmax | 2 | 3 | 0.8687±0.0029 | 0.8692±0.0009 |
| softmax | 3 | 3 | 0.8487±0.0024 | 0.8500±0.0023 |
| softmax | 4 | 3 | 0.8096±0.0125 | 0.8107±0.0123 |
| crf | 1 | 3 | 0.9202±0.0006 | 0.9154±0.0018 |
| crf | 2 | 3 | 0.9008±0.0043 | 0.8992±0.0031 |
| crf | 3 | 3 | 0.8820±0.0037 | 0.8820±0.0058 |
| crf | 4 | 3 | 0.8520±0.0107 | 0.8493±0.0081 |
| egp | 1 | 3 | 0.9170±0.0005 | 0.9127±0.0012 |
| egp | 2 | 3 | 0.8968±0.0035 | 0.8944±0.0025 |
| egp | 3 | 3 | 0.8706±0.0042 | 0.8705±0.0030 |
| egp | 4 | 3 | 0.8118±0.0129 | 0.8091±0.0158 |

当前最优组合为 `IDCNN + CRF + num_block=1`，Test F1 为 `0.9154±0.0018`。

## 实验结果图

图表由 `scripts/plot_results.py` 生成：

```bash
uv run python scripts/plot_results.py
```

每张图均保存为 `png`、`pdf`、`svg` 三种格式，位于 `outputs/figures/`。其中 `png` 方便 GitHub 和 Word/WPS 预览，`pdf/svg` 适合论文排版。

### Test F1 折线误差图

![Test F1 line errorbar](outputs/figures/test_f1_line_errorbar.png)

该图展示不同输出层随 IDCNN block 数量变化的 Test F1。`CRF` 和 `EGP` 在 `num_block=1` 时表现最好，随着 block 数增加整体下降。

### Test F1 分组柱形图

![Test F1 grouped bar](outputs/figures/test_f1_grouped_bar.png)

该图横向比较每个 `num_block` 下三种输出层的性能。`CRF + num_block=1` 是最优组合，`EGP + num_block=1` 与其接近。

### Dev/Test F1 对比图

![Dev test F1 comparison](outputs/figures/dev_test_f1_comparison.png)

验证集和测试集趋势基本一致，说明实验结论较稳定，不是单一测试集波动造成。

### Test Precision/Recall/F1 三指标图

![Test precision recall F1](outputs/figures/test_precision_recall_f1.png)

该图展示测试集 Precision、Recall 和 F1 的变化。`CRF` 整体最优，`EGP` 在浅层配置下接近 `CRF`。

### Test F1 热力图

![Test F1 heatmap](outputs/figures/test_f1_heatmap.png)

热力图用于快速定位最优配置。颜色最深的位置对应 `CRF + num_block=1`。

### Test F1 排名图

![Test F1 ranking](outputs/figures/test_f1_ranking.png)

该图按 Test F1 对 12 个组合排序。前两名为 `CRF + num_block=1` 和 `EGP + num_block=1`。

### Seed 稳定性图

![Seed variation test F1](outputs/figures/seed_variation_test_f1.png)

该图展示 `seed=42,43,44` 下的结果波动。多数配置波动较小，`num_block=4` 的波动更明显。

## 结果分析

实验结果表明，结构化解码和 span-level 建模明显优于普通 token-level Softmax。`IDCNN + CRF + num_block=1` 取得最佳 Test F1，说明 CRF 对相邻标签转移的建模对 PII 实体识别有明显帮助。`IDCNN + EGP + num_block=1` 表现接近 CRF，可以作为强对比模型。

从 IDCNN block 数量看，更深的 block 没有带来稳定收益。`CRF` 和 `EGP` 均在 `num_block=1` 达到最优，随后随着 block 数增加逐步下降；`Softmax` 在 `num_block=2` 时达到最优，但继续加深后也下降。当前配置下，过多 IDCNN block 可能导致过拟合、优化困难或局部特征过度平滑。

综合效果和稳定性，推荐将 `IDCNN + CRF + num_block=1` 作为最终报告模型，将 `IDCNN + EGP + num_block=1` 作为主要对比模型，将 `IDCNN + Softmax` 作为 baseline。
