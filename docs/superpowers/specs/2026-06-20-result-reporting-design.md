# 实验结果报告设计

## 目标

让同一套汇总和画图脚本能够分别生成：

- Softmax、CRF、EGP 三头基线报告；
- Softmax、CRF、EGP、Cascade 四头完整报告。

报告必须互不覆盖，原始 `metrics.json`、模型、预测和训练日志保持不变。

## 命令接口

两个脚本统一接受：

```text
--heads HEAD [HEAD ...]
--tag TAG
```

支持的 head 为：

```text
softmax crf egp cascade
```

`summarize_results.py` 从 `outputs/*/metrics.json` 中筛选指定 head。
`plot_results.py` 读取同一 tag 的汇总数据，并只绘制指定 head。

## 输出结构

每套报告写入：

```text
outputs/reports/{tag}/
├── summary.csv
├── summary_mean_std.csv
└── figures/
```

日志侧表格写入：

```text
logs/experiments_{tag}.csv
```

图表继续输出 PNG、PDF、SVG。

## 图表适配

- Cascade 使用独立名称和颜色。
- 分组柱状图根据 head 数量动态计算柱宽和偏移。
- Seed 波动图根据 head 数量动态创建子图。
- 热力图行数根据 head 数量变化。
- Y 轴范围由当前数据动态计算，避免裁掉 Softmax 1-block 的低 F1。

## 兼容性与验证

- 不修改训练、数据处理或指标计算代码。
- 三头报告应包含 36 行、12 个均值组合。
- 四头报告应包含 48 行、16 个均值组合。
- 两种报告均应生成 7 类图，每类 3 种格式。
