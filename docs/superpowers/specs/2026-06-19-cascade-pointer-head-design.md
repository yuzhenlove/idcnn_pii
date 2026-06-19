# Cascade Pointer Head 设计

## 目标

为 IDCNN PII 项目增加第四个输出头 `cascade`，用于 flat NER。该头在搜索和
最终训练中保持相同结构，以比 CRF 和 Efficient GlobalPointer 更低的训练成本
提供显式实体边界建模。

现有 Softmax、CRF、EGP 的实现、配置和实验结果不做修改。

## 数据假设

- 数据预处理已经拒绝重叠实体，因此每个 token 最多是一个实体的起点。
- Cascade 最大实体长度固定为 64 个字符。
- 长度不超过 64 的实体覆盖 123,701 / 123,702 个标注。
- 唯一长度为 100 的实体不生成 Cascade 训练标签，但仍保留在评测金标中，
  因而最多贡献一个不可召回的假阴性。

## 模型结构

输入为 IDCNN 输出特征：

```text
features: [batch, length, hidden_size]
```

### 第一阶段：起点类型分类

每个 token 预测：

```text
NONE + 16 个实体类型
```

输出：

```text
start_logits: [batch, length, entity_type_num + 1]
```

标签规则：

- 非实体起点为 0（NONE）。
- 实体起点为 `entity_type_id + 1`。
- padding 为 `-100`，不参与损失。
- 长度超过 64 的实体不标记起点。

### 第二阶段：条件终点指针

起点和终点特征分别经过线性投影：

```text
start_query = Linear(hidden_size, pointer_size)
end_key = Linear(hidden_size, pointer_size)
```

对每个起点 `i`，只与 `[i, i + 63]` 范围内真实存在的 token 做点积：

```text
end_score(i, offset) =
    dot(start_query[i], end_key[i + offset]) / sqrt(pointer_size)
```

输出：

```text
end_logits: [batch, length, 64]
```

超出序列范围或落在 padding 的 offset 被 mask。这里不先构造完整 `L × L`
矩阵，实际复杂度保持为 `O(batch × length × 64 × pointer_size)`。

终点标签：

- 仅金标实体起点参与终点损失。
- 标签为 `entity_end - entity_start - 1`，范围为 `[0, 63]`。
- 其他位置为 `-100`。

## 损失

每个 IDCNN block 使用同一个 Cascade Pointer Head：

```text
loss_block = start_cross_entropy + end_cross_entropy
total_loss = sum(loss_block for each IDCNN block)
```

Cascade 不使用 expectation-linear dropout penalty。该 penalty 来自作者的 token
分类输出实现；Cascade 的复合指针输出不直接套用该正则。

## 解码

1. 对每个有效 token 的 `start_logits` 取 argmax。
2. 如果结果不是 NONE，则得到实体类型。
3. 在该 token 对应的有效 `end_logits` 中取最高分 offset。
4. 输出 `[start, start + offset + 1)` 实体。

不额外做 beam search、阈值调参或冲突消解。保持实现最小，并让第一阶段负责
控制实体候选数量。预测结果沿用项目现有的精确 span/type micro-F1 评测。

## 配置与命令

配置增加：

```yaml
model:
  cascade_max_span_len: 64
  cascade_pointer_size: 64
```

命令：

```bash
uv run python src/train.py --head cascade --num_blocks 4 --seed 42
```

批量脚本同样接受：

```bash
uv run python scripts/run_experiments.py \
  --heads cascade \
  --num_blocks 1 2 3 4 \
  --seeds 42 43 44
```

输出目录遵循已有规则，例如：

```text
outputs/cascade_b4_seed42/
logs/cascade_b4_seed42/
```

## 验证标准

- Cascade 标签正确表示实体类型和终点 offset。
- 长度 64 的实体被保留，长度 65 的实体被忽略。
- 指针只计算 64 宽的带状候选，不产生完整 `L × L` 输出。
- padding 和越界 offset 不可被解码。
- 多 block 损失继续求和，预测只使用最后 block。
- Cascade 的 `drop_penalty` 为 0，Softmax/CRF/EGP 行为不变。
- 完整单元测试通过。
- tiny smoke training 可以完成前向、反向、解码并产生有限 loss。

## 资源约束

当前服务器上有两个正式 EGP 实验占用 GPU。实现和验证期间：

- 单元测试全部在 CPU 上运行。
- tiny smoke training 显式使用 CPU 和小型合成 batch。
- 不启动 CUDA benchmark，不与正式实验争抢显存。
- 只有正式实验结束后，才单独进行 Cascade 的 GPU 速度和显存基准。
