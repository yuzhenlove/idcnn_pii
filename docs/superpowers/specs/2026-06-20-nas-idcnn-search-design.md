# NAS-IDCNN 多目标架构搜索设计

## 目标

在现有中文 PII NER 项目中实现完整 NAS-IDCNN 搜索流程，以验证集实体级
Micro-F1 最大化、长度 128 前向 FLOPs 最小化为双目标，使用 NSGA-II 与
Archive 部分权重迁移搜索 Pareto 架构。

搜索期间只使用训练集和验证集。全部搜索完成后，单独对最终 Pareto 解集使用
测试集评估，不重新训练模型。

## 搜索空间

个体采用 12 维整数编码：

```text
[C, ratio, Cell_num,
 op1, k1, d1,
 op2, k2, d2,
 op3, k3, d3]
```

- `C ∈ {64, 128, 256, 512}`
- `ratio ∈ {0.5, 1, 2}`，Cell 内目标通道 `c = C × ratio`
- `Cell_num ∈ {1, 2, 3, 4}`
- `op ∈ {Conv, DWConv, SepConv, Identity}`
- `k ∈ {3, 5, 7}`
- `d ∈ {1, 2, 4, 8}`

当 `op=Identity` 时，规范化为 `k_id=0, d_id=0`。去重、缓存和 Hamming
距离均使用规范化编码。

三个实验互相独立，Archive 不共享：

1. 实验1：三个 op 固定为 Conv，`k=3`，`d=[1,2,4]`，只搜索
   `C/ratio/Cell_num`。
2. 实验2：op 固定为 Conv，搜索 `C/ratio/Cell_num/k/d`。
3. 实验3：搜索全部 12 维变量。

运行顺序为实验3、实验1、实验2。

## NAS 模型

网络结构：

```text
Embedding(C)
→ Dropout(0.35)
→ Conv1d(k=3, C→C)
→ shared Cell repeated Cell_num times
→ Dropout(0.15)
→ CascadePointer
```

Cell 由 `op1 → op2 → op3 → Conv1d(k=3,d=1,out=C)` 构成。非 Identity
操作后使用 ReLU；Identity 不添加激活。Conv 与 SepConv 输出目标通道 `c`；
DWConv 和 Identity 保持输入通道；末尾卷积恢复到 `C`，保证同一 Cell 可以
共享参数并重复调用。CascadePointer 只接收最后一个 Cell 的输出并计算一次
损失。

## NSGA-II

- 种群数量：10
- 初始种群不计入代数
- 试运行：初始10个 + 5代子代，最多60个候选
- 正式运行：初始10个 + 50代子代，最多510个候选
- 所有候选固定训练随机种子42
- 交叉概率0.9，逐基因均匀交叉
- 每个基因以 `1/12` 概率变异为其他合法值
- 使用非支配排序、拥挤距离和二元锦标赛
- 重复的规范化架构直接复用已有 F1、FLOPs 和 checkpoint

## Archive 与权重迁移

搜索开始时 Archive 为空。初始种群全部随机初始化；第一代子代开始使用
Archive 权重迁移。

每代将旧 Archive、当前 Pareto 前沿和 Dev F1 Top-3 合并，按 Pareto rank
升序、拥挤距离降序、F1 降序排序并去重。第一轮只接收与已选个体 Hamming
距离大于1的个体，第二轮放宽为大于0，容量上限30。

新候选只从相同 `C` 和 `ratio` 的 Archive 个体中选权重来源；按 Hamming
距离升序、Dev F1 降序选择。迁移 Embedding，并按 `op1/op2/op3/final_conv`
顺序迁移首个类型、卷积参数、输入输出通道完全兼容的卷积模块。

## 评估与产物

候选训练使用现有固定 Train/Dev 划分、CascadePointer、动态 padding、
最大100 epoch、早停耐心20。搜索目标 F1 明确为 Dev 实体级 Micro-F1。

FLOPs 使用 `batch_size=1, sequence_length=128` 的前向推理统计，包含
Embedding、NAS 编码器和 CascadePointer 前向，不含实体解码和后处理。

同一代全部候选通过独立子进程并行训练。搜索入口定期输出完成进度、ETA 和
实验耗时，并持久化状态以支持中断后继续。

每个实验独立保存：

- 候选编码、Dev F1、FLOPs、训练状态与来源 checkpoint
- 候选最佳权重
- 每代种群与 Archive
- 最终 Pareto CSV/JSON
- Test 评估结果（仅搜索全部完成后显式执行）

## 验证标准

- 编码解码、Identity 规范化和三个实验空间正确。
- NAS Cell 的通道推导、参数共享及四种操作正确。
- NSGA-II 排序、拥挤距离、交叉和变异正确。
- Archive 容量、多样性筛选和部分权重迁移正确。
- 重复架构不重复训练。
- 搜索训练不访问 Test；最终命令可单独评估 Pareto checkpoint。
- 单元测试全部通过，并完成不启动正式训练的搜索流程 smoke test。
