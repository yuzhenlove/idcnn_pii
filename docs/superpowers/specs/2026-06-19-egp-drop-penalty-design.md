# EGP Dropout Penalty 修复设计

## 问题

当前 `drop_penalty=1e-4` 会统一应用到 Softmax、CRF 和 EGP。该正则按输出
logits 的平方差直接求和。Softmax/CRF 的输出规模为 `batch × length × labels`，
而 EGP 的输出规模为 `batch × entity_types × length × length`，导致 EGP 的
正则项约为监督损失的 28 倍。

实测结果是 IDCNN 特征被压缩为全零，所有有效 span logits 均小于解码阈值 0，
因此 EGP 在 100 个 epoch 后仍不预测任何实体。

## 方案

- Softmax 和 CRF 保持现有 `drop_penalty=1e-4`。
- EGP 构建模型时显式使用 `drop_penalty=0.0`。
- 不修改共享配置值，避免改变已验证的 Softmax/CRF 行为。
- 不引入 EGP 专用可调参数，也不尝试未经验证的 span 数归一化。

## 验证

- 回归测试断言 Softmax/CRF 继续使用配置中的 penalty。
- 回归测试断言 EGP 的 penalty 为 0。
- 运行完整单元测试。
- 运行短 EGP 训练，确认模型不再因正则项主导而立即特征塌缩。

