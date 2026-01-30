# 向量点积（Vector Dot Product）- CUDA Kernel 实现

## 问题描述

计算两个 float 向量的点积：

$$
result =  \sum_{i=0}^{n-1} a_{i} \cdot b_{i} 
$$

## 实现方案

| Kernel | 策略 | 同步方式 | 预期性能 | 状态 |
| -------- | -------- | -------- | -------- | -------- |
| Stage1Navive | 单阶段原子归约 | 全局原子操作（高竞争） | 基线（最差） | 已实现 |
| Stage2Naive | 两阶段 + 每个 block 原子操作 | 每个 block 一次原子 | 更好 | 已实现 |
| SharedMem | 两阶段 + 共享内存归约 | 共享内存 + 树形归约 | 更优 | 已实现 |
| WarpShuffle | Warp 级原语 | Warp shuffle 指令 | 最优 | 已实现 |

## Kernel 说明

### Stage1Navive

**策略**：

- 使用 grid-stride loop，适配任意规模的数据
- 每个线程计算自己的部分和
- 所有线程将结果通过原子操作累加到同一个全局 `result` 变量

**代码**：

```cuda
for (int i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
}
atomicAdd(result, sum); // 高竞争
```

**性能表现**：

- 非常慢，因为所有线程都在更新同一个内存地址，GPU 被迫串行化这些原子操作。

## 优化演进过程

### 问题本质

点积是一个**归约（reduction）**问题：多个输入 → 单一输出。

朴素实现面临以下问题：

1. 竞争（Contention）：所有线程争抢同一个输出位置
2. 串行化（Serialization）：原子操作强制顺序执行

**解决思路**：分层归约（Hierarchical Reduction）

**核心思想**：匹配 GPU 的内存层级结构（全局内存 → 共享内存 → 寄存器）

![alt text](kernel.png)

## 代码模式

### Grid-stride loop

所有 kernel 都使用 grid-stride loop 来增强鲁棒性：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = idx; i < N; i += stride) {
    // 处理第 i 个元素 ...
}
```

**优点**：

- 能处理任意规模的 n，与 grid 配置无关
- 当 n 不能被 grid 尺寸整除时，依然具备良好的负载均衡
- 允许限制 grid 大小，但仍可处理数十亿规模的数据

### 两阶段归约（Two-stage reduction）

阶段 1：

- 输入：向量 a[n]，b[n]
- 输出：partialSum[numBlocks] // 每个 block 一个值

阶段 2：

- 输入：partialSum[numBlocks]
- 输出：result // 单一标量

**为什么采用两阶段？**：

- 避免要求所有 block 同时驻留在 GPU 上
- 支持处理超过单次 GPU 可容纳规模的数据集
- 第二阶段开销极小（最多 1024 个元素）

## 构建与运行

### 构建

```bash
# 在项目根目录
mkdir -p build && cd build
cmake ..
cmake --build . --target dot-product
```

### 运行基准测试

```bash
./bin/dot-product
```

输出：

生成包含时间统计信息的 CSV 文件 dot_product_benchmark.csv

### 运行测试

```bash
ctest -R dot-product
```

### 结果可视化

```bash
python scripts/analyze_benchmark.py dot_product_benchmark.csv
```

**生成的图表**：

- bandwidth_vs_size.png：不同数据规模下的带宽变化
- kernel_comparison.png：不同 kernel 执行时间对比（含误差棒）
- speedup_comparison.png：相对于基线实现的加速比
- variance_analysis.png：时间稳定性分析（变异系数）

## 性能表现

**测试配置**:

- GPU：NVIDIA RTX 3090（Ampere，计算能力 8.6）
- Block 大小：256 线程
- 最大 grid block 数：1024
- 预热迭代次数：10
- 正式测试次数：5
- 数据类型：float（4 字节）

**带宽表现**:
![alt text](bandwidth.png)

**精度说明**：

- CPU 参考实现使用 double 进行累加，以获得更高精度
- GPU kernel 使用 float，并采用 1e-5 的误差容忍
- 大规模数据集下：
- 浮点累加顺序不同会带来轻微差异
- 原子操作的顺序是非确定性的
- 对大多数使用场景而言，相对误差仍然可接受（< 1e-5）
