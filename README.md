# FPGA 异步 FIFO 指针跨域检查模块（`fifo_cdc`）

离线数字硬件验证用的可复用 Python 库：对深度 `2^k`（`k=1..5`，深度 2..32）的
经典异步 FIFO（二进制指针 + 跨域两级同步格雷码）做**离散事件级**建模与检查，
输出每条时钟边沿的接受/拒绝结果和完整同步轨迹，并用独立参考 + 抽象队列核对
数据顺序与不越界。

## 运行环境

- Python **3.14.7**（Linux 原生验证；仅使用标准库）
- **无需安装任何第三方依赖**，不需要网络、账号、密钥、硬件、数据库、中间件、
  云 API 或 Docker
- 全部计算离线、确定性；无浮点运算，因此**不存在浮点误差口径问题**

## 准确命令

```bash
# 必要测试（27 个）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.05 秒，远小于 8 秒上限；无 sleep、无预录结果）
python demo.py
```

## 公开接口

```python
from fifo_cdc import (
    AsyncFIFOModel,      # 核心模型
    ClockKind, ClockEdge, write_edge, read_edge, nop_edge,
    EdgeRecord, FIFOSnapshot,
    binary_to_gray, gray_to_binary,
    TraceOracle, OracleReport,   # 独立参考核对器
    FIFOError, FIFOConfigurationError, FIFOInputError, FIFOScheduleError,
)
```

### 核心模型 `AsyncFIFOModel(k)`

- `k`：深度指数整数，深度 `= 2**k`，只接受 `1..5`。
- 复位：写/读二进制指针 = 0（位宽 `k+1`），四个同步寄存器 = 0，
  `full=False`、`empty=True`。
- `step(edge) -> EdgeRecord`：提交一条时钟边沿。
- `run(edges) -> tuple[EdgeRecord, ...]`：批量提交；**先全量校验后提交**，
  任一条非法则整体不产生任何状态变更（无部分变更）。
- `snapshot() -> FIFOSnapshot`、`trace`、以及
  `wptr_bin/rptr_bin/wptr_gray/rptr_gray/wq2_rptr/rq2_wptr/full/empty`、
  接受/拒绝计数等只读属性。

边沿事件 `ClockEdge(clock, time, write=False, read=False, data=None)`：

- `time` 为非负整数时刻；时刻只能非递减；同一时刻写域、读域各至多一条边沿，
  两条即"同刻边沿"；
- 写域边沿只能携带写请求，读域边沿只能携带读请求，单边沿不得同时读写，
  读边沿不得携带数据；
- 工厂 `write_edge(time, data)` / `read_edge(time)` / `nop_edge(clock, time)`
  在构造时即校验 `time`。

`EdgeRecord` 记录该边沿**之后**的二进制/格雷指针、两级同步寄存器
（`wq1_rptr/wq2_rptr/rq1_wptr/rq2_wptr`）、本域旗标新值与旧值
（`full/empty/full_before/empty_before`）、尝试与接受标志，以及接受读时的
`read_data`。

### 独立参考 `TraceOracle(k)`

`observe(edge, record)` 喂入与模型相同的边沿和模型记录，`report()` 返回
`OracleReport`。该参考在 `fifo_cdc/reference.py` 中以**另一套独立写法**
（逐位拼格雷码、独立寄存器推演）实现，不调用被测核心算法；同时维护一个与
硬件旗标无关的抽象队列（`collections.deque`），独立核对：

- 接受读不下溢、接受写不上溢（峰值占用 ≤ 深度）；
- 读出数据严格等于已接受写数据序列的前缀（FIFO 顺序）；
- 逐字段比对指针、格雷、同步寄存器、旗标、接受/拒绝标志。

格雷码与二进制互逆的"往返一致"在测试中仅作补充性质。

## 硬件语义（设计取舍）

- 格雷码：`g = b ^ (b >> 1)`，指针在 `k+1` 位上自然回绕。
- **旗标是本域寄存器**：`full` 属写域、`empty` 属读域，复位 `False/True`。
- 每个时钟边沿严格按寄存器非阻塞语义：
  1. 用**边沿前旧旗标**决定本边沿操作是否接受（有效写仅在非满、有效读仅在非空）；
  2. 计算接受后的本地下一指针格雷；
  3. `empty_next = gray(rptr_next) == rq2_wptr_old`；
     `full_next = gray(wptr_next) == (wq2_rptr_old ^ (0b11 << (k-1)))`
     （仅把同步读格雷的最高两位取反，全位相等比较）；
  4. 两级同步链与指针在同一边沿提交：一级采样对方**边沿前**格雷，二级取一级旧值。
- **同刻边沿**：写、读两条边沿共享"该时刻开始时"的旧状态；两边一级同步器
  采样到的都是对方边沿前的指针，故同刻提交顺序不影响任何结果（测试有
  顺序无关性核对）。
- 因此旗标相对另一域真实占用至少滞后两个本域时钟拍，且只会**保守**
  （可能提前判满/判空而拒绝，绝不溢出/漏读）。
- 核心旗标判定**不使用全局队列长度**：核心内不存在队列/占用量结构
  （有源码结构守卫测试）；抽象队列只存在于独立参考中用于验收核对。
- 存储按指针低位寻址，仅用于让接受的读返回对应写数据，不参与任何旗标判定。

## 输入上限与错误语义

- `k ∈ [1,5]`（深度 2..32，指针位宽 2..6）；这是接口边界。
- 数值输入拒绝 `NaN`/`±Infinity`；整数参数（含 `k`、`time`）拒绝 `bool`，
  浮点不静默取整；`time` 必须为非负整数。
- 非法输入抛可定位异常：
  - `FIFOConfigurationError`：`k` 越界等构造错误；
  - `FIFOInputError`：类型错误、NaN/Infinity、跨域携带错误操作、负时刻等，
    信息带序号/字段；
  - `FIFOScheduleError`：时间倒流、同一域同一时刻重复边沿。
- 错误类型共同基类 `FIFOError`。校验先于任何状态修改；`run()` 批处理具有
  全有或全无原子性。
- 本模型不涉及搜索预算，也不存在"预算耗尽 vs 无解"之分；事件序列的接受/
  拒绝是确定的寄存器推演结果。

## 已知限制 / 不宣称的内容

- 这是题面契约下的离散事件功能模型：时间是整数时刻，不建模亚稳态、
  setup/hold、复位释放同步、格雷路径以外的物理时序，也不做形式化证明；
  **不宣称**完整行业标准（任何厂商 FIFO IP / AMBA 等）兼容。
- 同步器只建模固定两级；未参数化同步级数。
- 存储为 Python 字典，仅承载演示/核对数据，容量随已接受但未读出的数据增长，
  受深度上限约束（≤32 个有效项）。
- 写负载 `data` 可为任意可比较 Python 对象（仅做相等比较）。

## 仓库结构

```
fifo_cdc/__init__.py    # 公开接口
fifo_cdc/core.py        # 核心：事件模型、旗标语义、输入/调度校验
fifo_cdc/reference.py   # 独立参考 + 抽象队列核对（不调用核心算法）
tests/test_fifo.py      # 必要测试（手算常量 + 独立参考 + 性质/错误用例）
demo.py                 # 固定输入演示
requirements.txt        # 无第三方依赖（占位说明）
```
