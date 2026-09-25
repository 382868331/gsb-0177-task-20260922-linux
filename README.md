# afifo — FPGA 异步 FIFO 指针跨域检查模块

离散事件的异步 FIFO 指针/旗标硬件模型，用于离线数字硬件验证。仅用 Python
标准库从零实现，核心算法不依赖、不调用任何第三方 FIFO 实现。

读侧另提供 `ReadyValidFIFO`：在**不改动**核心写域协议与跨域同步器的前提
下，给异步 FIFO 外包一个读域 ready/valid 输出寄存器，供下游握手消费。

## 环境与依赖

- Python **3.14.7**（Linux 原生验证；不使用 3.14 之外的语法特性时亦可在
  较早的 3.x 上运行，但以 3.14.7 为准）。
- **无第三方依赖**，无需安装任何包；离线运行，不访问网络、硬件、数据库或
  云服务。

## 准确命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试（41 个用例）
./.venv/bin/python demo.py                            # 固定输入演示，实时计算
```

（仓库自带 `.venv`，即 Python 3.14.7；任何 3.14.7 解释器等价。）

`demo.py` 为固定字面值调度（无随机、无 `sleep`、无预录输出），在本机实测
远小于 90 秒（约 0.04 秒），打印一段正常工作轨迹、一段指针回绕轨迹、一段
ready/valid 输出寄存器轨迹（含连续背压与同拍消费+装入），以及真实触发的
硬件拒绝与契约异常。

## 包结构与公开接口

```
afifo/
  __init__.py
  gray.py        # binary_to_gray / gray_to_binary
  model.py       # AsyncFIFO, Event, EdgeRecord, AsyncFIFOError
  readyvalid.py  # ReadyValidFIFO, RVEvent, RVRecord（读域输出寄存器包装）
tests/
  reference_model.py  # 独立参考（不 import afifo）+ 抽象队列核对
  test_afifo.py
  test_readyvalid.py
demo.py
```

### `AsyncFIFO(k)`

- `k`：深度对数，**仅接受整数 1..5**（`bool` 拒绝），深度 = 2^k（2..32），
  读写指针均为 **k+1 位**（k 位地址 + 1 位回绕位）。
- `run_one(event) -> EdgeRecord`：施加一个时间步并返回完整硬件轨迹。
- `run(iterable_of_events) -> list[EdgeRecord]`：原子批次。整个批次先全部
  校验再施加，任何一步非法则批次完全不生效（无部分变更）；接受生成器；
  单批上限 **100 000** 个事件。
- `reset()`：回到上电态并清空轨迹。
- 属性：`k`、`depth`、`wptr`、`rptr`、`full`、`empty`、`steps`、
  `accepted_writes`、`accepted_reads`、`snapshot()`。

### `Event(time, wclk, rclk, wen=False, ren=False, wdata=None)`

- `time`：整数时间戳，`0 <= time < 2^63`；同一 FIFO 内必须**严格递增**。
- `wclk`/`rclk`：必须为 `bool`，且每步至少一个为 `True`（双停非法）。
- `wen`/`ren`：必须为 `bool`。
- `wdata`：`wen=False` 时必须为 `None`；浮点载荷必须有限（NaN、+/-Inf
  一律拒绝）。其他类型载荷原样存入按地址寻址的存储器。

### `EdgeRecord`（每步轨迹，字段即文档）

`time, wclk, rclk, wen, ren, w_accept, r_accept, wptr_before/after,
rptr_before/after, wgray_before/after, rgray_before/after,
full_before/after, empty_before/after, r2w_before/after,
w2r_before/after, wdata, rdata`。

- `w_accept`/`r_accept` 为 `None` 表示该域本步未时钟；`True/False` 表示
  接受/拒绝。
- `r2w`/`w2r` 为 `(一级同步, 二级同步)` 格雷元组；`before` 是边沿采样的
  旧寄存器值。

### `ReadyValidFIFO(k)` — 读域 ready/valid 输出寄存器包装

在**未修改**的 `AsyncFIFO` 核心外叠加一个读域输出寄存器；写域协议、格雷
指针与两级同步器语义与核心完全一致。

- `run_one(RVEvent)` / `run(iterable)` / `reset()`：与核心同样的契约
  （严格递增时间、原子批次、单批上限 100 000）。
- `RVEvent(time, wclk, rclk, wen=False, ready=False, wdata=None)`：读侧
  只有 `ready`；**出队请求由包装器自己生成**，调用方不再提供 `ren`。
- 属性：`out_valid`、`out_data`、`delivered`（下游交付计数）、
  `accepted_writes`、`accepted_reads`（核心出队=装入次数）、`in_flight`
  （= 接受写 − 交付）、`depth`、`full`、`empty`、`steps`、`inner`、
  `snapshot()`。
- `RVRecord` 每步轨迹：`ren_issued`（包装器实际发出的出队请求）、
  `r_accept`、`delivered`（本拍是否交付）、`out_valid/out_data` 前后值，
  以及完整的核心轨迹 `inner: EdgeRecord`。

输出寄存器语义（读域寄存器，复位清 valid）：

1. 仅当**输出寄存器为空**（`not out_valid`）**或当前值正被 ready 消费**
   时，才发出 `ren = 1`；出队是否发生仍只看核心的**读域旧 empty**。
2. `out_valid=1 且 ready=0`：`ren=0`，读指针冻结，`out_data` 逐位稳定。
3. 读沿上**消费旧值与装入新值可同拍完成**（`delivered` 与 `r_accept`
   同拍为真，valid 保持 1）；交付计数只统计“旧 valid 且 ready”的读沿。
4. 已接受写入但尚未跨域同步到读域的数据**不会被凭真实队列长度预取**：
   同步延迟期间 `ren` 照发但 `r_accept=0`、`out_valid` 保持 0。
5. 出队接受但核心暂空（empty 保守）时，若旧值被消费则 valid 落 0；
   `out_data` 保持旧位（无效期间为 don't-care）。

**容量口径**：输出寄存器**不**计入 `depth`——`depth` 仍是核心深度 2^k；
由于预取，系统在途（接受写 − 交付）最多为 **depth + 1**（核心满 depth
字，输出寄存器再预取 1 字）。`demo.py` 的 ready/valid 段实测到该上限。

### 错误语义

- 所有契约错误抛 `afifo.AsyncFIFOError`（`ValueError` 子类），消息包含
  参数名；批次错误包含 `events[i]` 索引，可定位，不静默纠正。
- 数值输入拒绝 NaN/Infinity（见上）；所有整数形参拒绝 `bool`。
- 核心没有搜索/求解过程，因此**不存在“搜索预算耗尽”这一结果类别**；
  非法输入直接报错，语义上不与任何“无解”混淆。

## 硬件语义（实现的契约）

1. 只有格雷码指针跨时钟域；每域两级同步器。**读写同刻边沿时双方都采样
   旧状态**，包括同步器输入级锁存的对端旧格雷（`s1<=旧对端格雷，
   s2<=旧s1`）。
2. `full` 是写域寄存器，复位 `False`；`empty` 是读域寄存器，复位 `True`。
3. 当前边沿先用**旧旗标**决定是否接受：写接受当且仅当
   `wclk and wen and not full_old`，读接受当且仅当
   `rclk and ren and not empty_old`。
4. 旗标随后按“接受后的本地下一指针格雷”与“**旧**二级同步输出”更新：
   - `empty_next = (gray(rptr_next) == w2r_s2_old)`（仅 rclk 沿刷新）
   - `full_next  = (gray(wptr_next) == (r2w_s2_old ^ 0b11<<(k-1)))`
     （最高两位取反比较；仅 wclk 沿刷新）
5. 停钟（`wclk/rclk=False`）冻结该域全部寄存器（指针、旗标、两级同步器）。
6. 旗标**只由指针格雷与同步寄存器推出**，从不读取存储器或任何全局队列
   长度；存储器按 k 位地址读写，仅用于数据面与读出核对。因此旗标天然
   保守：跨域两级延迟期间可能出现“非满报满 / 非空报空”，造成合法请求被
   拒绝——`demo.py` 的 t=8..15、t=21..22 与 t=2..4 明确展示该延迟。

## 输入上限汇总

| 项目 | 上限 / 范围 |
|---|---|
| k | 整数 1..5（深度 2..32，指针 2..6 位） |
| time | 0 .. 2^63-1，严格递增 |
| 单批事件数 | 100 000 |
| 载荷 | 任意 Python 对象；浮点必须有限；`wen=False` 时必须为 `None` |

这些是接口边界；短演示不穷举上限规模。

## 验证方法

- **手工锚点**：`tests/test_afifo.py` 含手工推导的格雷表（0..15）与一张
  k=1 的完整复位→填满→保守拒绝→排空寄存器波形表（逐字段比对）；
  `tests/test_readyvalid.py` 另含一张手工推导的 k=1 ready/valid 波形段
  （t=1..9：同步延迟不预取→装入→背压保持→同拍消费+装入→valid 落 0），
  逐拍核对 `ren_issued`、`r_accept`、`delivered`、`out_valid/out_data`
  与全部核心寄存器。
- **独立参考差分**：`tests/reference_model.py` 不导入 `afifo`，独立重述
  格雷公式、全部时序与输出寄存器规则（`RefFIFO` / `RefReadyValid`）。
  对 k=1..5、多种时钟频比（含同刻双沿）、停钟窗口、连续背压窗口、长程
  回绕（k=1 与 k=5）逐寄存器差分。
- **双参考序列**：参考模型独立维护“被接受的写入”与“下游交付”两条
  序列，断言交付序列恒为写入序列的前缀，且
  `在途 = 核心队列残留 + 输出寄存器占位 <= depth + 1`。
- **抽象队列核对**：另用 `collections.deque` 仅核对“已接受数据”的 FIFO
  顺序与占用恒在 `[0, 深度]`（防越界），它不参与被测旗标逻辑。
- 格雷 `gray_to_binary(binary_to_gray(v)) == v` 仅作补充往返检查。

## 浮点与误差口径

核心只做**整数位运算**（指针、格雷、旗标），不存在浮点计算结果，因此
**没有数值误差、不使用任何容限**。浮点只可能作为不透明载荷出现，仅做
有限性校验后原样存取，不参与任何比较或算术。

## 设计取舍

- 事件时间用整数轴（不模拟物理亚稳态与具体相位）：同一时刻可给出
  wclk/rclk 双沿，语义统一定义为“双方采样旧寄存器”，确定可复现。
- 格雷跨域只建模“合法但可能滞后一到两拍”的值，不建模亚稳态采样窗口。
- 数据面用按地址索引的列表，纯粹为演示读出数据顺序；旗标逻辑结构上不
  接触它（有测试扫描源码保证）。
- `Event` 使用带校验的普通不可变值类（`__slots__`），因为 Python 3.14
  禁止覆盖 `typing.NamedTuple.__new__`；仍支持属性访问、相等比较与解包。

## 已知限制 / 非目标

- 只实现本题契约的**有限子集**：k=1..5、离散整数时间、单一二进制反射
  格雷与两级同步、单级 ready/valid 输出寄存器；**不宣称兼容任何行业
  标准总线/FIFO 规范**，不建模亚稳态概率、复位释放同步、时序约束或
  真实 FPGA 原语。
- 输出寄存器为**单级**（非两级 skid buffer/pipeline 寄存器）；`ready`
  在读沿采样，无组合前传（fall-through）路径。
- 不做几乎满/几乎空（almost-full/empty）、分组计数、数据校验（ECC/
  parity）、多写多读端口。
- 不提供网站、通用 CLI、看板或持久化；终端命令仅用于测试与演示。
