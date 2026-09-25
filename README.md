# afifo — FPGA 异步 FIFO 指针跨域检查模块

离散事件的异步 FIFO 指针/旗标硬件模型，用于离线数字硬件验证。读侧可选
ready/valid 输出寄存器包装（`ReadyValidFIFO`）。仅用 Python 标准库从零
实现，核心算法不依赖、不调用任何第三方 FIFO 实现。

## 环境与依赖

- Python **3.14.7**（Linux 原生验证；不使用 3.14 之外的语法特性时亦可在
  较早的 3.x 上运行，但以 3.14.7 为准）。
- **无第三方依赖**，无需安装任何包；离线运行，不访问网络、硬件、数据库或
  云服务。

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（44 个用例）
python demo.py                            # 固定输入演示，实时计算
```

`demo.py` 为固定字面值调度（无随机、无 `sleep`、无预录输出），在本机实测
远小于 90 秒（约 0.3 秒），打印一段正常工作轨迹、一段指针回绕轨迹、一段
ready/valid 输出寄存器轨迹（含背压与容量口径），以及真实触发的硬件拒绝与
契约异常。

## 包结构与公开接口

```
afifo/
  __init__.py
  gray.py     # binary_to_gray / gray_to_binary
  model.py    # AsyncFIFO, Event, EdgeRecord, AsyncFIFOError
  outreg.py   # ReadyValidFIFO, RVEvent, RVRecord（读域输出寄存器包装）
tests/
  reference_model.py  # 独立参考（不 import afifo）+ 抽象队列核对
  test_afifo.py       # 内层 AsyncFIFO 测试
  test_outreg.py      # ready/valid 包装测试
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

在 `AsyncFIFO` 读域外包一级输出寄存器，向下游提供 ready/valid 握手；
写域协议（`wen`/`full`）与两级格雷同步器完全保持内层原样。

- `k`：与 `AsyncFIFO` 相同，整数 1..5。**`depth` 仍只是内层 FIFO 深度
  （2^k），输出寄存器是额外的 1 个槽位，不藏进 depth 参数**（见下
  “容量口径”）。
- `run_one(RVEvent) -> RVRecord`、`run(iterable) -> list[RVRecord]`
  （原子批次，上限同为 100 000）、`reset()`（**清 valid**，即使输出
  寄存器正持有未交付数据；内层一并回到上电态）。
- 属性：`valid`、`out_data`（无效时为保持的旧值，don't-care）、
  `delivered`（下游交付计数）、`inflight`（接受写 − 交付）、
  `accepted_writes`、`accepted_reads`（= 内层出队 = 装入输出寄存器
  次数）、`full`、`empty`、`wptr`、`rptr`、`depth`、`steps`、
  `snapshot()`、`inner`（只读查看内层）。

### `RVEvent(time, wclk, rclk, wen=False, ready=False, wdata=None)`

与 `Event` 相同的校验契约（时间严格递增、时钟/使能为 bool、`wdata`
规则一致），读侧使能由 `ren` 换成下游 `ready` 电平；无 `rclk` 的步
`ready` 被忽略。

### `RVRecord`（每步轨迹）

`time, wclk, rclk, wen, ready, wdata, ren, w_accept, r_accept, load,
delivered, delivered_data, valid_before/after, out_before/after, inner`。

- `ren` 是包装层向内层发出的出队请求（无 `rclk` 时为 `None`）；
  `load` 表示本读沿把内层字装入输出寄存器；`delivered` 表示本读沿
  采样到 **旧 valid 且 ready**（交付计数只计这种情况）；`inner` 是
  内层 `EdgeRecord`。

### 输出寄存器硬件语义

1. 每个读沿发出 `ren = (not valid_old) or ready`：输出寄存器为空、或
   当前值本拍正被 ready 消费时，才请求下一次出队。该请求是否被接受仍
   只由内层**旧 empty** 决定——已接受写入但尚未跨域可见的数据**不会**
   凭真实队列长度被预取。
2. `valid and not ready` 时不发 `ren`：读指针不动，`out_data` 逐位
   稳定，`valid` 保持。
3. 读沿上消费旧值与装入新值可同拍完成：`valid_old and ready` 且内层
   出队被接受时，旧值交付、新值装入，`valid` 保持为 1；内层已空则
   只交付，`valid` 落下。
4. 复位清 `valid`；写域协议与跨域同步器语义与内层完全一致。

### 容量口径（预取导致的统计差异）

包装层预取会使“内层占用”与“下游在途”出现差异，二者不要混用：

- 内层占用 = `accepted_writes − accepted_reads`，恒在 `[0, depth]`；
- 下游在途 = `inflight` = `accepted_writes − delivered`
  = 内层占用 + `valid`，上界为 **depth + 1**。

多出的 1 就是输出寄存器槽位：内层可能已经报空（占用 0），却仍有一个
字停在输出寄存器里等待 ready。`depth` 参数始终只指内层深度，不把这
一级寄存器藏进去。

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
  `tests/test_outreg.py` 另含一张手工推导的 k=1（深度 2）ready/valid
  波形表，逐拍核对 22 拍：预取、持续背压冻结、同拍消费+装入、读写指针
  回绕、保守 full 延迟全覆盖。
- **独立参考差分**：`tests/reference_model.py` 不导入 `afifo`，独立重述
  格雷公式、全部时序与输出寄存器契约（`RefFIFO` / `RefOutReg`）。内层对
  k=1..5、9 种时钟频比（含同刻）、停钟窗口、长程回绕逐寄存器差分；包装层
  对不等频、同刻双沿、连续背压窗口、停钟与回绕逐拍差分。
- **独立双序列核对**：按“被接受的写入”与“下游交付”分别维护参考序列，
  校验交付恒为写入的同序前缀，且剩余序列 = 输出寄存器字（最旧）+ 内层
  队列；另用 `collections.deque` 核对占用恒在 `[0, 深度]`（防越界），
  它们不参与被测旗标逻辑。
- **预取限制**：专测证明写入已接受但未跨域可见时，连续读沿拒绝预取
  （不凭真实队列长度）；复位清 valid（含输出寄存器满时复位）有专测。
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
  格雷与两级同步；**不宣称兼容任何行业标准总线/FIFO 规范**，不建模
  亚稳态概率、复位释放同步、时序约束或真实 FPGA 原语。
- 输出寄存器为**单槽位**（不是多 entry 的 skid buffer）；`ready` 只在
  读沿采样，不建模组合通路（如 ready 同拍反压 ren 的零延迟旁路）。
- 不做几乎满/几乎空（almost-full/empty）、分组计数、数据校验（ECC/
  parity）、多写多读端口。
- 不提供网站、通用 CLI、看板或持久化；终端命令仅用于测试与演示。
