# afifo — FPGA 异步 FIFO 指针跨域检查模块

离散事件的异步 FIFO 指针/旗标硬件模型，用于离线数字硬件验证。仅用 Python
标准库从零实现，核心算法不依赖、不调用任何第三方 FIFO 实现。

## 环境与依赖

- Python **3.14.7**（Linux 原生验证；不使用 3.14 之外的语法特性时亦可在
  较早的 3.x 上运行，但以 3.14.7 为准）。
- **无第三方依赖**，无需安装任何包；离线运行，不访问网络、硬件、数据库或
  云服务。

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（55 个用例）
python demo.py                            # 固定输入演示，实时计算
```

`demo.py` 为固定字面值调度（无随机、无 `sleep`、无预录输出），在本机实测
远小于 8 秒（约 0.04 秒），打印正常工作轨迹、指针回绕轨迹、**读写域单独
复位与世代双向确认轨迹**，以及真实触发的硬件拒绝与契约异常。

## 包结构与公开接口

```
afifo/
  __init__.py
  gray.py     # binary_to_gray / gray_to_binary
  model.py    # AsyncFIFO, Event, EdgeRecord, AsyncFIFOError
tests/
  reference_model.py  # 独立参考（不 import afifo）+ 抽象队列核对
  test_afifo.py       # 原有：格雷、波形、差分、契约、批次原子性
  test_reset.py       # 新增：单域复位/世代双向确认/再次复位/满与回绕/枚举
demo.py
```

### `AsyncFIFO(k)`

- `k`：深度对数，**仅接受整数 1..5**（`bool` 拒绝），深度 = 2^k（2..32），
  读写指针均为 **k+1 位**（k 位地址 + 1 位回绕位）。
- `run_one(event) -> EdgeRecord`：施加一个时间步并返回完整硬件轨迹。
- `run(iterable_of_events) -> list[EdgeRecord]`：原子批次。整个批次先全部
  校验再施加，任何一步非法则批次完全不生效（无部分变更）；接受生成器；
  单批上限 **100 000** 个事件。
- `reset()`：双域同时回到上电态（世代归零、清空轨迹与存储器）。
- 属性：`k`、`depth`、`wptr`、`rptr`、`full`、`empty`、`steps`、
  `accepted_writes`、`accepted_reads`、`w_generation`、`r_generation`、
  `synced`、`snapshot()`。

### `Event(time, wclk, rclk, wen=False, ren=False, wdata=None, *, wrst=False, rrst=False)`

- `time`：整数时间戳，`0 <= time < 2^63`；同一 FIFO 内必须**严格递增**。
- `wclk`/`rclk`：必须为 `bool`；每步至少一个时钟为 `True`，**或**
  `wrst`/`rrst` 之一为 `True`（复位是异步的，不需要时钟沿）。
- `wen`/`ren`/`wrst`/`rrst`：必须为 `bool`；`wrst`/`rrst` 为仅关键字参数。
- `wdata`：`wen=False` 时必须为 `None`；浮点载荷必须有限（NaN、+/-Inf
  一律拒绝）。其他类型载荷原样存入按地址寻址的存储器。
- 旧位置参数调用（`time, wclk, rclk, wen, ren, wdata`）与 6 元组解包
  完全保持兼容。

### `EdgeRecord`（每步轨迹，字段即文档）

原有字段：`time, wclk, rclk, wen, ren, w_accept, r_accept,
wptr_before/after, rptr_before/after, wgray_before/after,
rgray_before/after, full_before/after, empty_before/after, r2w_before/after,
w2r_before/after, wdata, rdata`。

本次在**末尾追加**（旧字段位置不变）：`wrst, rrst, wgen_before/after,
rgen_before/after, wgen2r_before/after, rgen2w_before/after,
rack2w_before/after, wack2r_before/after, wack_reg_before/after,
rack_reg_before/after, synced_before/after`。

- `w_accept`/`r_accept` 为 `None` 表示该域本步未时钟；`True/False` 表示
  接受/拒绝。
- `r2w`/`w2r` 为 `(一级同步, 二级同步)` 格雷元组。
- `wgen2r`/`rgen2w` 为世代号跨域的 `(s1, s2)` 元组；`rack2w`/`wack2r`
  为对端应答返回本域的 `(s1, s2)` 元组；`rack_reg`/`wack_reg` 是各域
  本地锁存的"已学到的对端世代"。
- `*_before` 是上一步结束（本步异步复位**之前**）的寄存器值；同步器输入
  级在同刻边沿采样的是复位**之后**的值（先复位后采样）。

### 世代 / 确认状态

- `w_generation` / `r_generation`：单调递增世代号（该域每次复位 +1）。
- `synced`（=`rack2w_s2 == w_generation 且 wack2r_s2 == r_generation`）：
  双向确认是否完成。仅当 `synced` 为真时数据才可能被接受；`False` 期间
  读写请求一律拒绝（旗标保持安全值：写域 `full=False`、读域 `empty=True`）。

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
   保守：跨域两级延迟期间可能出现”非满报满 / 非空报空”，造成合法请求被
   拒绝——`demo.py` 的正常段明确展示该延迟。

### 单域复位与世代双向确认

7. `wrst`/`rrst` 为各域**异步**复位（该步无需对应时钟沿），可单独或同刻
   断言。复位在该步边沿逻辑**之前**处理（先复位后采样）：与复位同刻的
   对端时钟沿，其同步器输入级锁存的是复位后的格雷 0 与新世代。
8. 任一域复位立即：本域指针归零、旗标归安全值（`full=False`/
   `empty=True`）、世代号 +1、本域内全部同步级与本地应答锁存清零；
   **清空存储器**与接受计数；当步双向禁止传递（使能再有效也拒绝）。
9. 重新同步沿用原有两级同步延迟，新增世代与应答两条通道：
   - 新世代经对域两级同步器，对域在 s2 **首次见到**新世代的那一沿：
     把本地指针归零、旗标置安全值，并把本地应答锁存更新为该世代；
   - 应答再经本域两级同步器回到复位发起域；
   - 只有 `rack2w_s2 == w_generation` **且** `wack2r_s2 == r_generation`
     （双向均确认）时 `synced=True`；在此之前两个方向都不接受数据，
     未同步窗口内旗标强制保持安全值，因此同步恢复的**第一个**沿也不会
     空读/溢写。
10. 握手未完成时再次复位只是开启更新的世代；世代号单调递增，管道中旧
    世代值随两级移位自然老化，最终确认收敛到最新世代。

## 输入上限汇总

| 项目 | 上限 / 范围 |
|---|---|
| k | 整数 1..5（深度 2..32，指针 2..6 位） |
| time | 0 .. 2^63-1，严格递增 |
| 单批事件数 | 100 000 |
| 时钟/复位 | 每步至少一个时钟沿或一个复位；`wrst`/`rrst` 异步、仅关键字 |
| 载荷 | 任意 Python 对象；浮点必须有限；`wen=False` 时必须为 `None` |

这些是接口边界；短演示不穷举上限规模。

## 验证方法

- **手工锚点**：`tests/test_afifo.py` 含手工推导的格雷表（0..15）与一张
  k=1 的完整复位→填满→保守拒绝→排空寄存器波形表（逐字段比对）；
  `tests/test_reset.py` 含一张 k=2 写域复位的逐拍世代/应答握手表
  （世代+1、两级传播、学到世代、应答回传、synced 翻转均逐字段断言）。
- **独立参考差分**：`tests/reference_model.py` 不导入 `afifo`，独立重述
  格雷公式、全部时序与世代握手；对 k=1..5、9 种时钟频比（含同刻）、
  停钟窗口、长程回绕（k=1 与 k=5），以及单边复位、同刻双复位、握手中
  再次复位、停钟期间复位逐寄存器差分。
- **小规模全枚举**：k=1 下穷举长度 1..7 的全部 4^1+…+4^7 = 21 844 条
  “写/读/写复位/读复位”动作序列，逐寄存器对拍，并核对复位后读出的
  载荷必为最近一次复位之后接受的数据（旧数据绝不复活）。
- **抽象队列核对**：另用 `collections.deque` 仅核对”已接受数据”的 FIFO
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
  格雷与两级同步、基于世代号的单域复位重新同步；**不宣称兼容任何行业
  标准总线/FIFO 规范**，不建模亚稳态概率、复位释放时序的物理过程、
  时序约束或真实 FPGA 原语（复位同步器的具体电路被抽象为"世代+两级
  同步+双向确认"的寄存器级协议）。
- 不做几乎满/几乎空（almost-full/empty）、分组计数、数据校验（ECC/
  parity）、多写多读端口。
- 不提供网站、通用 CLI、看板或持久化；终端命令仅用于测试与演示。
