# afifo — FPGA 异步 FIFO 指针跨域检查模块

离散事件的异步 FIFO 指针/旗标硬件模型，用于离线数字硬件验证。仅用 Python
标准库从零实现，核心算法不依赖、不调用任何第三方 FIFO 实现。

## 环境与依赖

- Python **3.14.7**（Linux 原生验证；不使用 3.14 之外的语法特性时亦可在
  较早的 3.x 上运行，但以 3.14.7 为准）。仓库已备好隔离环境，环境内
  Python 命令统一写作 **`./.venv/bin/python`**（下文 `python` 均指它，
  如 `./.venv/bin/python -m unittest ...`）。
- **无第三方依赖**，无需安装任何包；离线运行，不访问网络、硬件、数据库或
  云服务。

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（47 个用例）
python demo.py                            # 固定输入演示，实时计算
```

`demo.py` 为固定字面值调度（无随机、无 `sleep`、无预录输出），在本机实测
远小于 90 秒（约 0.04 秒），打印一段正常工作轨迹、一段指针回绕轨迹、一段
带迟滞的 almost_full/almost_empty 波形（含估计值与标志跳变），以及真实触发
的硬件拒绝、非法阈值与其余契约异常。

## 包结构与公开接口

```
afifo/
  __init__.py
  gray.py     # binary_to_gray / gray_to_binary
  model.py    # AsyncFIFO, Event, EdgeRecord, AsyncFIFOError
tests/
  reference_model.py  # 独立参考（不 import afifo）+ 抽象队列核对
  test_afifo.py
demo.py
```

### `AsyncFIFO(k, *, almost_full_low=None, almost_full_high=None, almost_empty_low=None, almost_empty_high=None)`

- `k`：深度对数，**仅接受整数 1..5**（`bool` 拒绝），深度 = 2^k（2..32），
  读写指针均为 **k+1 位**（k 位地址 + 1 位回绕位）。
- 四个迟滞阈值：均为 **0..depth 的整数**（`bool` 拒绝），且每组必须
  `low < high`（相等或倒挂直接抛 `AsyncFIFOError`）。默认：快满
  `depth-1 / depth`，快空 `0 / 1`；任一项省略则用默认值。
- `run_one(event) -> EdgeRecord`：施加一个时间步并返回完整硬件轨迹。
- `run(iterable_of_events) -> list[EdgeRecord]`：原子批次。整个批次先全部
  校验再施加，任何一步非法则批次完全不生效（无部分变更）；接受生成器；
  单批上限 **100 000** 个事件。
- `reset()`：回到上电态并清空轨迹（`almost_full=False`、
  `almost_empty=True`）。
- 属性：`k`、`depth`、`wptr`、`rptr`、`full`、`empty`、`almost_full`、
  `almost_empty`、`almost_full_low/high`、`almost_empty_low/high`、
  `steps`、`accepted_writes`、`accepted_reads`、`snapshot()`。

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
w2r_before/after, wdata, rdata`，以及本次新增的 almost 接口字段
`w_used_est, r_avail_est, almost_full_before/after,
almost_empty_before/after`。

- `w_accept`/`r_accept` 为 `None` 表示该域本步未时钟；`True/False` 表示
  接受/拒绝。
- `r2w`/`w2r` 为 `(一级同步, 二级同步)` 格雷元组；`before` 是边沿采样的
  旧寄存器值。
- `w_used_est`/`r_avail_est` 为本域边沿上算出的占用/可读估计（模
  2^(k+1)），该域未时钟时为 `None`；新字段**追加在末尾**，旧字段的位置
  顺序与旧调用方式（属性访问、前 25 个位置字段解包）保持不变。

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
5. 停钟（`wclk/rclk=False`）冻结该域全部寄存器（指针、旗标、两级同步器、
   almost 标志）。
6. 旗标**只由指针格雷与同步寄存器推出**，从不读取存储器或任何全局队列
   长度；存储器按 k 位地址读写，仅用于数据面与读出核对。因此旗标天然
   保守：跨域两级延迟期间可能出现“非满报满 / 非空报空”，造成合法请求被
   拒绝——`demo.py` 的 t=8..15、t=21..22 与 t=2..4 明确展示该延迟。
7. **almost_full / almost_empty（迟滞提示）**：`almost_full` 是写域寄存器
   （复位 `False`），`almost_empty` 是读域寄存器（复位 `True`）。每个本域
   时钟沿上，先按“接受后的本地指针”与“**该边沿之前的旧二级同步对端
   指针**”算出模 2^(k+1) 的估计值：
   - 写域占用估计 `w_used = (wptr_next - gray_to_binary(r2w_s2_old)) mod 2^(k+1)`
   - 读域可读估计 `r_avail = (gray_to_binary(w2r_s2_old) - rptr_next) mod 2^(k+1)`

   随后按迟滞更新：`w_used >= almost_full_high` 置位、
   `w_used <= almost_full_low` 清除、带内保持；`r_avail <=
   almost_empty_low` 置位、`r_avail >= almost_empty_high` 清除、带内保持。
   同刻双沿各自采样旧寄存器；停钟的域连标志一起冻结。写域估计可能偏
   大、读域估计可能偏小（保守方向，刻意保留）。**两个标志只是提示**：
   读写是否接受仍完全由 `full`/`empty` 决定。

## 输入上限汇总

| 项目 | 上限 / 范围 |
|---|---|
| k | 整数 1..5（深度 2..32，指针 2..6 位） |
| 迟滞阈值 | 整数 0..depth，每组 low < high；默认快满 depth-1/depth、快空 0/1 |
| time | 0 .. 2^63-1，严格递增 |
| 单批事件数 | 100 000 |
| 载荷 | 任意 Python 对象；浮点必须有限；`wen=False` 时必须为 `None` |

这些是接口边界；短演示不穷举上限规模。

## 验证方法

- **手工锚点**：`tests/test_afifo.py` 含手工推导的格雷表（0..15）、一张
  k=1 的完整复位→填满→保守拒绝→排空寄存器波形表（逐字段比对），以及一张
  k=2（深度 4）读写不等频（写 t%2、读 t%3，含同刻）的 almost 波形表：
  逐拍比对两个估计值、AF/AE 标志、full/empty、同步器与接受/拒绝，并锚定
  迟滞置位/清除时刻（如 t=6 估计=3 置位、t=38/40 估计=2 带内保持、
  t=42 估计=1 清除）。
- **独立参考差分**：`tests/reference_model.py` 不导入 `afifo`，独立重述
  格雷公式、估计与迟滞时序；对 k=1..5、9 种时钟频比（含同刻）、停钟窗口
  （含长停钟冻结新标志）、非默认阈值、长程回绕（k=1 与 k=5）逐寄存器差分。
- **专项覆盖**：阈值相等/倒挂拒绝、阈值端点 0 与 depth、估计值模距离跨
  回绕、占用在迟滞带内来回变化时标志不抖动、almost 标志只作提示不影响
  接受/拒绝。
- **抽象队列核对**：另用 `collections.deque` 仅核对“已接受数据”的 FIFO
  顺序与占用恒在 `[0, 深度]`（防越界），它不参与被测旗标逻辑；手推波形还
  逐拍核对写域估计 ≥ 真实占用、读域估计 ≤ 真实可读。
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
- almost_full/almost_empty 为**迟滞提示**（可编程双阈值），不做分组计数、
  数据校验（ECC/parity）、多写多读端口。
- 不提供网站、通用 CLI、看板或持久化；终端命令仅用于测试与演示。
