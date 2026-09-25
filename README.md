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
python -m unittest discover -s tests -v   # 必要测试（53 个用例）
python demo.py                            # 固定输入演示，实时计算
```

`demo.py` 为固定字面值调度（无随机、无 `sleep`、无预录输出），在本机实测
远小于 90 秒（约 0.04 秒），打印一段正常工作轨迹、一段指针回绕轨迹、一段
almost_full/almost_empty 迟滞轨迹，以及真实触发的硬件拒绝、非法阈值与其他
契约异常。

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

### `AsyncFIFO(k, almost_full_low=depth-1, almost_full_high=depth, almost_empty_low=0, almost_empty_high=1)`

- `k`：深度对数，**仅接受整数 1..5**（`bool` 拒绝），深度 = 2^k（2..32），
  读写指针均为 **k+1 位**（k 位地址 + 1 位回绕位）。
- 四个阈值都是 **0 到 depth 的整数**（`bool`、浮点等拒绝），同一组必须
  **`low < high`**（相等或反序构造期即抛错），两组阈值彼此独立配置。
  - `almost_full`（写域寄存器，复位假）：写域占用估计 `w_occ >= high`
    时置位，降到 `<= low` 才清除，介于其间（迟滞带）保持旧值。
  - `almost_empty`（读域寄存器，复位真）：读域可读量估计
    `r_readable <= low` 时置位，升到 `>= high` 才清除，迟滞带保持旧值。
  - 默认值：快满 `(depth-1, depth)`、快空 `(0, 1)`。
  - 两标志**仅作提前提示**，不参与 full/empty 的接受判定，也不读存储器或
    真实队列长度；详见“almost-full/almost-empty 语义”。
- `run_one(event) -> EdgeRecord`：施加一个时间步并返回完整硬件轨迹。
- `run(iterable_of_events) -> list[EdgeRecord]`：原子批次。整个批次先全部
  校验再施加，任何一步非法则批次完全不生效（无部分变更）；接受生成器；
  单批上限 **100 000** 个事件。
- `reset()`：回到上电态并清空轨迹。
- 属性：`k`、`depth`、`wptr`、`rptr`、`full`、`empty`、`almost_full`、
  `almost_empty`、`almost_full_low/high`、`almost_empty_low/high`、`steps`、
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
w2r_before/after, wdata, rdata`（以上 25 个为原字段，位置与名称不变），
其后**追加** almost 提示字段：
`almost_full_before/after, almost_empty_before/after,
w_occ_before/after, r_readable_before/after`。

- `w_occ_*` 是写域占用估计、`r_readable_*` 是读域可读量估计；某域本步未
  时钟时对应估计为 `None`。旧字段与旧的按位置/按名访问继续有效
  （元组长度由 25 增至 33，新增字段都在尾部）。
- `w_accept`/`r_accept` 为 `None` 表示该域本步未时钟；`True/False` 表示
  接受/拒绝。
- `r2w`/`w2r` 为 `(一级同步, 二级同步)` 格雷元组；`before` 是边沿采样的
  旧寄存器值。

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

## almost-full / almost-empty 语义（带迟滞的提前提示）

为让上游能在真正满/空**之前**提前减速，新增两个本域提示寄存器：

- `almost_full` 在写域，复位 `False`；`almost_empty` 在读域，复位 `True`。
- 每个时钟沿，该域只用**本地接受操作后的二进制指针**与**该边沿之前的
  二级同步对端格雷值**（在本域反解回二进制），按 k+1 位取模做估计：

  ```
  w_occ      = (wptr_after  - gray_decode(r2w_s2_old)) mod 2**(k+1)   # 写域
  r_readable = (gray_decode(w2r_s2_old) - rptr_after ) mod 2**(k+1)   # 读域
  ```

  估计值**绝不**从共享存储器或真实队列长度取答案（有源码扫描测试保证
  估计行不含 `_mem`/`len(`/`queue`）。
- 由于同步值滞后，`w_occ` 只可能**偏大**（已读走但写域还没看到）、
  `r_readable` 只可能**偏小**（已写入但读域还没看到）——这是刻意保留的
  保守方向，提示宁可早报。
- 迟滞：每域独立的 `low < high`。AF 在 `w_occ >= high` 置位、`<= low`
  清除；AE 在 `r_readable <= low` 置位、`>= high` 清除；严格落在
  `(low, high)` 区间时保持原标志，因此占用在迟滞带内来回变化**不抖动**。
  两组阈值独立配置，默认 AF=`(depth-1, depth)`、AE=`(0, 1)`。
- 提示**只作提示**：读写是否接受仍完全由旧 `full`/`empty` 决定；
  almost 标志不参与接受逻辑（差分测试验证不同阈值下接受/拒绝逐拍一致）。
- 同刻双沿仍各自采样旧寄存器；停钟的域连同 almost 标志与估计一起冻结
  （该域未时钟时记录中估计为 `None`）。
- 阈值为 0..depth 整数且同组 `low < high`，相等、反序、越界、`bool`/浮点
  等在构造期抛 `AsyncFIFOError`。

## 输入上限汇总

| 项目 | 上限 / 范围 |
|---|---|
| k | 整数 1..5（深度 2..32，指针 2..6 位） |
| almost 阈值 | 四个阈值各为 0..depth 的整数，同组必须 `low < high` |
| time | 0 .. 2^63-1，严格递增 |
| 单批事件数 | 100 000 |
| 载荷 | 任意 Python 对象；浮点必须有限；`wen=False` 时必须为 `None` |

这些是接口边界；短演示不穷举上限规模。

## 验证方法

- **手工锚点**：`tests/test_afifo.py` 含手工推导的格雷表（0..15）、一张
  k=1 的完整复位→填满→保守拒绝→排空寄存器波形表（逐字段比对），以及一张
  k=2（深度4）读写不等频的 25 行 almost 波形表（含使能/接受/双同步器/
  双域估计/AF/AE 逐拍比对，期望值用测试内独立的格雷反查表重算，不调用
  被测标志函数）。
- **独立参考差分**：`tests/reference_model.py` 不导入 `afifo`，独立重述
  格雷公式、两级同步与 almost 估计/迟滞时序；对 k=1..5、多组阈值、多种
  时钟频比（含同刻）、停钟窗口、长程回绕（k=1 与 k=5）逐寄存器差分。
- **迟滞不抖动**：长程运行中断言标志只可能在估计跨过各自置位/清除阈值时
  翻转，落在迟滞带内时严格保持；另含阈值相等拒绝、阈值端点（0、depth）、
  长时间停钟冻结、指针回绕与"提示不改接受判定"的专项用例。
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
  格雷与两级同步；**不宣称兼容任何行业标准总线/FIFO 规范**，不建模
  亚稳态概率、复位释放同步、时序约束或真实 FPGA 原语。
- 已实现带迟滞的 almost-full/almost-empty（两组独立阈值，估计只用本域
  接受后指针与边沿前二级同步对端指针，保守偏大/偏小）；不做分组计数、
  数据校验（ECC/parity）、多写多读端口。
- almost 标志是纯提示，不参与握手；本模型不引入独立复位协议，也不仿真
  亚稳态（仍只建模合法但可能滞后一到两拍的同步值）。
- 不提供网站、通用 CLI、看板或持久化；终端命令仅用于测试与演示。
