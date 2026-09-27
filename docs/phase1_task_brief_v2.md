# 任务简报：Phase 1 v2 — 主基准与干预实验运行

> 整份交给执行方。需要 Python、一块或多块 NVIDIA GPU，以及足够的磁盘（预测文件合计可能达数十 GB）。
> 计算量大，按模块分批跑。**必须逐数据集、逐变体写检查点，支持断点续跑。**
> 本简报取代 v1 版 `phase1_task_brief.md`。

---

## 0. 前置条件（任何一条不满足都不得开跑）

1. **预注册 v2 已在 OSF 完成注册。**委托方会提供注册链接和登记提交的 `MANIFEST.sha256`。没有收到就停下报告。
2. 用 `MANIFEST.sha256` 逐一校验以下文件的哈希，不一致就停下报告：
   - Phase 0b 与 Phase 0c 的产物；
   - `code/` 下全部冻结代码，以及 `synth_calibration.json`、`removal_plan.json`、`stratifier_v2.csv`；
   - 最终数据池 `final_pool/datasets_stratifiers.csv`（478 个纳入数据集）与 `final_pool/marginal_aucs.json`；
   - 预检交付物 `preflight_outputs/`、流水线代码 `pipeline_code/` 与 `pipeline_code_sha256.txt`。
3. 使用 Phase 0c 建立的运行环境。当前环境 `pip freeze` 的输出必须与 `environment_phase1.lock.txt` **逐行一致**。每台租来的 CPU 实例都按 `phase1_preflight_brief.md` 第 1 节第 2 条的方法，从同一个本地 wheel 目录安装。
4. 数据一律从 Phase 0c 落盘的 `cleaned_data/` 读取，**不得重新下载或重新清洗**。只用 Phase 0c 交付的 `cleaned_data.zip` 原件解压；解压前核对其 SHA-256 为 `93d97f647acc7ebbdde1e55afc95c5cd55eb5cb9fd814f3070356676b724c646`，不一致就停下报告。
5. **代码**：
   - `pipeline_code/` 随预注册登记。开跑前在每台机器上进入 `pipeline_code/`，用 `sha256sum -c ../pipeline_code_sha256.txt` 核对，必须全部 OK。
   - **传统模型**：从 `pipeline_code/` 原样导入，不得复制、改写或重新定义。`lr` 用 Phase 0b 的 `prepare_feature_frame` 加 `build_lr_pipeline`（与预检算逻辑回归对照值时完全相同，外层和内层都用它）；其余五个用试跑的 `preprocessors` 与 `cpu_fit_predict`。
   - **基础模型与 FT-Transformer**：试跑里的 `gpu_fit_predict`、`ftt_fit_predict` 是为纯数值的合成数据写的，没有按第 3 节声明类别列、没有插补缺失值，FT-Transformer 也是整批训练。**Phase 1 不用这两个函数**，由驱动脚本按第 2 节表格和第 3 节重新实现输入准备与训练。
   - **驱动脚本**（读 `cleaned_data`、用冻结生成器造变体、折循环、第 3 节的内层选择、基础模型和 FT-Transformer 的输入准备与训练、第 5 节的输出、检查点）可以新写，但除上一条外不得定义任何模型、预处理或超参数。
   - **开跑前的冒烟测试与登记**：在一个玩具数据集上把 10 个模型各跑一个外层折。玩具数据用 `make_classification(n_samples=600, n_features=10, random_state=0)`，另加两个类别列（3 个和 5 个水平），并在 3 列（其中 1 列是类别列）中随机置 5% 缺失。日志里写明每个模型实际收到的类别列位置、类型或参数，以及送进基础模型和 EBM 的数据中没有 NaN。玩具数据上**不计算、不报告任何性能指标**。然后把 `phase1_driver/`、`phase1_driver_sha256.txt` 和 `smoke_phase1.log` 交给委托方，**等委托方确认并登记后，才能在池内数据上做任何拟合。**之后驱动脚本不得改动；如必须修 bug，停下报告，说明改了什么、为什么，得到确认后再继续。
6. **计算范围（注册时已确定，预注册第 5.9 节）：M1、M2、M4、M5、M6 都跑，M3 不跑；`INJECT_REPS = 3`，M2 不删任何变体。**按该范围执行，不得自行增减。
7. **机器分工：**
   - 六个传统模型（包括全部内层选择）在两台 Intel Xeon Platinum 8352V 实例（各分配 32 核）上运行：预检用的 686 机，以及第二台 214 机。每台同时运行的单线程工作进程数与预检一致（32）；
   - **第二台（以及以后任何新增实例）的准入**：`lscpu` 型号必须相同，环境按第 3 条安装且 `pip freeze` 逐行一致；然后按预检第 3 节在这台上重算全部 478 个数据集的参考逻辑回归，与 `preflight_outputs/lr_reference_phase1_preflight.csv` 逐个比较，**每个数据集的绝对差都须 < 1e-12**，输出 `lr_reference_<host>.csv`（列同预检文件，另加 `abs_diff_vs_preflight`）随第一次交付一起交回。任何一个不达标，这台就不用，停下报告；
   - **M5 的全部合成数据都在 686 机上跑**，不分到第二台；
   - **按数据集分配**：每个数据集整体分给一台，它在所有模块、变体、折和模型上的传统模型拟合都在这台上完成；分配表 `host_assignment.csv`（`openml_id, host`）在开跑前写好，随第一次交付交回，之后不得改动（某台机器彻底故障时除外，须在报告中说明）；
   - 三个基础模型和 FT-Transformer 全部在 RTX 4090 上运行；
   - 4090 服务器的 CPU 不跑任何传统模型。
8. **预检已完成**：`phase1_preflight_brief.md` 的交付物已随预注册登记（`preflight_outputs/`），且没有触发停止条件。传统模型所用的 CPU 型号必须与预检时的一致，实例台数见 `frozen/compute_plan.json`；逻辑回归复现的对照值用 `preflight_outputs/lr_reference_phase1_preflight.csv`（见第 6 节第 1 条）。
9. **权重文件**：三个基础模型的权重文件，SHA-256 必须与 `phase0c_outputs/environment_phase1.json` 中 `checkpoints` 登记的值一致（预注册第 5.1 节）。不一致就停下报告。

---

## 1. 本轮做什么，不做什么

在冻结的数据池上运行下表中的 6 个模块，产出逐样本预测概率、逐折指标和内层选择记录。

**不做任何汇总分析：**

- 不计算任何模型间差值（Δ），不做跨数据集汇总；
- 不做分层对比，不报告"某模型在某类数据上更好"。

这些全部由委托方按预注册的分析计划完成。执行方若自行分析并报告结论，会造成分析自由度泄漏，这批结果只能作废。

| 模块 | 数据 | 变体 | 模型 |
|---|---|---|---|
| M1 主实验 | 最终池全部数据集 | 原始数据 | 10 个（含 FT-Transformer） |
| M2 注入 | 目标区全部数据集 | 每个数据集 13 × `INJECT_REPS` 个：每个重复含噪声 1 + 集中 4 + 分散 4 + 分散加噪声 4（默认 3 次重复，共 39 个） | 9 个（不含 FT-Transformer） |
| M3 移除 | **本轮不跑**（预注册第 5.9 节） | — | — |
| M4 样本量 | 目标区全部数据集 | 每个合格大小 × 5 次重复 | 9 个 |
| M5 合成 | 192 个格子 × 20 次重复 | 每个训练集大小 | 9 个 |
| M6 嵌套边际 AUROC | 目标区全部数据集 | 每个外层训练折 | 不拟合模型，只算边际 AUROC |

"最终池"指 `final_pool/datasets_stratifiers.csv` 中 `excluded_reason` 为空的 478 个数据集。"目标区"指 `stratifier_v2.csv` 中 `target_region == True` 的 238 个数据集。

**执行顺序（预注册第 5.9 节，必须遵守）**：

1. M1
2. M6
3. M2 的集中臂与分散加噪声臂：先跑完重复 0，再重复 1，再重复 2
4. M4
5. M2 的其余变体（噪声臂、分散臂）
6. M5

每个模块跑完先交一次，委托方确认格式后再继续。

**计算量**：M2 最重，Phase 0c 的试跑已给出预计耗时。请按数据集并行，并逐变体写检查点。若实际耗时明显超出预计，停下报告，**不要自行减少重复次数、变体或模块**；是否缩减由委托方按预注册的规则决定，并记为偏离。

---

## 2. 模型与配置

全部使用库默认参数。**禁止调参。禁止多种子选优。禁止因某模型表现不佳而更换配置。**

| 代号 | 模型 | 配置 |
|---|---|---|
| `tabpfn35` | TabPFN-3.5 | `tabpfn==9.0.0`，`TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)`（**不是** Fast 版本），检查点 `tabpfn-v3.5-20260909.safetensors`，`random_state=13`（若构造器支持），本地权重 |
| `tabicl2` | TabICLv2 | `tabicl==2.2.0`，`checkpoint_version="tabicl-classifier-v2-20260212.ckpt"`，`random_state=13`，本地权重 |
| `tabpfn2` | TabPFN-2 | `tabpfn==9.0.0`，`create_default_for_version(ModelVersion.V2)`，检查点 `tabpfn-v2-classifier-v2_default.ckpt`（Phase 0c 以内容相同的别名 `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` 加载；以 SHA-256 为准），`random_state=13`，本地权重 |
| `lr` | 逻辑回归 | L2，C=1，`max_iter=2000`，与 Phase 0b 参考逻辑回归流水线**完全相同** |
| `ebm` | EBM | `interpret` 的 `ExplainableBoostingClassifier(random_state=13)`，其余默认 |
| `catboost` | CatBoost | 默认，`verbose=0`，`random_seed=13` |
| `lgbm` | LightGBM | 默认，`random_state=13`，`verbose=-1` |
| `xgb` | XGBoost | 默认，`random_state=13` |
| `rf` | 随机森林 | sklearn 默认，`random_state=13` |
| `ftt` | FT-Transformer | `rtdl_revisiting_models` 的 `FTTransformer.get_default_kwargs()`；AdamW（学习率 1e-4，weight decay 1e-5），每个 epoch 打乱后按 256 行一个 mini-batch 训练；最多 100 epoch，patience=10；早停用训练折内分层 80/20 划分（`random_state=13`）；`BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`，`pos_weight` 在 80% 训练部分上计算；取验证损失最低的那一轮。输入按第 3 节做 one-hot 加标准化。**只在 M1 运行** |

**全局随机性**

- `PYTHONHASHSEED=13`；`numpy` 与 `random` 的种子设为 13。
- `torch.manual_seed(13)`、`torch.cuda.manual_seed_all(13)`。
- 开启 `torch.use_deterministic_algorithms(True)`。若因算子不支持而报错，关闭它并在环境记录中写明。

**线程与机器（为可复现，必须遵守）**

部分库的结果会随线程数变化。例如 LightGBM 的官方文档写明，默认设置下线程数不同，结果可能不同；换系统或换编译器，结果也会不同。因此：

- **传统模型一律单线程。**`lr, ebm, catboost, lgbm, xgb, rf` 的每一次拟合，包括内层选择中的拟合，都只用 1 个线程：
  - `lgbm`：`n_jobs=1`；`xgb`：`n_jobs=1`；`catboost`：`thread_count=1`；`rf`：`n_jobs=1`；`ebm`：`n_jobs=1`；
  - `lr`：用 `threadpoolctl.threadpool_limits(1)` 包住拟合与预测；
  - 每个工作进程启动前设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。
- 这些是执行设置，不是超参数；其余参数仍为库默认。**并行只靠同时运行多个进程**（按数据集 × 变体 × 折分发任务）。
- **机型固定。**传统模型的全部拟合在同一种 CPU 上完成（可以是多台同型号实例），基础模型和 FT-Transformer 的全部拟合在同一种 GPU 上完成。所有机器使用同一份 `environment_phase1.lock.txt`。**同一个模型不得一部分在 A 型机器、一部分在 B 型机器上跑。**
- 开跑前把实际使用的 CPU 型号、GPU 型号和实例数量报给委托方。

**基础模型**

- 只用本地权重，禁止任何远程推理 API。
- 在 GPU 上运行。若库因"仅 CPU 时的样本数保护"拒绝，改用 GPU，**不得子采样**。
- 权重已在 Phase 0c 下载到本地缓存，本轮不需要 token。若加载时要求 token 或联网，停下报告。

---

## 3. 通用流程（每个数据集、每个变体）

1. **外层折**：用 `cleaned_data` 中的 `fold` 列，五折。不得重新划分。
2. **预处理**：全部只在训练折上拟合，再应用到 held-out 折。
   - 数值列：中位数插补。
   - 类别列：众数插补。
   - LR 与 FT-Transformer：one-hot（`handle_unknown='ignore'`）+ 标准化。
   - 树模型、EBM 与三个基础模型：整数编码，使用 `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`，再按下表声明类别列（与 Phase 0c 冒烟测试一致，不得更改）：

     | 模型 | 类别列的声明方式 |
     |---|---|
     | `tabpfn35`、`tabpfn2` | `categorical_features_indices=<类别列位置>` |
     | `tabicl2` | 类别列转为 pandas `category` 类型 |
     | `ebm` | `feature_types` 中类别列记为 `"nominal"` |
     | `catboost` | `cat_features=<类别列位置>`，类别列的整数编码转为字符串 |
     | `lgbm` | `fit(..., categorical_feature=<类别列位置>)` |
     | `xgb` | `enable_categorical=True`，类别列转为 pandas `category` 类型 |
     | `rf` | 无类别参数，直接使用整数编码 |
   - 基础模型和 EBM **不使用**自带的缺失值处理，统一先插补。
3. **拟合与预测**：每个模型在外层训练折上拟合，对 held-out 折输出未校准的 `predict_proba` 正类概率。
4. **内层选择**：仅对 6 个传统模型（`lr, ebm, catboost, lgbm, xgb, rf`）进行。
   - 在外层训练折内用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=13)` 划分内层折。
   - 每个传统模型在每个内层训练集上拟合（预处理同样只在内层训练集上拟合），在内层 held-out 上算 AUROC，取 5 个内层 AUROC 的均值。
   - 均值最高者为选中模型；并列时按 `lr, ebm, catboost, lgbm, xgb, rf` 的顺序取前者。
   - 某个候选模型只要在任一内层折失败，就不参加该外层折的选择，并在记录中写明；六个全部失败时，该外层折的选中模型记为空。
   - 选中模型在外层 held-out 折上的预测，就是第 3 步中它在整个外层训练折上拟合的结果，不另外重拟。
   - **内层选择绝不能使用外层 held-out 折的任何信息。**
5. **逐折指标**：对每个模型、每个外层折，在 held-out 折上计算 AUROC、log loss（概率截断到 [1e-15, 1−1e-15]）、Brier。只算逐折值，**不汇总、不求差**。

---

## 4. 各模块细则

### M1 主实验

- 数据：最终池全部数据集，原始特征。
- 模型：10 个。
- **额外要求（仅 M1）**：三个基础模型也要在第 3 节第 4 步的同一组内层折上拟合，记录它们的内层 AUROC，写入 `selection/M1_fm_inner.csv`。列与 5.2 相同，但 `selected` 一律为 False。这些记录只供路由分析使用，**不参与传统模型的选择**。

### M2 注入（冻结函数：`generators.injection_block`）

1. 对每个目标区数据集、每个重复 `r ∈ range(INJECT_REPS)`，在**完整清洗数据**上（划分外层折之前）调用：
   - `generators.injection_block(y, openml_id, "noise", None, rep=r)`
   - `generators.injection_block(y, openml_id, "concentrated", lev, rep=r)`，`lev ∈ (0.65, 0.75, 0.85, 0.95)`
   - `generators.injection_block(y, openml_id, "dispersed", lev, rep=r)`，同样 4 个 lev
   - `generators.injection_block(y, openml_id, "dispersed_noise", lev, rep=r)`，同样 4 个 lev

   其中 `y` 为 `cleaned_data` 的标签，按 `row_index` 顺序排列。
2. 把返回的列（`noise`、`concentrated`、`dispersed` 各 8 列，`dispersed_noise` 15 列；列名 `inj_1`… ，数值型）追加在原始特征之后。
   - 每个重复 13 个变体，每个数据集共 13 × `INJECT_REPS` 个。
   - 变体名：`noise_r{r}`、`conc_{lev}_r{r}`、`disp_{lev}_r{r}`、`dispn_{lev}_r{r}`，例如 `dispn_0.85_r2`。
   - **注册时确定的全部重复都属于预注册设计。**M2 全部完成后才交付分析。
3. 每个变体完整执行第 3 节流程，使用同样的外层折。模型 9 个。
4. **合理性检查**（只针对注入列，不涉及模型）：在完整数据上计算集中变体的信号列边际 AUROC，与 `lev` 相差应 < 0.06（小样本数据集允许更大波动，超出就记录，不要停）。

### M3 移除（冻结文件：`removal_plan.json`）——**本轮不跑，以下仅作记录**

1. 对每个目标区数据集，按 `removal_plan.json` 删除列，得到以下变体：
   - `top1`；`rand1_d0`…`rand1_d4`；
   - `top3`；`rand3_d0`…`rand3_d4`（仅当计划中 k=3 不为 null）。
2. 每个变体完整执行第 3 节流程。模型 9 个。

### M4 样本量（冻结函数：`generators.subsample_indices`、`generators.subsample_eligible`）

1. 对每个目标区数据集：
   - 设 `n_train_min` 为五个外层训练折中最小的样本数；
   - 对 `size ∈ (100, 250, 500, 1000)`，仅当 `generators.subsample_eligible(size, n_train_min, minority_rate)` 为真时运行。
2. 对每个外层折、每个合格 `size`、每个 `rep ∈ 0..4`：
   - `idx = generators.subsample_indices(y_train, openml_id, fold, size, rep)`，其中 `y_train` 是该外层训练折的标签，按 `row_index` 升序排列；`idx` 是其中的位置。
   - 只用这 `size` 行训练：预处理、内层选择和全部 9 个模型都在这个子样本内完成。
   - 在**完整的**外层 held-out 折上预测。
3. 变体名：`n{size}_r{rep}`。

### M5 合成（冻结函数：`generators.synth_draw` 与 `synth_calibration.json`）

1. 格子：`gen ∈ SYN_GEN`，`k ∈ SYN_K`，`rho ∈ SYN_RHO`，`auc ∈ SYN_AUC`，`n_train ∈ SYN_N_TRAIN`；`rep ∈ 0..19`。
2. 训练集：`synth_draw(gen, k, rho, auc, n_train, rep, "train")`。
3. 测试集：`synth_draw(gen, k, rho, auc, SYN_N_TEST, rep, "test")`。同一格子和重复下，所有训练集大小共用这个测试集。
4. 全部特征为数值型，无缺失。模型 9 个；内层选择在训练集内完成。
5. 只保存测试集上的指标（AUROC、log loss、Brier）和选择记录，**不保存逐样本预测**。

### M6 外层训练折内的边际 AUROC（供路由分析用）

1. 对每个目标区数据集的每个外层训练折，用 **Phase 0b 的边际 AUROC 函数**（方向在训练部分确定、折内百分位映射、汇总）计算全部特征的边际 AUROC。
2. 该函数所需的五折划分，在该外层训练折内部用 `StratifiedKFold(5, shuffle=True, random_state=42)` 生成。
3. 不拟合任何模型。

---

## 5. 输出规格（`phase1_outputs/`）

### 5.1 逐样本预测：`preds/<module>/<openml_id>.parquet`

适用于 M1–M4。长格式，列如下：

```
openml_id, variant, fold, row_index, y, model, p
```

- `row_index` 为 `cleaned_data` 中的行序号。
- `p` 为 float64 的未校准正类概率。
- 使用 snappy 压缩。

### 5.2 内层选择记录：`selection/<module>.csv`

每个（数据集或格子）× 变体 × 外层折 × 传统模型一行：

```
id, variant, fold, model, inner_auroc_1..inner_auroc_5, inner_auroc_mean, inner_status, selected
```

- 对 M1–M4，`id` 是 openml_id；对 M5，`id` 是 `gen|k|rho|auc`，`variant` 为 `n{n_train}_r{rep}`，`fold` 记为 0。
- `inner_status` 取 `ok` 或 `failed`。某候选在任一内层折失败时记 `failed`，相应的 `inner_auroc_*` 留空，并且不参加选择。
- 每组**至多**一行 `selected=True`。只有当 6 个候选全部失败时才为 0 行，此时该外层折的 S 未定义，并在 `run_log.txt` 中记录。
- S_add（从 LR 与 EBM 中选）由委托方根据这些记录在分析时确定，执行方不需要另外记录。

### 5.3 逐折指标：`metrics/<module>.csv`

每个（数据集或格子）× 变体 × 外层折 × 模型一行：

```
id, variant, fold, model, auroc, logloss, brier, n_test, n_pos_test, n_train, fit_seconds, host, cpu_model, gpu_model, n_threads, status, failure_reason
```

`status` 取 `ok` 或 `failed`。失败的行也要保留。

### 5.4 其他文件

- **M6**：`marginal_train/<openml_id>.json`，格式为 `{fold: {feature: auc}}`。
- **FT-Transformer**：`ftt_log.csv`，列为 `openml_id, fold, best_epoch, n_params`。
- **运行日志**：`run_log.txt`。每个数据集 × 模块 × 变体一行，记录耗时、完成情况和失败原因。
- **环境记录**：`environment_run.json`，包括 lock 文件的哈希、GPU、确定性设置、实际使用的权重文件哈希。

---

## 6. 自检（全部通过才算完成）

1. **逻辑回归复现（最重要）**：把 M1 中 `lr` 的五折预测汇总后算 AUROC，与 `preflight_outputs/lr_reference_phase1_preflight.csv`（预检时在同型号 CPU、单线程下算出）相比，绝对误差须 < 1e-12，**任何一个数据集不达标都要停下报告**。（目标区划分和协变量 `auc_lr` 一律使用冻结值，不用这里的对照值。）
2. 对每个模块、数据集和变体，预测行数 = 该数据集的 n × 模型数；M4 的预测行数 = held-out 折行数 × 模型数。
3. `status='ok'` 的行，`p` 全部落在 [0, 1] 内且为有限值。
4. 随机抽 3 个数据集：用 `preds` 重算逐折 AUROC，与 `metrics` 一致（误差 < 1e-12）。
5. `selection`：每组至多一个 `selected=True`；有 `inner_status='ok'` 的候选时恰有一个，并且它等于这些候选中 `inner_auroc_mean` 的最大者（并列时按规定顺序）；0 个的组全部记录在 `run_log.txt` 中。
6. 注入变体的特征数：`dispn_*` 为 p + 15，其余为 p + 8。（M3 不跑，移除变体不适用。）
7. **生成器指纹**：在运行环境中、注册包根目录下执行 `python3 code/fingerprints.py check phase0c_outputs/generator_fingerprints_phase1env.json`，必须**完全一致**；不一致就停下报告。
8. M5 的格子与重复须完整：192 × 20 × 9 个模型。
9. 确认基础模型全程离线、使用本地权重，token 未出现在任何输出文件中。
10. 确认没有做任何调参、种子选优、跨数据集汇总或差值计算。
11. **跨实例一致性**：若传统模型用了多台实例，随机抽 3 个（数据集、变体、折），把 `lgbm`、`catboost`、`ebm` 在另一台同型号实例上重跑，预测概率须与原结果一致（最大绝对差 < 1e-12）。结果写进报告。
12. `metrics` 中所有传统模型行的 `n_threads` 均为 1，`cpu_model` 只有一种取值，且每个数据集的传统模型行 `host` 只有一种取值、与 `host_assignment.csv` 一致，M5 的传统模型行 `host` 全部是 686 机；所有基础模型与 FT-Transformer 行的 `gpu_model` 只有一种取值。

---

## 7. 失败与中断

- 逐数据集 × 变体写检查点，已完成的不重跑。
- 单个模型失败：GPU 上重试一次；仍失败则记为 `failed` 并写明原因，然后继续。**不得子采样，不得换配置。**
- **不要因为某个模型的结果异常（过高或过低）而重跑或更换配置。**异常值是数据，不是错误。
- 任一模块的失败率超过 10% 时，跑完后在报告中单列失败清单，附每个失败数据集的 n、p 与 `tertile_dn`。

---

## 8. 不要做的事

- 不计算任何 Δ，不做任何跨数据集汇总，不做分层对比，不报告模型间的优劣。
- 不调参，不用多个种子，不换配置。
- 不改动数据、折划分、正类定义、注入/移除/子采样规则，也不改写冻结代码。
- 不对概率做任何校准。
- 不使用任何远程推理 API。
- 不静默跳过失败。

---

## 9. 交付

每个模块完成后打包 `phase1_outputs/<module>` 回传。`metrics`、`selection`、`run_log.txt`、`environment_run.json` 必须随每次交付一起提供。第一次交付还要附上 `host_assignment.csv`，以及第二台的 `lr_reference_<host>.csv`。

回传说明中只写四件事：

- 完成数；
- 失败数与失败原因分布；
- 自检第 1 条的最大复现误差；
- 各自检项是否通过。

**不要写任何模型比较的结论。**
