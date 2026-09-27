# 任务简报：Phase 0c 续跑

> 整份交给执行方。在上次 Phase 0c 使用的同一台服务器、同一个环境里继续。
> 原简报 `phase0c_task_brief.md` 的全部约束（尤其第 0 节）继续有效。本简报只说明哪些要补、哪些要重做，以及和原简报不同的地方。

---

## 0. 为什么续跑

上次按原简报第 3 节的停止条件停下：旧池参考逻辑回归复现有 11 个数据集的误差超过 1e-6。委托方的诊断如下：

- 11 个误差全部是"并列单位" 0.5 / (正例数 × 负例数) 的整数倍，也就是说只有并列或几乎并列的样本对换了先后顺序。
- 两次运行的库版本完全相同，差异来自机器和线程数不同造成的浮点差别。

因此，委托方在注册前（没有任何模型对比结果）把复现判定改为预注册第 3.3 节的新规则：误差超过 1e-6 的数据集，只要全部单变量 AUROC 逐位复现、误差是并列单位的整数倍、小于 1e-3，且目标区归属不变，就予以保留。判定由委托方在合并数据池时完成，**执行方不做任何取舍**。

---

## 1. 最重要的约束（与原简报相同）

- 在数据池（旧池 364 个 + 新纳入 114 个，共 478 个）上，**唯一允许拟合的模型是 Phase 0b 的参考逻辑回归**。
- 性能指标只允许在第 6 节规定的 12 个 `make_classification` 合成数据集上计算。
- `code/` 里的冻结代码必须原样导入使用，不得改写。
- 不上传、不推送、不公开任何文件；不做任何分析，也不给出结论。

---

## 2. 任务 R1：11 个超限数据集的全变量边际 AUROC 复现

对以下 11 个数据集，用 Phase 0b 的同一函数重算**全部特征**的单变量边际 AUROC，与 `marginal_aucs.json` 逐一比较：

`1063, 336, 44467, 44464, 44466, 43947, 44447, 993, 44463, 1002, 44413`

输出 `reproduction_flagged_marginals.csv`，列为 `openml_id, n_features, marginal_all_max_abs_error`。

**任何一个数据集的 `marginal_all_max_abs_error` > 1e-9，就停下报告。**这说明数据、标签或折划分与 Phase 0b 不同，是真正的流水线偏离。

上次交付的 `reproduction_check.csv` 保持原样，不要修改。

---

## 3. 任务 R2：补交同源候选清单

按原简报第 2 节，交回 `alias_candidates.csv`（列为 `openml_id_a, name_a, openml_id_b, name_b, reason`）。上次的输出包里没有这个文件。

---

## 4. 任务 R3：环境记录补全，统一线程设置

1. `environment_phase1.json` 中的 `cpu_model` 上次只写了 `x86_64`。改为 `lscpu` 输出的完整型号（"Model name"），并补上物理核心数、逻辑核心数和内存大小。
2. 从本次开始，所有传统模型（包括逻辑回归）的每一次拟合都严格按 `phase1_task_brief_v2.md` 第 2 节"线程与机器"执行：
   - 单线程；
   - 工作进程的 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 均为 1；
   - 逻辑回归用 `threadpoolctl.threadpool_limits(1)` 包住。

   上次记录的 `OMP_NUM_THREADS=16`、`MKL_NUM_THREADS=16` 不再使用。在 `environment_phase1.json` 的 `threads` 中如实记录新设置。
3. 类别列的声明方式，按 `phase1_task_brief_v2.md` 第 3 节第 2 步的表格执行（XGBoost 用 `enable_categorical=True` 加 pandas `category` 类型）。

---

## 5. 任务 R4：元数据与替代分层变量（原简报第 4、5 节）

对全部 478 个数据集，按原简报第 4 节和第 5 节执行，交回：

- `metadata.csv`
- `metadata_descriptions.json`
- `clusters.json`
- `lr_coefficients.json`
- `alt_stratifiers.csv`

其中 `N_eff_cond` 需要拟合的参考逻辑回归，按第 4 节的单线程设置运行。

---

## 6. 任务 R5：合成数据试跑（原简报第 7 节，从头重跑）

- **上次中断的试跑检查点全部作废**，不要续用，从头开始。
- 数据、变体、模型和输出与原简报第 7 节完全相同。
- 传统模型严格单线程，并按第 4 节的类别声明方式运行。
- `compute_projection.md` 按原简报第 7.3 节的要求写。另外单独列出 EBM 的耗时占比：冒烟测试中 EBM 单次拟合约 69 秒，可能是主要耗时。

---

## 7. 任务 R6：Phase 1 用的逻辑回归对照值

在这台服务器上，按第 4 节的单线程设置，对全部 478 个数据集重算参考逻辑回归：

- 使用 Phase 0b 流水线；
- 使用 `cleaned_data` 中的 `fold` 列；
- 汇总折外预测后算 AUROC。

输出 `lr_reference_phase1.csv`，列为：

`openml_id, lr_auc, lr_abs_diff_vs_frozen, tie_units, cpu_model, n_threads`

- `lr_abs_diff_vs_frozen`：与冻结值的绝对差。旧池数据集的冻结值取 Phase 0b `datasets_stratifiers.csv` 的 `auc_lr`，新数据集取 `datasets_stratifiers_ext.csv` 的 `auc_lr`。
- `tie_units`：该差值除以 0.5 / (正例数 × 负例数)。

**若有任何一个数据集的差值 > 1e-6，且 `tie_units` 不是整数（与最近整数相差 > 1e-3）或差值 ≥ 1e-3，就停下报告。**差值小于 1e-6，或者是并列单位的整数倍且小于 1e-3，都属正常，照实记录即可。

这份文件是 Phase 1 逻辑回归复现的对照值。

---

## 8. 交付物（`phase0c_outputs/`，与上次的文件放在一起）

新增或更新：

- `reproduction_flagged_marginals.csv`
- `alias_candidates.csv`
- `environment_phase1.json`（更新）
- `environment_phase1.lock.txt`（如有变化则更新）
- `metadata.csv`、`metadata_descriptions.json`
- `clusters.json`、`lr_coefficients.json`、`alt_stratifiers.csv`
- `pilot_timing.csv`、`pilot_results.csv`、`pilot_C.csv`、`compute_projection.md`
- `lr_reference_phase1.csv`
- `phase0c_report.md`（重写，覆盖上次的停止报告）

`phase0c_report.md` 第一行写**完成**，或**停止**并附原因。正文按原简报第 8 节的要求写，另加三项：

- 任务 R1 的最大误差；
- 任务 R6 差值的分布（多少个为 0 或 < 1e-6，多少个是并列单位整数倍，最大值）；
- CPU 型号。

---

## 9. 自检（全部通过才算完成）

1. Phase 0b 三个文件的哈希未变。
2. R1：11 个数据集的 `marginal_all_max_abs_error` 全部 ≤ 1e-9。
3. R6：满足第 7 节的判定，没有触发停止条件。
4. `alt_stratifiers.csv` 与 `metadata.csv` 的行数均为 478。
5. 试跑的 `pilot_timing.csv` 覆盖 12 个数据集 × 13 个变体 × 5 折 × 9 个模型，另加 M1 的 10 个模型；传统模型的每一行都是单线程。
6. 在池内数据上只拟合过参考逻辑回归（附 `grep` 结果）。
7. `test_generators.py`、`test_phase0c_functions.py` 全部 PASS；`fingerprints.py check generator_fingerprints_phase1env.json` 无不一致。
8. token 与服务器密码没有出现在任何文件中（附 `grep` 结果）。
