# 任务简报：Phase 0c — 注册前准备（扩池、元数据、替代分层变量、环境、合成数据试跑）

> 整份交给执行方。需要 Python、网络、一块 NVIDIA GPU（用于第 6 节的冒烟测试和第 7 节的试跑）。
> 预计一到两天。必须支持断点续跑。
> 本轮结束后，委托方定稿预注册并在 OSF 注册，然后才进入 Phase 1。

---

## 0. 最重要的约束（与 Phase 0 / 0b 相同，违反则整个项目作废）

**本轮禁止任何模型对比实验。**

- 在数据池的任何数据集上，**唯一允许拟合的模型是 Phase 0b 的参考逻辑回归**，而且只用于第 3 节的复现校验和第 5 节的 N_eff_cond。
- **不得**在任何池内数据集上运行 TabPFN、TabICL、CatBoost、XGBoost、LightGBM、随机森林、EBM、FT-Transformer。
- 第 6 节的冒烟测试只在 `sklearn.datasets.make_classification` 生成的玩具数据上跑，**只报告能否运行和耗时，不报告任何性能指标**。
- 第 7 节的试跑是**唯一的例外**：只在第 7 节规定的 12 个 `make_classification` 合成数据集上，完整跑一遍 Phase 1 的 M2 注入流程，包括 AUROC 和差值。**这 12 个数据集以外的任何数据，一律不计算性能指标。**
- 在池内数据上，不计算、不报告任何模型间的性能差值。

原因：正式研究是预注册设计。一旦本阶段在池内数据上产生模型对比结果，这些数据集就永久失去预注册资格。

**随包附带的冻结代码**（`code/` 目录）必须原样导入使用，不得改写、不得复制粘贴后修改：

- `generators.py`
- `stratifier.py`
- `phase0c_functions.py`
- `fingerprints.py`
- `synth_calibration.json`
- 以及它们的测试文件

---

## 1. 前置条件

1. Phase 0b 的三个产物可用，且哈希如下。**不一致就停下报告。**
   - `datasets_stratifiers.csv`：`5BB2A3CE033A2A112E58024EEA26ABB8FC6D909A4FB293D871CFECCDFE2169DE`
   - `marginal_aucs.json`：`0C6D23707FB6ECF93BA39D8E3421398FEF91B103097FFE0762781FFACBCC2780`
   - `environment.json`：`CCAC64AE468720478A84B9526CE264C15ECECDA5FDDB49BC9F9791A8F5ABE909`
2. Phase 0b 的脚本与运行环境可复用：Python 3.12.10、scikit-learn 1.9.1、openml 0.15.1、pandas 3.0.5、numpy 2.5.3、scipy 1.18.1。**第 2–5 节必须在这个环境里跑。**
3. 上述三个文件**不得修改**。本轮所有新产物都写成新文件。

---

## 2. 任务 A：扩大数据池

**目的**：增加独立来源家族数，尤其是信号集中那一端（目前只有约 20 个家族）。

1. **扫描范围**：接着 Phase 0b 的 OpenML 全库升序扫描，从 `openml_id = 44598` 开始，一直扫到扫描时 OpenML 上存在的最大 ID。**不设候选数上限。**
2. **规则**：筛选、去重、标识列删除、时间序列与图像/文本排除、家族标注，全部使用 Phase 0b 的代码和 `environment.json` 里的取值，**一个字不改**。
3. **边际 AUROC**：对新纳入的数据集，用 Phase 0b 的同一函数计算，包括嵌套方向和折内百分位映射。
4. **不要在看到覆盖情况后调整任何规则。**数量多少都如实报告。

**家族别名的判断交给委托方，不要自行合并。**把"可能是同一份底层数据"的新旧数据集对列入 `alias_candidates.csv`，列为：`openml_id_a, name_a, openml_id_b, name_b, reason`。判定线索包括：

- 相同的 `original_data_url`；
- 描述文本高度相似；
- n 相同、p 相同且名称相近；
- 名称是已有家族名的变体，例如大小写不同，或多了 `_dataset`、`-v2` 之类的后缀。

**输出**（均为新文件）：

- `datasets_stratifiers_ext.csv`：只含新候选，列与 Phase 0b 完全相同，被排除的也要保留并写明原因。
- `marginal_aucs_ext.json`：只含新纳入的数据集。
- `alias_candidates.csv`
- `run_log_ext.txt`

---

## 3. 任务 B：清洗数据落盘 + 复现校验

对**旧池 364 个加上新纳入的全部数据集**：

1. 用 Phase 0b 的清洗代码，把清洗后的数据写成 `cleaned_data/<openml_id>.parquet`。
   - 包含全部保留特征列，特征的列名和顺序与 `marginal_aucs.json` 中的键一致。
   - 标签列 `y`：正类 = Phase 0b 的少数类，编码为 1。
   - `fold` 列：`StratifiedKFold(5, shuffle=True, random_state=42)` 在清洗后完整数据上的划分，取值 0–4。
   - 类别列保存 Phase 0b 使用的整数编码，并附 `cleaned_data/<openml_id>.schema.json`，记录每列是数值型还是类别型，以及类别编码表。
   - 缺失值保留为缺失，**不做插补**。
2. **复现校验一**：在旧 364 个数据集上重跑参考逻辑回归，AUROC 与 `datasets_stratifiers.csv` 的 `auc_lr` 相比，**绝对误差须 < 1e-6**。若超过 5 个数据集不达标，停下报告。
3. **复现校验二**：随机抽 5 个旧数据集，重算全部特征的边际 AUROC，与 `marginal_aucs.json` 相比，误差须 < 1e-9。
4. 把每个数据集的误差写进 `reproduction_check.csv`，列为：`openml_id, lr_abs_error, marginal_max_abs_error`。未抽中做复现校验二的数据集，`marginal_max_abs_error` 留空。
5. 逻辑回归误差 > 1e-6 的旧数据集会由委托方在合并时排除出最终池，执行方不要自行删除，只需如实记录。**每一个旧的纳入数据集都必须有一行，`lr_abs_error` 必须是有限数值**；缺行或缺值的数据集在合并时同样会被排除。

---

## 4. 任务 C：元数据（用于医学子集标注）

对最终池的全部数据集（旧加新），从 OpenML 抓取以下字段：

- `name`、`version`、`description`（全文）、`tags`
- `original_data_url`、`citation`、`collection_date`、`creator`

输出：

- `metadata.csv`：每行一个数据集，description 以外的字段放这里。
- `metadata_descriptions.json`：`{openml_id: description 全文}`。
- `metadata.csv` 额外加以下列：
  - `medical_terms`：调用冻结函数 `phase0c_functions.medical_candidate(name, description, tags)` 的返回值，用分号连接。
  - `n_numeric`、`n_categorical`
  - `missing_share`：插补前全部特征单元格中缺失的比例。
  - `n1`、`n0`：清洗后的正类数与负类数。

**不要**给出"是否医学数据"的判断。标注由委托方人工完成。

---

## 5. 任务 D：替代分层变量

对最终池的全部数据集，调用冻结函数计算以下两个量。**这一步只拟合参考逻辑回归。**

### 5.1 N_eff_cluster

- 输入：`cleaned_data` 中插补前的特征矩阵，类别列用整数编码，缺失为 NaN，**不使用标签**。
- 调用 `phase0c_functions.feature_clusters(X_codes, names)` 得到分簇。
- 再调用 `phase0c_functions.neff_cluster(marginal_aucs[该数据集], clusters, n1, n0)`，其中 n1、n0 是清洗后的正类数与负类数。该函数使用去噪后的 lift。
- 同时保存分簇结果：`clusters.json`，格式为 `{openml_id: {feature: cluster_id}}`。

### 5.2 N_eff_cond

1. 在 `fold` 定义的每个外层训练折上，用 **Phase 0b 的参考逻辑回归流水线**拟合，预处理完全相同。
2. 把该折的 held-out 行做同样的变换，得到 `Xt_heldout`，也就是逻辑回归实际与系数相乘的矩阵。
3. 建立 `groups`：每个变换后的列属于哪个原始特征。数值特征对应一列，类别特征对应其全部 one-hot 列。
4. 对每一折调用 `phase0c_functions.feature_contributions(coef, Xt_heldout, groups)`，五折结果组成列表后调用 `phase0c_functions.neff_conditional(per_fold, feature_names)`，其中 `feature_names` 取该数据集 `marginal_aucs` 的键。
5. 保存每折的系数和 groups：`lr_coefficients.json`，供审计。

### 5.3 输出

`alt_stratifiers.csv`，列为：`openml_id, neff_cluster, neff_cond, n_clusters, p_used`。

---

## 6. 任务 E：Phase 1 运行环境与冒烟测试

在 Phase 0b 环境的基础上建立 Phase 1 环境。Phase 0b 已有的库版本**保持不变**，新增：

- `tabpfn==9.0.0`
- `tabicl==2.2.0`
- `interpret`（当前最新版）
- `catboost`、`lightgbm`、`xgboost`
- `threadpoolctl`
- `torch`（CUDA 版）
- `rtdl_revisiting_models`

### 6.1 许可与权重

- TabPFN-2.5 及以后版本的权重需要一次性接受许可。委托方会提供 token，**只能通过环境变量 `TABPFN_TOKEN` 传入，不得写入任何文件、日志、代码或 git 历史。**权重下载完成、确认能从本地缓存加载后，立即 `unset TABPFN_TOKEN`，并清除 shell 历史中含 token 的记录。
- 本地下载以下权重，**不得使用 `tabpfn-client` 或任何远程推理 API**：
  - TabPFN-3.5：`ModelVersion.V3_5`，**不是** Fast 版本；检查点文件应为 `tabpfn-v3.5-20260909.safetensors`
  - TabPFN-2：`ModelVersion.V2`；检查点应为 `tabpfn-v2-classifier-v2_default.ckpt`，也可能以其内容相同的别名 `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` 加载（名字里的 finetuned 是历史遗留，并非微调）。记录实际加载的文件名和 SHA-256。若加载的是其他文件，停下报告
  - TabICL：`tabicl-classifier-v2-20260212.ckpt`
- 三个检查点都要记录文件名、SHA-256，以及模型卡或论文中关于训练数据的原文表述（是否只用了合成数据）。TabPFN-2 的模型卡没有写训练数据，就记录"模型卡未说明"，并引用其论文（Hollmann et al., Nature 2025）的表述；同时查找发布说明、HF 提交历史或包内文档中能证明 `tabpfn-v2-classifier-v2_default.ckpt` 就是该论文所述模型的记录，原文摘录并附链接；找不到就写明"未找到"。任何一个来源若写明该检查点在真实数据上训练或微调过，停下报告。

### 6.2 离线冒烟测试（只在玩具数据上跑）

1. 数据：`sklearn.datasets.make_classification(n_samples=600, n_features=12, n_informative=4, weights=[0.7], random_state=0)`，前 480 行训练，后 120 行测试。另外把第 0、1 列离散化成 4 档类别特征，用来测试类别参数的传法。
2. 断网运行：设置 `HF_HUB_OFFLINE=1`，并断开网络或用防火墙阻断。依次拟合并预测以下 10 个模型：TabPFN-3.5、TabICLv2、TabPFN-2、LR、EBM、CatBoost、LightGBM、XGBoost、RF、FT-Transformer。
3. **每个模型只记录**：是否成功、`predict_proba` 的形状、拟合与预测耗时、实际生效的默认超参数（`get_params()` 或等价方式）、类别特征通过什么参数传入。**不要计算或报告 AUROC 或任何性能指标。**
4. 另外在 GPU 上跑一个 n = 8000、p = 70 的 `make_classification` 玩具数据，确认 TabPFN-3.5、TabICLv2 和 TabPFN-2 不会因规模限制拒绝运行。若库有"仅 CPU 时限制样本数"的保护，确认它在 GPU 上不触发，并记录相关参数名。

### 6.3 冻结代码校验

在 **Phase 1 环境**中：

1. 运行 `python3 test_generators.py` 和 `python3 test_phase0c_functions.py`，必须全部 PASS。
2. 运行 `python3 fingerprints.py > generator_fingerprints_phase1env.json` 并交回。**这份文件会随预注册一起登记，成为 Phase 1 必须逐位复现的基准。**
3. 另外运行 `python3 fingerprints.py check generator_fingerprints.json`，与委托方环境的结果对比。不一致不算错误（不同 numpy 版本的随机数流可能不同），但必须报告两边的 numpy 版本和不一致的键。**不要改代码。**

### 6.4 输出

- `environment_phase1.json`，记录：
  - Python 与全部库的精确版本；
  - 三个基础模型检查点的训练数据表述（见 6.1）；
  - GPU 型号、CUDA 版本、驱动版本；
  - 三个基础模型的权重文件名与 SHA-256；
  - 每个模型实际生效的默认超参数；
  - 类别特征的传参方式；
  - 确定性算子设置；
  - 线程设置（见 Phase 1 简报第 2 节"线程与机器"）与 CPU 型号。
- `environment_phase1.lock.txt`：`pip freeze` 的输出。
- `smoke_test.csv`：列为 `model, ok, proba_shape, fit_seconds, predict_seconds, offline, note`。

---

## 7. 任务 F：合成数据试跑（计时与方差，不碰数据池）

**目的**：在注册前估计 Phase 1 的总耗时，并得到注入实验中 C_i 的离散程度（预注册第 5.9 节和第 7 节）。这一步同时是 Phase 1 流水线的一次完整预演。

### 7.1 数据（只用这 12 个）

对 `i = 0..11`，按下面的顺序遍历 `n ∈ (500, 1000, 2000)`、`p ∈ (10, 30)`、`frac ∈ (0.3, 0.8)`（`n` 在最外层，`frac` 在最内层），生成：

```python
X, y = sklearn.datasets.make_classification(
    n_samples=n, n_features=p, n_informative=max(2, round(frac * p)), n_redundant=0, n_repeated=0,
    n_classes=2, weights=[0.65], flip_y=0.05, class_sep=0.5, random_state=1000 + i)
```

数据集的标识用字符串 `f"pilot{i}"`，在需要 `openml_id` 的地方传入它，例如 `injection_block(y, f"pilot{i}", ...)`。外层折为 `StratifiedKFold(5, shuffle=True, random_state=42)`。

### 7.2 运行

- 按 Phase 1 简报（`phase1_task_brief_v2.md`）第 2–5 节实现流水线，**与 Phase 1 使用同一套代码**，包括第 2 节"线程与机器"的单线程设置。
- 在每个试跑数据集上跑 M2：噪声臂、集中臂、分散臂、分散加噪声臂，4 个强度，**只跑重复 0**，共 13 个变体。
- 模型：三个基础模型和六个传统模型，含内层选择（选出 S 和 S_add）。
- 同时在每个试跑数据集的原始数据（不注入）上跑一次 M1 的全部 10 个模型，用于计时。

### 7.3 输出

- `pilot_timing.csv`：每行一次拟合。列为 `dataset, variant, fold, model, stage (outer|inner), n_train, p_train, fit_seconds, predict_seconds, device`。
- `pilot_results.csv`：每个试跑数据集、每个变体、每个基础模型的 `Δ_sel`、`Δ_add`，以及所用的 S 和 S_add。
- `pilot_C.csv`：每个试跑数据集、每个基础模型的 C_i 和 C_i^add（定义见预注册第 5.8 节，只用重复 0）。
- `compute_projection.md`：按最终池的数据集规模分布，预测 Phase 1 每个模块（M1–M6）需要的**单线程 CPU 核时**（传统模型）和 **GPU 小时**（基础模型与 FT-Transformer），分开列出。写明试跑机器的 CPU 型号和 GPU 型号，并分别给出以下两种情况的墙钟时间：只用现有服务器；另加 N 个 32 核 CPU 实例（N = 1、2、4、8）。

### 7.4 限制

- 这一节的结果只用于两件事：预注册第 7 节的功效说明，以及第 5.9 节的计算规模决定。**不要**根据试跑结果提出修改假设、指标或分析的建议。
- 试跑数据集以外，不得计算任何性能指标。

---

## 8. 交付物清单（`phase0c_outputs/`）

- `datasets_stratifiers_ext.csv`
- `marginal_aucs_ext.json`
- `alias_candidates.csv`
- `run_log_ext.txt`
- `reproduction_check.csv`
- `metadata.csv`
- `metadata_descriptions.json`
- `clusters.json`
- `lr_coefficients.json`
- `alt_stratifiers.csv`
- `environment_phase1.json`
- `environment_phase1.lock.txt`
- `smoke_test.csv`
- `generator_fingerprints_phase1env.json`
- `pilot_timing.csv`、`pilot_results.csv`、`pilot_C.csv`、`compute_projection.md`
- `phase0c_report.md`

`cleaned_data/` 单独打包回传。

`phase0c_report.md` 第一行写结论：**完成**，或**停止**并附原因。正文只报告：

- 扩池：新候选数、新纳入数、排除原因计数、新纳入的家族数；
- 两项复现校验的最大误差；
- 冒烟测试中每个模型是否成功，以及耗时；
- 试跑：每个模型的中位拟合耗时，以及 `compute_projection.md` 的汇总（性能数字只放在 `pilot_*.csv` 里，报告正文不写）；
- 测试与指纹校验结果；
- 遇到的问题。

---

## 9. 自检（全部通过才算完成）

1. Phase 0b 三个文件的哈希未变。
2. 复现校验一：超出 1e-6 的数据集 ≤ 5 个；复现校验二全部 < 1e-9。
3. `cleaned_data` 中每个数据集：行数 = n；`y` 的均值 = `minority_rate`；`fold` 恰有 5 个取值，各折阳性率与整体相差 ≤ 0.02；特征列名集合与 `marginal_aucs` 的键一致。
4. `alt_stratifiers.csv` 行数 = 最终池数据集数；`neff_cond` 和 `neff_cluster` 均为有限正数，个别无法计算的写明原因。
5. 全程在池内数据上只拟合过逻辑回归。请在报告中写明，并附 `grep` 结果，证明池内数据处理代码中没有调用其他模型。
6. 冒烟测试确实在断网状态下完成，且没有输出任何性能指标。
7. 试跑只用了第 7.1 节的 12 个数据集；`pilot_timing.csv` 覆盖全部 13 个变体 × 5 折 × 9 个模型，另加 M1 的 10 个模型。
8. token 没有出现在任何文件中。请在报告中附对输出目录执行 `grep -r` 查找 token 前 6 位的结果。

---

## 10. 不要做的事

- 不在池内数据上跑逻辑回归以外的任何模型。
- 不报告任何模型的性能（冒烟测试也不报告）。唯一例外是第 7 节 12 个合成数据集的试跑结果，只写进 `pilot_*.csv`。
- 不改动 Phase 0b 的任何文件、规则或代码。
- 不自行合并家族别名，只列候选。
- 不给数据集做医学标注。
- 不改写冻结代码。
- 不把 token 写进任何地方。
- 不静默跳过失败，每个失败都记入日志。
