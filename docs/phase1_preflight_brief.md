# 任务简报：Phase 1 预检（在一台租来的 CPU 实例上）

> 整份交给执行方。在预注册登记**之前**进行，交付物会随预注册一起登记。本步骤只拟合参考逻辑回归（池内数据），以及在 Phase 0c 的 12 个合成试跑数据集上重跑少量模型计时。
> 目的有两个：
> 1. 在 Phase 1 实际使用的 CPU 型号上，生成逻辑回归复现的对照值；
> 2. 实测这种 CPU 的单核速度，据此决定租几台。

---

## 0. 约束（与 Phase 0c 相同）

- 在数据池（478 个数据集）上，唯一允许拟合的模型是 Phase 0b 的参考逻辑回归。
- 第 4 节的计时只在 Phase 0c 第 7 节规定的 12 个 `make_classification` 合成数据集上进行，**只记录耗时，不计算任何性能指标**。
- `code/` 里的冻结代码原样导入使用，不得改写。
- 不上传、不推送、不公开任何文件；服务器密码不写进任何文件。

---

## 1. 准备

1. 在 Phase 1 跑传统模型的那台 CPU 实例上进行：AutoDL"686 机"，Intel Xeon Platinum 8352V，平台分配 32 核、60 GB 内存（委托方已租下，登录方式单独给你）。**预检和 Phase 1 用同一台，预检结束后不要释放、不要重装系统。**
2. 用 `phase0c_outputs/environment_phase1.lock.txt` 建立完全相同的 Python 环境（Python 3.12.10），`pip freeze` 必须逐行一致。
   锁文件里的包指向 4090 服务器上的本地 wheel 目录 `/root/autodl-tmp/tabular_prior_phase0c/phase1_linux_wheelhouse/`。把这个目录整个拷到新实例的**同一路径**，再用 `pip install --no-index -r environment_phase1.lock.txt` 安装；不要从网上重新下载任何包。
3. 拷入以下文件：
   - 委托方给的注册包，整个目录原样拷入，保持目录结构（下文命令都在这个目录的根下运行）；
   - 你自己写的全部流水线代码，原样拷入，不得改写：Phase 0b 的全部代码（数据清洗、边际 AUROC、参考逻辑回归），以及 Phase 0c 的全部代码（R1–R6 与合成数据试跑）。Phase 1 也将只用这套代码；
   - `cleaned_data.zip` 原件：解压前核对 SHA-256 为 `93d97f647acc7ebbdde1e55afc95c5cd55eb5cb9fd814f3070356676b724c646`，不一致就停下报告；然后解压到注册包根目录下，得到 `cleaned_data/`。
4. 记录 `lscpu` 的完整输出，重点是型号（Model name）、`Socket(s)`、`Core(s) per socket`、`Thread(s) per core`。容器里 `lscpu` 显示的是整台宿主机，所以另外记录平台**分配给本实例**的核数（`nproc`，以及 `/sys/fs/cgroup/cpu.max` 或 `cpu.cfs_quota_us` / `cpu.cfs_period_us`）和内存上限。

---

## 2. 冻结代码校验

- 若注册包里有 `MANIFEST.sha256`：`sha256sum -c MANIFEST.sha256` 必须全部 OK。
- `python3 code/test_generators.py`、`python3 code/test_phase0c_functions.py` 全部 PASS。
- `python3 code/fingerprints.py check phase0c_outputs/generator_fingerprints_phase1env.json`：必须 0 个不一致。不一致就停下报告。

---

## 3. 逻辑回归对照值（与 Phase 0c 续跑任务 R6 的方法相同）

在这台实例上，按 `phase1_task_brief_v2.md` 第 2 节"线程与机器"的单线程设置，对 `final_pool` 的全部 478 个数据集重算参考逻辑回归：

- 使用 Phase 0b 流水线；
- 使用 `cleaned_data` 中的 `fold` 列；
- 汇总折外预测后算 AUROC。

输出 `lr_reference_phase1_preflight.csv`，列为：

`openml_id, lr_auc, abs_diff_vs_phase0c_reference, tie_units, cpu_model, n_threads`

- `abs_diff_vs_phase0c_reference`：与 `lr_reference_phase1.csv` 中 `lr_auc` 的绝对差。
- `tie_units`：该差值除以 0.5 / (正例数 × 负例数)。

**停止条件：**若有任何一个数据集的差值 > 1e-6，并且满足以下任一条，就停下报告：

- `tie_units` 与最近整数相差 > 1e-3；
- 差值 ≥ 1e-3。

其余情况照实记录即可。

---

## 4. 单核速度实测（只在合成试跑数据集上）

在这台实例上，单线程重跑 `pilot_timing.csv` 中以下拟合，**只记录耗时**：

- 数据集：Phase 0c 第 7.1 节的 12 个 `make_classification` 数据集，生成方式完全相同；
- 变体：原始数据（M1），以及注入变体 `dispn_0.85_r0`；
- 模型：`ebm`、`catboost`、`lgbm`、`xgb`、`rf`、`lr`，外层拟合与内层选择拟合都要跑，只跑第 0 折，即 `pilot_timing.csv` 中 `fold` 为 `0`（`stage=outer`）和 `o0_i0`–`o0_i4`（`stage=inner`）的行。

同时开的单线程工作进程数，取平台**分配给本实例的核数**（应为 32），与 Phase 0c 试跑时"分配 16 核、同时开 16 个进程"的做法一致。报告里写明实际数字。

输出 `preflight_timing.csv`，列为：

`dataset, variant, fold, model, stage, fit_seconds, predict_seconds, pilot_fit_seconds, pilot_predict_seconds, ratio`

- `pilot_*`：`pilot_timing.csv` 中同一拟合的耗时。
- `ratio`：本机耗时 ÷ 试跑耗时。

---

## 5. 交付

打包交回以下三项，三者放在同一层（不要把后两项放进 `preflight_outputs/`）：

1. `pipeline_code/`：第 1 节第 3 条所列流水线代码的完整副本，保持原有目录结构。只放代码和配置文件，不放数据、权重、日志和缓存。
2. `pipeline_code_sha256.txt`：`pipeline_code/` 中每个文件的 SHA-256（`sha256sum` 格式，路径相对于 `pipeline_code/`）。Phase 1 只能用哈希完全相同的这套代码。
3. `preflight_outputs/`，内含：
   - `lr_reference_phase1_preflight.csv`
   - `preflight_timing.csv`
   - `lscpu.txt`
   - `environment_preflight.json`（库版本、CPU 型号、宿主机物理与逻辑核心数、分配给本实例的核数与内存、线程设置、同时运行的工作进程数）
   - `preflight_report.md`

`preflight_report.md` 第一行写**完成**，或**停止**并附原因。正文包括：

- 测试与指纹结果；
- 第 3 节差值的分布：多少个为 0 或 < 1e-6，多少个是并列单位的整数倍，最大值；
- 第 4 节各模型 `ratio` 的中位数，以及 EBM 与 CatBoost 的 `ratio` 中位数；
- 按 `phase0c_outputs/compute_projection.md` 各模块的 CPU 核时乘以 EBM 的 `ratio` 中位数，除以分配核数，得到在这 1 台上跑完传统模型的墙钟小时数（分模块列出，再给总数）；
- 池内数据上只拟合过参考逻辑回归的 `grep` 证明；
- 没有任何密钥出现在 `pipeline_code/` 和其他交付文件中的 `grep` 证明：TabPFN token、服务器密码、OpenML `apikey`、`HF_TOKEN` 或其他 Hugging Face token，以及任何形如 API key 的字符串。`pipeline_code/` 会公开，发现任何一处都要先删掉再交付，并在报告中写明。

**不要**报告任何模型的性能指标。
