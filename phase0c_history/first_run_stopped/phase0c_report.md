停止：旧池逻辑回归复现中有 11 个数据集的 `lr_abs_error > 1e-6`，超过简报允许的最多 5 个，因此按停止条件终止后续工作。

## 扩池

- 扫描得到新候选 131 个；纳入 114 个；排除 17 个。
- 排除原因计数：`p_used_outside_5_100 = 17`。
- 新纳入数据集覆盖 64 个家族。
- Phase 0b 三个冻结输入的 SHA-256 复核未变：`datasets_stratifiers.csv`、`marginal_aucs.json`、`environment.json` 均与简报第 1 节一致。

## 复现校验

- 旧池参考逻辑回归逐例复现：364/364 行均有有限误差；最大绝对误差为 `0.0001238599255457`；其中 11 个数据集超过 `1e-6`，OpenML ID 为 `1063, 336, 44467, 44464, 44466, 43947, 44447, 993, 44463, 1002, 44413`。
- 随机抽取的边际 AUROC 复现：最大绝对误差为 `0.0`，低于 `1e-9`。
- 清洗阶段已落盘 478 个 Parquet 数据集；因逻辑回归复现停止条件触发，未继续计算替代分层变量，也未自行删除复现失败数据集。
- 池内数据处理全程只拟合 Phase 0b 参考逻辑回归。源码审计命令对 `phase0c_pool_runner.py` 搜索 `TabPFN|TabICL|CatBoost|XGBoost|XGBClassifier|LightGBM|LGBMClassifier|RandomForest|ExplainableBoosting|FTTransformer`，结果为 `POOL_MODEL_GREP_ZERO_MATCHES`。

## 离线冒烟测试

进程级 socket 阻断已先自检生效，并同时设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`；容器不允许创建独立网络命名空间（`unshare -n` 返回 `Operation not permitted`）。下列时间单位均为秒，未计算任何性能指标。

| 模型 | 成功 | predict_proba 形状 | 拟合 | 预测 |
|---|---:|---:|---:|---:|
| TabPFN-3.5 | 是 | `(120, 2)` | 2.211278 | 1.146392 |
| TabICLv2 | 是 | `(120, 2)` | 0.445552 | 0.110780 |
| TabPFN-2 | 是 | `(120, 2)` | 0.170390 | 0.231980 |
| LR | 是 | `(120, 2)` | 0.003849 | 0.000222 |
| EBM | 是 | `(120, 2)` | 69.333524 | 0.000914 |
| CatBoost | 是 | `(120, 2)` | 2.901473 | 0.001116 |
| LightGBM | 是 | `(120, 2)` | 0.042880 | 0.001690 |
| XGBoost | 是 | `(120, 2)` | 0.036679 | 0.000907 |
| RF | 是 | `(120, 2)` | 0.134951 | 0.006325 |
| FT-Transformer | 是 | `(120, 2)` | 5.514464 | 0.000000 |

GPU 规模保护测试也全部成功：TabPFN-3.5、TabICLv2、TabPFN-2 在 `n=8000, p=70` 上的输出形状均为 `(1600, 2)`，未触发 CPU 样本数限制。

## 合成试跑

- 停止条件触发时，CPU 已完成 56/840 个“数据集 × 变体 × 外层折”检查点，GPU 已完成 324/840 个检查点；这些只是中断恢复检查点，未汇总或交付任何不完整的性能结果。
- `pilot_timing.csv`、`pilot_results.csv`、`pilot_C.csv` 与 `compute_projection.md` 未生成；没有报告试跑性能数字，也没有据此提出分析或假设修改。

## 测试与指纹

- `test_generators.py`：全部 PASS。
- `test_phase0c_functions.py`：全部 PASS。
- 指纹检查：参考 NumPy `2.4.4`，本环境 NumPy `2.5.3`，不一致键为 0 个。

## 遇到的问题

- 服务器无法连接 Hugging Face，因此按委托方建议在本地下载三份权重后上传；服务器端从本地缓存断网加载成功。实际文件名与 SHA-256 已写入 `environment_phase1.json`。
- 首次冒烟脚本暴露当前库 API 适配问题：CatBoost 类别列需字符串类型，`rtdl_revisiting_models==0.0.2` 的 FT-Transformer 构造器需显式 `d_out=1`。修正调用方式后，完整冒烟测试全部成功；未改变冻结代码或模型超参数。
- token 仅通过隐藏输入进入下载进程环境，未写入脚本、配置、日志或报告。残留扫描对 `phase0c_outputs/` 检查 9 个文件、对 `/root` 检查 161983 个文件；完整 token 与其唯一载荷前 6 位的匹配数均为 0。缓存 token 文件与 shell 历史中未发现 token。
- 最终阻断原因是旧池逻辑回归复现超限；按照简报要求，GPU/CPU 试跑进程已终止，未继续元数据、替代分层变量、完整试跑、计算投影或最终自检。
