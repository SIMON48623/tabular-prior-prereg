# Phase 1 驱动更正 C1 改动说明

## 冻结标识

- 上一版驱动清单 SHA-256：`58d0f8527a563f8dc849cba8fb448b8661f8c39d2a6c92468d6639cf09835839`
- C1 驱动清单 SHA-256：`136c1e4207c5168be34288ac4846aae147bc4234ba4e566f23d8adcdfd0cc68c`
- C1 清单共覆盖 36 个文件。

## C1 改动

1. `runtime.py`
   - `verify_driver_manifest` 的路径解析和文件集合比较均改为以清单所在目录 `driver_dir.parent` 为基准，支持清单中的真实 `phase1_driver/` 前缀。
   - 增加 NVIDIA 驱动版本读取，用于 GPU 环境记录。
2. `scheduler.py`
   - 增加 `--verify-only`；执行注册包、驱动、cleaned_data、计划、环境、主机和硬件等启动检查后退出，不要求授权文件，不创建检查点，不调度单元。
   - GPU 调度固定为逻辑角色 `4090`，实际主机名从机器读取；不再要求该主机名出现在 `host_registry.json`，但仍严格要求 GPU 型号包含 `RTX 4090`。
   - GPU 环境记录增加实际 NVIDIA 驱动版本；CPU 686/214 仍按 `host_registry.json` 绑定。
3. `aggregate.py`
   - 增加必需参数 `--groups cpu/gpu/ftt`。
   - 覆盖、身份、唯一性和交付汇总只针对指定组；`cpu` 组同时包含 M6 的无模型 `none` 单元。
4. 测试
   - 真实前缀清单端到端核验。
   - CPU、GPU、FTT 三组独立覆盖与汇总。
   - GPU 主机名不绑定、RTX 4090 型号严格核验。
   - `--verify-only` 不读取授权文件且不进入调度。

## 文件范围审计

相对上一冻结版，驱动中只有以下文件变化：

- `phase1_driver/runtime.py`
- `phase1_driver/scheduler.py`
- `phase1_driver/aggregate.py`
- `phase1_driver/tests/test_runtime_contract.py`
- `phase1_driver/tests/test_scheduler_review3.py`
- `phase1_driver/tests/test_c1_contract.py`（新增）

其余驱动文件逐文件 SHA-256 保持不变。

## 验证结果

- 在 686 的既有 Phase 1 精确环境中运行全部单元测试：76/76 通过。
- 686 实机 `--verify-only`：通过；识别登记角色 686，CPU 为 Intel Xeon Platinum 8352V。
- 214 实机 `--verify-only`：通过；识别登记角色 214，CPU 为 Intel Xeon Platinum 8352V。
- 两台实机核对到的 C1 驱动清单 SHA-256 均为 `136c1e4207c5168be34288ac4846aae147bc4234ba4e566f23d8adcdfd0cc68c`。
- 两次 `--verify-only` 均未提供授权文件，未创建检查点，未调度或拟合任何单元。
- GPU 实机核验和生产模式冒烟测试按委托方要求延期至可用 RTX 4090 提供后执行。

## 日志哈希

- `verify_only_686.log`：`dad49db631feee4bc0f8cd8c3108172ba23e686e2018faa690c5578280d89ba2`
- `verify_only_214.log`：`c6023225fa5bd3e194f245057b5ba43808a4d448c82e8da07fb01e04c2826d11`
