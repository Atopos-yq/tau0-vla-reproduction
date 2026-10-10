# AutoDL 复现运行说明

## 环境与已知冲突

本次实验使用 RTX 4090 24GB。模型/示例环境为 Python 3.11.17、PyTorch 2.7.1+cu128（构建 CUDA 12.8）、Transformers 5.5.4。LIBERO 独立环境为 Python 3.10.22、robosuite 1.4.0、MuJoCo 3.2.3。

**依赖尚未锁定为可一键重建版本。** 原 requirements.txt 的 Transformers 5.5.4 与 LeRobot 0.4.1 的 huggingface-hub 约束冲突。实验曾在隔离环境内用 `--no-deps` 安装 LeRobot 并通过数据模块检查；这不是无冲突的正式依赖方案。不要直接把原 requirements.txt 当作锁文件。需保存实际环境 `pip freeze`，上游 scripts/setup.sh 使用前也需要处理冲突。

## 权重与数据

官方基础与 LIBERO 权重分别放在 `checkpoints/tau-0-vla-base` 与 `checkpoints/tau-0-vla-libero`。G1 评测使用本地后训练 checkpoint。

GitHub 提交不包含权重、outputs 或 example_data 数据文件，仅保留示例数据的来源和许可证说明。必须从上游获取完整示例数据（parquet、元数据、三路视频）后才能评测。

## 10 步动作专家训练检查

从仓库根目录执行，先确认基础权重、三路视频解码和 LeRobot 数据读取成功，并释放已有模型服务占用的显存：

```bash
export NPROC_PER_NODE=1
export AUTO_RESUME=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
bash scripts/train.sh configs/example_agibot_world_gong/train_head_smoke.yaml
```

配置冻结视觉、语言部分，仅训练动作专家及投影；batch=1、max_steps=10。基础权重路径改为相对仓库根目录路径。输出预计为 outputs/agibot_gong_head_smoke，实际位置以训练日志为准。

## G1 HTTP 开环

服务端：

```bash
python -m deploy.server \
  --model outputs/agibot_gong_head_smoke/checkpoint-10 \
  --host 127.0.0.1 --port 10089 --device cuda:0 \
  --infer-mode eager --warmup-steps 1
```

另一个终端，先将 max-inferences 改为 3 做流程检查，通过后跑 100 次：

```bash
mkdir -p outputs/gong_http_verified_100
set -o pipefail
python -u -m deploy.openloop_with_server \
  --server-url http://127.0.0.1:10089 \
  --max-inferences 100 --start-idx 0 \
  --out-dir outputs/gong_http_verified_100 \
  2>&1 | tee outputs/gong_http_verified_100/eval.log
```

deploy/openloop_with_server.py 包含 Tensor→NumPy、CHW→HWC uint8 的客户端修复；deploy/gong_http_eval.py 是保留的替代包装入口。浮点图像须为 [0,1]，状态保持数值数组，不能套用图像的 uint8 转换。

此 G1 路由恢复 16 维有效动作。/act_lerobot_bytes 对 g1_agibot_36 不完整 SDK 向量的拒绝检查保持不变，请使用语义 /act 端点；不要随意补零绕过检查。

## LIBERO 和高层组件

LIBERO 环境和评测入口见 configs/libero/README.md；使用 deploy.libero_server，与 G1 关节服务不同。Proposal 与世界模型见 high_level/README.md，本次未验证完整 TTC 或实机闭环。

## 复现证据与后续工作

需补齐每次服务启动的 checkpoint 绝对路径、全部权重 SHA256、Git commit、补丁、数据哈希、实际 pip freeze 和运行参数。Gong 应按完整 episode_index 划分训练/验证/测试，统计量仅由训练数据估计，并从基础权重重新训练。当前代码包不包含已完成的回合隔离实验。

保留 LICENSE、世界模型 NOTICE 和 example_data/README.md 中的 CC BY-NC-SA 4.0 数据许可证。
