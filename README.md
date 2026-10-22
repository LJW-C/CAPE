# CAPE

**Channel-Aware Prototype Energy** 独立代码与结果包。

本目录包含当前最终版本：时频编码器、CFDA、类别/SNR 分层采样、固定 ETF 分类头和经验原型能量拒识。训练目标为原始已知样本上的 `CE(x,y)`。模型名称统一为 **CAPE**。

## 1. 内容

```text
CAPE/
  cape/                    模型、数据加载、采样、评分、评估和文件工具
  configs/                 两个数据集的最终配置
  checkpoints/
    panoradio/seed1.pt      Panoradio HF 的三个最终模型
    panoradio/seed2.pt
    panoradio/seed3.pt
  protocols/               固定样本划分、权重转换审计、参考运行环境
  results/
    results_seed1.csv       两个数据集的 Seed 1 结果
    results_seed2.csv       两个数据集的 Seed 2 结果
    results_seed3.csv       两个数据集的 Seed 3 结果
    results_mean_std.csv    三种子均值与样本标准差
    results_all_seeds.csv   逐种子完整结果
    panoradio/             分数据集结果、逐扰动结果与训练日志
    raw/                   原始分数和类别预测，用于独立核验
  figures/                 绘图程序生成的 PDF / PNG
  train.py                 CAPE 从头训练
  evaluate.py              使用最终权重重评估
  predict.py               对新 I/Q 数据做识别与拒识
  show_results.py          显示三种子结果表
  plot_results.py          结果图与信道鲁棒性图
  verify_release.py        权重、阈值、逐种子指标和标准差核验
  requirements.txt
```

公开仓库仅提供 **Panoradio HF 的 Seed 1、2、3** 三个 Full CAPE 权重。HF-Digital 保留代码、配置、合并汇总表中的结果和原始评分，未上传其权重，且不再提供单独的 `results/hfdigital/` CSV 目录。没有保存消融权重、其他方法实现、其他方法权重、预训练代码或独立预训练权重。

权重只保存模型参数、最终配置、经验原型、阈值和必要来源摘要，不包含优化器和原项目的 Python 模型对象。加载使用 `weights_only=True`，可随整个文件夹迁移。

## 2. 安装

建议 Python 3.9 或更新版本，使用独立环境。在本目录运行：

```powershell
python -m pip install -r requirements.txt
```

GPU 版 PyTorch 请根据自己的 CUDA 环境安装。历史结果的 Python、PyTorch、NumPy 等版本记录在 `protocols/runtime_reference.json`。

## 3. 先查看与核验已有结果

无需原始数据即可运行：

```powershell
python -u show_results.py
python -u verify_release.py
python -u plot_results.py
```

`verify_release.py` 会检查 Panoradio 的三个最终权重，并从两个数据集保存的 clean/degraded 分数重新计算阈值、指标与三种子标准差。它不会训练模型。HF-Digital 的核验针对已保存结果，其权重不在公开仓库中。

绘图默认输出 `figures/fig_cape_seed_results.pdf` 和 `figures/fig_cape_channel_robustness.pdf`，同时保存 PNG。

缺少独立逐扰动 CSV 时，绘图脚本从 `results/raw/` 中保存的评分与阈值读取指标，不会重新训练模型或更改汇总表。

```powershell
python -u plot_results.py --dataset panoradio
python -u plot_results.py --dataset hfdigital
```

## 4. 原始数据

大体积数据集不在代码包中。数据格式详见 `data/README.md`，也可以通过 `--data-root` 指向已有数据目录。

Panoradio HF 需要 `dataset_panoradio_hf.npy` 与 `dataset_panoradio_hf_tags.csv`。

HF-Digital 需要 `X.npy`、`y.npy` 和 `snr_db.npy`。

`protocols/*_split.npz` 保存历史训练所用的原始样本 ID。必须使用相同的数据版本及行顺序，不能重新打乱源文件后继续使用这些 ID。

## 5. 用最终权重重新评估

```powershell
python -u evaluate.py --dataset panoradio --seeds 1 2 3 --data-root "你的Panoradio数据目录"
```

该命令依次加载 Panoradio 的三个最终模型，并输出独立的新评估目录 `evaluation/panoradio/`。默认使用权重内已经保存的经验原型和已知验证集阈值。HF-Digital 权重未公开，不能直接用此包重新执行其模型推理；仍可运行 `verify_release.py` 重算其历史结果。

保持 batch size=128、evaluation seed=100200，以及相同的运行环境，有助于复现 AWGN 等扰动的随机实现。更换设备、PyTorch/CUDA 或计算内核时可能存在浮点差异；本包另提供原始分数以核验历史表格。

## 6. 对新信号推理

```powershell
python -u predict.py --checkpoint checkpoints/panoradio/seed1.pt --input query_iq.npy --output predictions.npz
```

输入可为复数 `[N,L]`，或实数 `[N,2,L]` / `[N,L,2]`。输出中的 `predicted_class=-1` 表示拒识，其余为原始已知类别 ID。推理不需要 SNR。

## 7. 重新训练 CAPE

```powershell
python -u train.py --dataset panoradio --seed 1 --data-root "你的Panoradio数据目录" --output-dir training/panoradio_seed1
python -u train.py --dataset panoradio --seed 2 --data-root "你的Panoradio数据目录" --output-dir training/panoradio_seed2
python -u train.py --dataset panoradio --seed 3 --data-root "你的Panoradio数据目录" --output-dir training/panoradio_seed3
```

HF-Digital 将 `--dataset` 改为 `hfdigital`，并指定对应数据目录。训练默认 40 epochs、batch size=128、AdamW、学习率 0.0003、weight decay=0.0001。过程持续打印 epoch、loss、训练准确率和已知验证集指标。

**历史结果与从头训练的区别：** 本包的历史模型曾使用已知类编码器初始化，再进行 CE 微调；按照本次打包要求，该初始化的源码与独立权重不随包提供。最终模型已经包含完整编码器参数，因此已有权重的推理和评估不依赖初始化文件。`train.py` 是相同最终架构的随机初始化训练入口，不能把其结果直接声称为历史初始化实验的严格重现。要从零完全复现历史训练轨迹，还需要另行保留并提供原始初始化流程。

选模只看 known-validation 的接受且分类正确率。为对应历史选模，选模阶段保留原有的原型距离加分类熵辅助分数；最终拒识仍仅使用 PE。最终原型用唯一原始拟合样本构建，阈值用独立已知验证集校准，未知测试样本不参与训练、选模或校准。

## 8. 最终协议

- 数据划分固定，data seed=0，报告训练 seeds=1、2、3。
- 10% 已知类标注预算中的 20% 留作已知验证，其余用于监督拟合。
- Panoradio HF：已知类 0--11，未知类 12--17。
- HF-Digital：已知类 0--10，未知类 11--16。
- CFDA：attention dim=128、4 heads、dropout=0.1、suppression strength=1。
- 固定 ETF：orientation seed=0，训练分类温度=0.2。
- PE：经验已知类原型，energy temperature=0.2，无额外分数归一化。
- 阈值使 clean known-validation 中恰好 `floor(0.05*N)` 个样本满足 `score > threshold`。
- 评估扰动：AWGN=10、CFO=0.05、phase=60、amplitude=0.7、multipath=(4,0.3)、Doppler=0.05。具体操作与历史代码数值一致，属于受控压力测试。

Known Acc. 指干净已知测试样本中同时被接受且分类正确的比例。AUROC、Unknown F1 和 H-score 使用干净测试决策。KAR-deg 与 URR-deg 对六种扰动等权平均。

对每一种扰动先计算四组调和均值：

```text
H4(p) = 4 / sum(1 / (r_j + 1e-8))
r = [KAR-clean, KAR-deg(p), URR-clean, URR-deg(p)]
Avg. H4 = mean_p H4(p)
```

先对每个种子计算完整指标，再对三个种子的结果计算均值与 **样本标准差 ddof=1**。`results_seed*.csv`、`results_all_seeds.csv` 和 `results_mean_std.csv` 均为百分数；`perturbations_seed*.csv` 与 `results/raw/` 保留 [0,1] 原始率，绘图程序会转换为百分数。这些标准差不是方差、标准误或均方误差。

三个 CAPE 种子共享同一已知类编码器初始化、数据划分、ETF 方向和评估随机数，因此历史标准差反映第二阶段训练随机性。种子 1、2、3 是既有报告子集，不能声称为事前注册的随机种子选择。

## 9. 结果来源与核验

所有结果均从历史原始预测分数重新计算，未重新选择权重或按未知测试指标调参。

HF-Digital 三个 H4 为 **37.32494、27.03057、40.80342**，均值与样本标准差为 **35.05298 ± 7.16200**。若旧论文表中标准差写为 2.16，该值与原始三种子结果不符；本包保留真实的 7.16。

`protocols/export_audit.json` 记录源权重摘要、转换后权重摘要和原模型/独立实现的逐张量输出对比。`checksums.sha256` 用于检查交付文件完整性。
