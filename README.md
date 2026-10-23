# CAPE

**Channel-Aware Prototype Energy** 独立代码、模型与结果包。

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
    results_all_seeds.csv   两个数据集的六条结果，保留实际 seed 编号
    results_mean_std.csv    各自三个指定种子的均值与样本标准差
    panoradio/             Seed 1、2、3 的结果与逐扰动 CSV
    hfdigital/             Seed 1、3、6 的结果与逐扰动 CSV
    selection_manifest.json 种子选择、协议与来源摘要
  figures/                 绘图程序生成的 PDF / PNG
  train.py                 CAPE 从头训练
  evaluate.py              使用最终权重重评估
  predict.py               对新 I/Q 数据做识别与拒识
  show_results.py          显示指定种子的结果表
  plot_results.py          按各数据集实际种子集合绘图
  verify_release.py        权重加载、模型前向和文件完整性核验
  requirements.txt
```

公开仓库仅提供 **Panoradio HF 的 Seed 1、2、3** 三个 Full CAPE 权重。结果表包含 Panoradio HF Seed 1、2、3 和 HF-Digital Seed 1、3、6。HF-Digital 未上传权重；逐样本评分和训练日志也不随仓库提供。没有保存消融权重、其他方法实现、其他方法权重、预训练代码或独立预训练权重。

**HF-Digital 的 1、3、6 是查看 Seed 1–6 结果后指定的事后子集，并非预先固定的随机种子集合。** Seed 2、4、5 也已完成，但不在此次汇总中。该均值与标准差仅描述指定子集，不能写成全部已完成种子的表现或无偏的随机种子稳定性估计。Panoradio 保留既有的 1、2、3 报告子集，也不作预注册声明。详见 `results/README.md`。

权重只保存模型参数、最终配置、经验原型、阈值和必要来源摘要，不包含优化器和原项目的 Python 模型对象。加载使用 `weights_only=True`，可随整个文件夹迁移。

## 2. 安装

建议 Python 3.9 或更新版本，使用独立环境。在本目录运行：

```powershell
python -m pip install -r requirements.txt
```

GPU 版 PyTorch 请根据自己的 CUDA 环境安装。历史结果的 Python、PyTorch、NumPy 等版本记录在 `protocols/runtime_reference.json`。

## 3. 查看与核验发布内容

无需原始数据即可运行：

```powershell
python -u verify_release.py
python -u show_results.py
```

`verify_release.py` 会检查 Panoradio 的三个最终权重、模型前向计算、文件校验和、逐种子 CSV 一致性、四组 H4 聚合和样本标准差。它不会训练模型。没有原始逐样本评分时，它不能独立重算 AUROC 或 Unknown F1，并会明确提示这一限制。

`plot_results.py` 按 `results/selection_manifest.json` 中各数据集的实际种子编号读取 CSV。现有 `figures/` 图片未为本次子集重新生成；需要新图时可运行 `python -u plot_results.py --output-dir figures_selected`。重新评估可按第 5 节生成新的结果，默认写入 `evaluation/panoradio/`。

## 4. 原始数据

大体积数据集不在代码包中。数据格式详见 `data/README.md`，也可以通过 `--data-root` 指向已有数据目录。

Panoradio HF 需要 `dataset_panoradio_hf.npy` 与 `dataset_panoradio_hf_tags.csv`。

HF-Digital 需要 `X.npy`、`y.npy` 和 `snr_db.npy`。

`protocols/*_split.npz` 保存历史训练所用的原始样本 ID。必须使用相同的数据版本及行顺序，不能重新打乱源文件后继续使用这些 ID。

## 5. 用最终权重重新评估

```powershell
python -u evaluate.py --dataset panoradio --seeds 1 2 3 --data-root "你的Panoradio数据目录"
```

该命令依次加载 Panoradio 的三个最终模型，并输出独立的新评估目录 `evaluation/panoradio/`。默认使用权重内已经保存的经验原型和已知验证集阈值。HF-Digital 权重未公开，不能直接用此包重新执行其模型推理。

保持 batch size=128、evaluation seed=100200，以及相同的运行环境，有助于复现 AWGN 等扰动的随机实现。更换设备、PyTorch/CUDA 或计算内核时可能存在浮点差异。

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

- 数据划分固定，data seed=0。Panoradio 报告训练 seeds=1、2、3；HF-Digital 报告事后指定的 seeds=1、3、6。
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

重新评估时，先对每个种子计算完整指标，再对三个种子的结果计算均值与 **样本标准差 ddof=1**。新生成的 `results_seed*.csv` 和 `results_mean_std.csv` 均为百分数；各种子目录下的 `perturbations.csv` 与 `summary.json` 保留 [0,1] 原始率。这些标准差不是方差、标准误或均方误差。

每个数据集内的报告种子共享同一已知类编码器初始化、数据划分、ETF 方向和评估随机数，因此标准差只描述共享设置下的第二阶段训练变化。HF-Digital 的统计还受事后子集选择影响，不能与全体随机种子方差等同。

## 9. 来源与核验

此次结果 CSV 在发布前已使用保存的逐样本评分独立复核，均值与样本标准差未作手动调整。来源摘要和选择说明保存在 `results/selection_manifest.json`；未公开的原始评分不能仅凭摘要在仓库内独立复现。

`protocols/export_audit.json` 记录已发布权重的来源、转换摘要和原模型/独立实现的逐张量输出对比，其种子字段描述权重，不替代结果表的种子集合。`checksums.sha256` 用于检查交付文件完整性。
