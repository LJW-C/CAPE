# CAPE

**Channel-Aware Prototype Energy** 是面向信道扰动的无线电信号开集识别方法。CAPE 通过独立的时域和频域编码器及 CFDA 学习信号表征，使用类别/SNR 分层采样和固定 ETF 分类头，在原始已知类样本上以交叉熵训练；推理时使用经验原型能量识别已知类并拒识未知信号。

## 发布内容

- `cape/`、`configs/`：模型、数据处理、评分、评估与实验配置。
- `checkpoints/panoradio/`：Panoradio HF 的 Seed 1、2、3 最终权重，包含模型参数、经验原型和拒识阈值。
- `results/`：Panoradio HF 和 HF-Digital 的 Seed 1、2、3 结果、逐扰动指标及汇总。汇总采用三次运行的均值和样本标准差（`ddof=1`）。
- `protocols/`：固定样本划分、参考运行环境和权重转换记录。

HF-Digital 仅提供结果。发布包不包含原始数据、逐样本评分或历史编码器初始化文件。

## 安装与核验

```powershell
python -m pip install -r requirements.txt
python -u verify_release.py
python -u show_results.py
```

核验无需原始数据，检查已发布权重、模型前向、文件完整性及结果汇总。运行以下命令可从发布的 CSV 重新绘图：

```powershell
python -u plot_results.py --output-dir figures_selected
```

## 评估与推理

数据格式见 `data/README.md`。重新评估须使用与 `protocols/*_split.npz` 对应的数据版本和样本顺序。

```powershell
python -u evaluate.py --dataset panoradio --seeds 1 2 3 --data-root "你的Panoradio数据目录"
python -u predict.py --checkpoint checkpoints/panoradio/seed1.pt --input query_iq.npy --output predictions.npz
```

评估默认使用权重中保存的原型和阈值。推理接受复数 `[N,L]` 或实数 `[N,2,L]`、`[N,L,2]` 输入，无需 SNR；`predicted_class=-1` 表示拒识。

## 从头训练

```powershell
foreach ($seed in 1,2,3) {
  python -u train.py --dataset panoradio --seed $seed --data-root "你的Panoradio数据目录" --output-dir "training/panoradio_seed$seed"
}
```

训练 HF-Digital 时，将 `--dataset` 改为 `hfdigital`，并替换数据与输出路径。`train.py` 使用随机初始化；历史结果使用已知类编码器初始化后进行交叉熵微调。重新训练与历史初始化实验应分别报告。

## 实验协议

两个数据集均使用固定数据划分（`data seed=0`），报告实际训练 Seed 1、2、3。已知类标注预算为 10%，其中 20% 用于验证。选模和阈值校准仅使用已知类数据，校准后的阈值在六种测试扰动下保持固定。

Known Acc. 是干净已知样本中被接受且分类正确的比例。H4 对每种扰动分别计算干净已知、扰动已知的接受率与干净未知、扰动未知的拒识率的调和均值，再对六种扰动取平均。
