# Geant4 标准蒙特卡罗仿真无缝接入指南 (`geant4_interface`)

本目录提供与主工程 `DetectorDigitizer`、`LiteEventNet` 及 `LiteDualGammaNet` **100% 数据协议兼容** 的 C++ Geant4 宏配置与转换接口：

1. **几何对标**：
   - 前端准直层：$z = +24.0\text{ mm}$ 处 $33 \times 33$ 镶嵌钨合金（`G4_W`，厚度 $3.2\text{ mm}$）Rank-17 MURA 编码掩模（或中心针孔）；
   - 散射/准直层（Layer 0）：$z = 0.0\text{ mm}$ 处 $17 \times 17$ 像素化 $\text{Gd}_3\text{Al}_2\text{Ga}_3\text{O}_{12}\text{:Ce}$ 闪烁晶体阵列（单晶条 $2.0 \times 2.0 \times 4.5\text{ mm}^3$）；
   - 吸收层（Layer 1）：$z = -20.0\text{ mm}$ 处 $17 \times 17$ 像素化 $\text{GAGG:Ce}$ 阵列（单晶条 $2.0 \times 2.0 \times 16.0\text{ mm}^3$）。
2. **物理过程列表**：
   - 强制启用 `G4EmLivermorePhysics`（精确模拟核外束缚电子动量引起的多普勒展宽效应、光电吸收荧光逃逸峰与电子对产生）。
3. **数据对接流程**：
   ```bash
   # 将 Geant4 输出的 Hit CSV 文件直接转换为 Digitizer 标准输入并调用轻量化 AI 重建
   python demo/geant4_interface/convert_geant4_csv.py --input hits_output.csv --output geant4_digitized.npz
   ```
