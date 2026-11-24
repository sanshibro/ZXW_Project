# 四川大学大学生创新训练计划项目申报书（proposal 分支 · 69页完整详尽版）

本分支存放指导教师 **韩纪锋研究员（四川大学原子核科学技术研究所 / 720所）** 课题《**基于像素化阵列探测器的准直孔与康普顿散射双模智能伽马成像算法研究**》的完整版项目申报书及配套高分辨率实验图集。

## 核心交付文件清单

1. **四川大学大学生创新训练计划项目申报书_完整详尽版_69页.docx**（同 四川大学大学生创新训练计划项目申报书_正式排版稿.docx，3.21 MB）
   - 严格按学术规范排版（A4 标准页边距、小四宋体 1.5 倍行距、规范三线表、带页码导引线目录），经 WPS Office 实测共 **69 页**（逾 7.2 万字符）。
   - 采用 python-docx 标准 OpenXML 原生嵌图，完整内嵌 **12 幅高分辨率物理原理与初版原型系统（Demo）实测图**、**16 张定量技术规格与实验对比三线表** 以及 **72 篇经 DOI 核验的 GB/T 7714 标准参考文献**。
2. **四川大学大学生创新训练计划项目申报书_完整详尽版_69页.pdf**（3.84 MB）
   - 由 WPS Office 直接导出的 69 页高清定稿 PDF 版本，便于跨设备审阅与打印装订。
3. **四川大学大学生创新训练计划项目申报书_完整详尽版.md**
   - 完整 Markdown 源文稿（自动关联 ssets/fig01–ig12 高清图集，支持在 GitHub 页面直接在线浏览全文与配图）。
4. **关于智能伽马成像课题的初步理解与实施思路.docx**
   - 前期联系指导教师邮件所附的 3 页简明课题实施思路书归档备份。

## 内嵌 12 幅科研与初版 Demo 实验实测图索引（ssets/）

- ig01_cross_sections.png：GAGG:Ce 闪烁晶体光子相互作用截面演化与钨准直掩模高能穿透定量曲线
- ig02_dual_mode_geometry.png：双层 17×17 GAGG:Ce 阵列与近场放大率匹配 Rank-17 MURA 双模几何构型图
- ig03_technical_roadmap.png：项目总体研究技术路线图
- ig04_network_architecture.png：LiteEventNet + LiteDualGammaNet（0.815 M 参数）两级轻量化网络详细架构图
- ig05_physics_and_digitizer.png：理想 vs 五级退化数字化能谱（662 keV 峰与 477 keV 康普顿边）、MURA 互相关 PSF 及 ARM 分布（FWHM = 9.2°）
- ig06_shadowgrams_and_cone_buildup.png：59.5 keV vs 661.7 keV 探测器阴影对比与 8/40/2093 康普顿圆锥累积及 AI 去伪影结果
- ig07_event_net_performance.png：第一级 LiteEventNet 收敛曲线、四分类混淆矩阵（87.41%）与多类 ROC 曲线（Macro AUC = 0.9712）
- ig08_wideband_single_sources.png：59.5 / 356 / 662 keV 单点源四算法对比及跨模态门控权重自适应演化规律
- ig09_dual_point_and_extended_shapes.png：12° 近距双点源、356 keV 圆环源与 662 keV 非对称 L 形扩展面源重建对比
- ig10_mixed_isotope_flagship.png：低能 Am-241（59.5 keV）+ 高能 Cs-137（661.7 keV）双核素混合场一次同步重建对比（提速 23.2 倍）
- ig11_profiles_and_ablation.png：一维角分辨剖面、双源瑞利判据谷底、3.5%–14% 能量分辨率灵敏度曲线及 40 场景模块消融实验
- ig12_low_count_and_speed_benchmark.png：1.2×10⁴–9.0×10⁴ 极低计数稳健性曲线与单帧推理耗时—PSNR 帕累托前沿对比