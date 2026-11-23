# 第一阶段调研、文献核验与架构设计全过程记录日志

> **分支**：`analysis`  
> **阶段目标**：不急于盲目写代码或套模板写申请书，先彻底搞懂课题物理本质、导师研究脉络，完成 50–100 篇高相关性真实学术文献调研与总结，形成有理有据的大创申请书与初版 Demo 实施路线，全过程留痕备份于 Git。

---

## 过程里程碑记录表

| 时间节点 | 阶段动作 | 关键产出与核心结论 | 对应 Git 提交文件 |
| :--- | :--- | :--- | :--- |
| **Step 1** | 仓库初始化与多分支架构设计 | 建立 `main`（总览）、`analysis`（调研分析）、`proposal`（大创申请书）、`demo`（初版原型开发）四分支隔离备份机制，关联远端仓库 `https://github.com/sanshibro/ZXW_Project.git`。 | `README.md`, `.gitignore` |
| **Step 2** | 指导教师与川大720所背景深挖 | 厘清第一指导教师**韩纪锋研究员**（川大720所副所长、辐射物理及技术教育部重点实验室主任）在位置灵敏闪烁探测器、Geant4模拟、脉冲堆积反卷积及FPGA硬件验证、机器学习辐射甄别方向的研究积淀，明确课题重在**“真实探测器物理退化修正 + 轻量化边缘可部署快速伽马重建”**。 | `docs/02_advisor_and_lab_context.md` |
| **Step 3** | 80篇真实学术文献跨库检索与核验 | 通过 Crossref Polite Pool API 与 NCBI PubMed E-utilities 检索初筛 200+ 篇候选文献，人工剔除光学高光谱等非核辐射噪声文献，严格核验保留 **80 篇 100% 真实可查 DOI 的核技术/医学物理/AI计算成像核心文献**（含川大本院编码准直与韩纪锋研究员最新论文、中科院高能所 Kong 2022 双模相机、早稻田大学 Omata 2020 主动针孔相机、*ComptonNet* 2024、*SwinCCIR* 2026、*CoReSi* 2025 等），归纳出当前三大研究空白。 | `docs/03_literature_review_synthesis.md`<br>`docs/04_literature_database_80papers.md`<br>`docs/04_literature_database_80papers.csv` |
| **Step 4** | 课题白话拆解与数学物理建模 | 从光学透镜为何无法聚焦伽马射线讲起，用通俗类比+严谨公式打通“光电效应准直投影 + 康普顿圆锥散射运动学 + 多普勒与能量/位置分辨退化 Digitizer + 传统联合 LM-MLEM 复杂度瓶颈 + 双分支跨模态融合 AI 网络”。 | `docs/01_project_deep_dive.md` |
| **Step 5** | 申报书与初版 Demo 路线图制定 | 凝练大创申请书三大核心创新点与八大章节写作攻略；针对初版 Demo 提出**“统一事件接口 + Python矢量化蒙特卡罗快验引擎(v0.1)与Geant4标准引擎(v1.0)解耦 + 两级轻量化网络(`LiteEventNet` 32K + `LiteDualGammaNet` 0.62M)”**工程实施方案。 | `docs/05_proposal_strategy_and_roadmap.md`<br>`docs/06_demo_architecture_and_roadmap.md` |
