# ZXW_Project (`analysis` 分支) — 第一阶段：项目深度解读、80篇核心文献综述与实施路线图

> **当前分支**：`analysis`（第一阶段：调研分析与总体方案设计专有分支）  
> **项目名称**：基于像素化阵列探测器的准直孔和康普顿散射智能伽马成像算法研究  
> **指导教师**：韩纪锋 研究员（四川大学原子核科学技术研究所 / 720所）

---

## 📚 阅读导航（建议按以下顺序阅读，30分钟彻底吃透整个课题）

| 序号 | 文档路径 | 核心解决的问题 | 推荐优先级 |
| :---: | :--- | :--- | :---: |
| **01** | [`docs/01_project_deep_dive.md`](./docs/01_project_deep_dive.md) | **【零门槛白话 + 硬核物理全解】** 到底什么是像素化阵列、准直孔成像、康普顿散射圆锥？为什么要双模融合？探测器修正修什么？传统迭代算法慢在哪？AI 怎么替代它？ | ⭐⭐⭐⭐⭐ **首推必读** |
| **02** | [`docs/02_advisor_and_lab_context.md`](./docs/02_advisor_and_lab_context.md) | **【知己知彼：导师与720所背景剖析】** 韩纪锋老师团队擅长什么？手里有什么硬件基础？为什么出这道题？怎么准备才能在联系导师时让人眼前一亮？ | ⭐⭐⭐⭐⭐ **首推必读** |
| **03** | [`docs/03_literature_review_synthesis.md`](./docs/03_literature_review_synthesis.md) | **【80篇文献全景精读综述】** 七大技术模块演进脉络、国内外代表性样机参数、传统算法 vs AI 算法定量对比，以及支撑我们立项的**三大研究空白（Research Gaps）**。 | ⭐⭐⭐⭐⭐ |
| **04** | [`docs/04_literature_database_80papers.md`](./docs/04_literature_database_80papers.md) <br> [`docs/04_literature_database_80papers.csv`](./docs/04_literature_database_80papers.csv) | **【80篇 100% 真实核验 DOI 文献库】** 涵盖川大本院及导师团队、双模伽马相机、GAGG/CZT阵列、Geant4仿真、AI事例甄别、传统MLEM与轻量化AI重建网络，支持直接导入 Excel / Zotero。 | ⭐⭐⭐⭐ |
| **05** | [`docs/05_proposal_strategy_and_roadmap.md`](./docs/05_proposal_strategy_and_roadmap.md) | **【第二阶段蓝图：大创申报书写作路线】** 四川大学大创申请书八大章节布局、三大创新点标准表述、考核指标设定与后续 `proposal` 分支执行计划。 | ⭐⭐⭐⭐⭐ |
| **06** | [`docs/06_demo_architecture_and_roadmap.md`](./docs/06_demo_architecture_and_roadmap.md) | **【第三阶段蓝图：初版 Demo 架构与路线】** 双引擎解耦架构、`DetectorDigitizer` 参数、`LiteEventNet` (32K) + `LiteDualGammaNet` (0.62M) 轻量化双模融合网络与 5 步开发里程碑。 | ⭐⭐⭐⭐⭐ |
| **00** | [`docs/00_process_log.md`](./docs/00_process_log.md) | **【过程备份记录】** 第一阶段调研检索、数据核验与方案演进完整记录日志。 | ⭐⭐⭐ |

---

## 🌿 仓库多分支总览

- `main`：仓库顶层说明与多分支规范索引
- `analysis`（**当前分支**）：第一阶段深度解读、80篇文献数据库与实施路线图（**已完成**）
- `proposal`：第二阶段四川大学大创项目申报书撰写分支（**骨架已就绪，待指令启动**）
- `demo`：第三阶段初版双模伽马仿真与轻量化 AI 重建 Demo 开发分支（**骨架已就绪，待指令启动**）
