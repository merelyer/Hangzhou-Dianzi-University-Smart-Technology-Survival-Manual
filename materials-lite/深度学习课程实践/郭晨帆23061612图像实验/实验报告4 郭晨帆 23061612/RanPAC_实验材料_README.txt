RanPAC 实验材料说明

一、主要文档
1. RanPAC实验报告_带图表_郭晨帆_23061612.docx
   最终版实验报告，格式仿照“实验4.docx”，已插入运行结果图表。

2. RanPAC实验报告_模仿实验4_郭晨帆_23061612.docx
   未插入图表前的仿实验四版本。

二、运行结果
1. ranpac_core_quick_results.csv
   本机快速验证实验结果：
   Task 1 Top-1 = 48.75%
   Task 2 Top-1 = 42.25%

2. ranpac_core_quick_results_summary.txt
   对快速验证实验配置和结果的文字摘要。

三、图表
1. ranpac_local_accuracy_curve.png
   本机快速验证实验的增量准确率变化图。

2. ranpac_official_cifar224_comparison.png
   RanPAC官方CIFAR-100 224x224结果对比图。

四、代码
1. code/quick_ranpac_core_experiment.py
   生成本机快速验证结果的脚本。核心流程包括：
   合成类增量数据 -> 随机投影 -> 累积统计量 -> 岭回归分类头 -> 任务阶段准确率评估。

2. code/build_ranpac_report_like_exp4.py
   以“实验4.docx”为模板生成仿格式实验报告的脚本。

3. code/add_ranpac_charts_to_report.py
   生成图表并插入Word报告的脚本。

4. code/build_ranpac_report.py
   早期正式报告版本生成脚本，保留作参考。

五、说明
完整RanPAC官方实验需要下载CIFAR-100和预训练模型权重，且运行时间较长。本材料中的本机结果来自轻量化RanPAC核心机制验证实验，不能直接等同于论文完整实验精度；论文级对比结果来自官方仓库paper_tables表格。
