# 甲骨文字形身份响应检索

对应论文：Compact Identity-Response Representations for Oracle Bone Character Image Retrieval。作者：周洁、陆忠麟、毕俊峰；通讯作者：周洁。稿件尚未发表。项目代码采用MIT许可，第三方代码保留原许可。

## 先运行轻量复现

使用Python 3.10—3.13与独立环境：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s feature_level/tests -v
python -m unittest discover -s tests -v
python feature_level/reproduce.py table-summary --output table_summary.json
python examples/synthetic_demo.py --output demo_output
```

人工特征演示不依赖原始图像或权重，用于检查训练统计、支持原型和独立查询检索的完整接口，不是论文实验结果。表格汇总直接读取随包数值证据。

## 图像级复现

安装适合CPU/GPU的PyTorch 2.6，再安装requirements-images.txt。从论文配套Online Resource 4（ESM_4.zip / frozen_assets_and_manifests.zip）解压assets和benchmark_manifests至仓库根目录；原始OBC306、HWOBC图像从数据提供方获取。Git源代码不包含权重或原始图像，当前没有额外的模型下载链接。

```bash
python tools/verify_assets.py --assets assets
python reproduce_images.py encode --assets assets --image-root /your/OBC/images --manifest benchmark_manifests/OBC_test.json --vocabulary obc --device cpu --output outputs/OBC_Test
python reproduce_images.py evaluate --gallery outputs/OBC_Test --readout identity --output outputs/OBC_identity_metrics
```

完整用法、新查询格式、数值路线、指标和复现范围见英文README。OBC输出277维；HWOBC以目标Train原型形成3881维，保留相同视觉与几何资产。查询编码不输入真实身份标签，完整图库评价另行读取类别前缀。

本次发布覆盖冻结模型推理与评价，并保留特征级复现；全部从零训练流程不在此包内。归档GPU数值核验覆盖32张OBC Val和8张HWOBC Val探针，不等同于在每个新环境重跑完整Test。
