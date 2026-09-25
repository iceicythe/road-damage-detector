# 数据来源

- RDD2022 官方仓库：https://github.com/sekilab/RoadDamageDetector
- 官方 Figshare：https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547
- 起步子集：Japan、India；保留下载来源、版本和原始包校验值。
- 官方 train 部分含 XML；官方 test 不含公开真值。本项目从有标注部分重新划分，自定义测试成绩不等于官方榜单成绩。
- 数据权利与引用要求以原始发布页为准；本仓库不打包第三方图片。

解压后的约定结构：

```text
data/raw/RDD2022/
  Japan/train/images/*.jpg
  Japan/train/annotations/xmls/*.xml
  India/train/images/*.jpg
  India/train/annotations/xmls/*.xml
```

可选分组 CSV 示例（内容是格式示意，需填入真实分组）：

```csv
image,group
Japan/train/images/example_001.jpg,japan_route_001
Japan/train/images/example_002.jpg,japan_route_001
India/train/images/example_001.jpg,india_route_001
```

group 必须全局唯一地标识道路/视频/采集段或经确认的近重复组。提供 CSV 时必须覆盖所有有 XML 的图片。不要仅凭相邻文件名推断道路关系。

