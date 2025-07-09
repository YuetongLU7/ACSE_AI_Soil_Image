# 测试说明

## 目录结构

```
tests/
├── src/           # 测试代码
│   └── test_algorithms.py  # 算法测试脚本
├── data/          # 测试数据
│   ├── images/    # 测试图像（可选）
│   └── results/   # 测试结果输出
└── README.md      # 本文件
```

## 运行测试

### 基本测试
```bash
# 从项目根目录运行
python tests/src/test_algorithms.py
```

### 测试功能
- 米尺检测算法测试
- 土壤分割算法测试
- 可视化结果生成
- 批量图像处理
- 性能统计

### 测试结果
结果会保存在 `tests/data/results/` 目录中：
- `*_results.png` - 4合1可视化结果
- `*_soil_mask.png` - 土壤掩码
- `*_processed.png` - 处理后图像

## 测试图像
测试脚本会在以下目录寻找图像：
1. `data/raw/` - 原始图像
2. `data/processed/` - 已处理图像
3. `tests/data/images/` - 专门的测试图像

你可以在 `tests/data/images/` 目录放置专门的测试图像。