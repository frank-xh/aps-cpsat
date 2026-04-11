# APS + Google CP-SAT

这个项目已经初始化为可直接使用 Google OR-Tools 的 CP-SAT 求解器。

## 目录结构

```text
APS/
  src/aps_cp_sat/
    __init__.py
    scheduler.py
  examples/
    basic_schedule.py
  tests/
    test_scheduler.py
  requirements.txt
```

## 快速开始

1. 安装依赖：

```powershell
pip install -r requirements.txt
```

2. 运行示例：

```powershell
python examples/basic_schedule.py
```

3. 运行测试：

```powershell
pytest -q
```

