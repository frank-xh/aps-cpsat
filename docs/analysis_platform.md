# 排程结果分析平台（本地）

本模块为 **增量分析能力**：不改变现有求解主架构，不影响 `examples/run_cold_rolling_schedule.py` 的基本行为。  
它提供：

1. MySQL 本地落库（每次运行成功/失败都可入库）
2. FastAPI 后端（查询/筛选/对比）
3. React 前端（可视化表格 + 图表）

---

## 1. 环境准备

1. 本地 MySQL：
   - host=`127.0.0.1`
   - port=`3306`
   - user=`root`
   - password 从 `.env` 读取（不要写死在代码里）

2. Python 依赖：
   - 见 `requirements.txt`（包含 fastapi/uvicorn/sqlalchemy/pymysql/python-dotenv）

3. Node.js：
   - 用于启动 `webapp/`

---

## 2. 配置

根目录提供 `.env.example`，按需在根目录放置 `.env`（不要提交到 git）：

```bash
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=aps_schedule_analysis

BACKEND_HOST=127.0.0.1
BACKEND_PORT=18080
```

---

## 3. 初始化数据库

```bash
python scripts/init_mysql.py
```

会自动建库并执行 DDL：
- `src/aps_cp_sat/persistence/ddl.sql`

---

## 4. 导入一次排程结果（分析 Excel -> MySQL）

导入 APS 导出的分析 Excel（成功或失败都可）：

```bash
python scripts/import_schedule_result.py "D:\Desktop\SAT排程结果\SAT排程结果_20260411_121650_FAILED_ROUTING_ANALYSIS.xlsx"
```

导入特性：
- 允许缺失 sheet，做部分导入
- 对 `run_code`/`result_file_path` 具备幂等更新能力（重复导入不会产生重复 run）

---

## 5. 启动后端

```bash
python scripts/start_analysis_backend.py
```

默认地址：`http://127.0.0.1:18080`  
OpenAPI：`http://127.0.0.1:18080/docs`

---

## 6. 启动前端

```bash
python scripts/start_analysis_frontend.py
```

默认 Vite 开发端口：`http://127.0.0.1:5173`

---

## 7. 与 APS 主工程的可选集成

当前提供持久化入口：
- `aps_cp_sat.persistence.service.persist_run_analysis_from_excel(xlsx_path)`

建议做法：
- APS 每次 run 结束后（已生成分析 Excel），在可控开关下调用该函数自动入库。

本平台本身不会修改求解主流程，只提供可选调用点。

