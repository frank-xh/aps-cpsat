import { Layout, Menu, Typography } from "antd";
import { BarChartOutlined, DatabaseOutlined } from "@ant-design/icons";
import { Link, Route, Routes, useLocation } from "react-router-dom";

import { RunsListPage } from "./pages/RunsListPage";
import { RunOverviewPage } from "./pages/RunOverviewPage";
import { RunOrdersPage } from "./pages/RunOrdersPage";
import { RunSlotsPage } from "./pages/RunSlotsPage";
import { RunViolationsPage } from "./pages/RunViolationsPage";
import { RunChartsPage } from "./pages/RunChartsPage";
import { CompareRunsPage } from "./pages/CompareRunsPage";

const { Header, Sider, Content } = Layout;

function useSelectedKeys(): string[] {
  const loc = useLocation();
  if (loc.pathname.startsWith("/compare")) return ["compare"];
  return ["runs"];
}

export default function App() {
  const selectedKeys = useSelectedKeys();
  return (
    <Layout style={{ minHeight: "100%" }}>
      <Sider width={220} theme="light" style={{ borderRight: "1px solid #eee" }}>
        <div style={{ padding: 16 }}>
          <Typography.Title level={5} style={{ margin: 0 }}>
            APS 分析平台
          </Typography.Title>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            本地 MySQL + FastAPI + React
          </Typography.Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={selectedKeys}
          items={[
            {
              key: "runs",
              icon: <DatabaseOutlined />,
              label: <Link to="/runs">运行列表</Link>,
            },
            {
              key: "compare",
              icon: <BarChartOutlined />,
              label: <Link to="/compare">运行对比</Link>,
            },
          ]}
        />
      </Sider>
      <Layout>
        <Header style={{ background: "#fff", borderBottom: "1px solid #eee" }}>
          <Typography.Text>
            后端 API: <span className="mono">{import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:18080"}</span>
          </Typography.Text>
        </Header>
        <Content style={{ padding: 16 }}>
          <Routes>
            <Route path="/" element={<RunsListPage />} />
            <Route path="/runs" element={<RunsListPage />} />
            <Route path="/runs/:runId" element={<RunOverviewPage />} />
            <Route path="/runs/:runId/orders" element={<RunOrdersPage />} />
            <Route path="/runs/:runId/slots" element={<RunSlotsPage />} />
            <Route path="/runs/:runId/violations" element={<RunViolationsPage />} />
            <Route path="/runs/:runId/charts" element={<RunChartsPage />} />
            <Route path="/compare" element={<CompareRunsPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}

