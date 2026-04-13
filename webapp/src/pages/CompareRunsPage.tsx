import { Button, Card, Input, Space, Table, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useMemo, useState } from "react";

import { api } from "../api/client";

type CompareRow = {
  id: number;
  run_code: string;
  profile_name?: string | null;
  acceptance?: string | null;
  failure_mode?: string | null;
  routing_feasible?: boolean | null;
  scheduled_orders?: number | null;
  unscheduled_orders?: number | null;
  dropped_orders?: number | null;
  total_run_seconds?: number | null;
  template_build_seconds?: number | null;
  joint_master_seconds?: number | null;
  fallback_total_seconds?: number | null;
  violations?: Record<string, unknown> | null;
  bridge?: Record<string, unknown> | null;
};

export function CompareRunsPage() {
  const [runIds, setRunIds] = useState("1,2");
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<CompareRow[]>([]);

  const columns: ColumnsType<CompareRow> = useMemo(
    () => [
      { title: "id", dataIndex: "id" },
      { title: "run_code", dataIndex: "run_code", render: (v) => <Typography.Text className="mono">{v}</Typography.Text> },
      { title: "profile", dataIndex: "profile_name" },
      { title: "acceptance", dataIndex: "acceptance" },
      { title: "failure_mode", dataIndex: "failure_mode" },
      { title: "routing_feasible", dataIndex: "routing_feasible" },
      { title: "scheduled_orders", dataIndex: "scheduled_orders" },
      { title: "unscheduled_orders", dataIndex: "unscheduled_orders" },
      { title: "dropped_orders", dataIndex: "dropped_orders" },
      { title: "total_run_seconds", dataIndex: "total_run_seconds" },
      { title: "bridge", dataIndex: "bridge", render: (v) => (v ? JSON.stringify(v) : "") },
      { title: "violations", dataIndex: "violations", render: (v) => (v ? JSON.stringify(v) : "") },
    ],
    []
  );

  async function load() {
    const ids = runIds
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => Number(s))
      .filter((n) => Number.isFinite(n));
    if (!ids.length) return;
    setLoading(true);
    try {
      const params = new URLSearchParams();
      for (const id of ids) params.append("run_id", String(id));
      const resp = await api.get(`/compare/runs?${params.toString()}`);
      setRows(resp.data.items ?? []);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card>
        <Space wrap>
          <Input value={runIds} onChange={(e) => setRunIds(e.target.value)} style={{ width: 360 }} />
          <Button type="primary" onClick={() => void load()} loading={loading}>
            对比
          </Button>
        </Space>
      </Card>
      <Card title="运行对比">
        <Table rowKey="id" loading={loading} columns={columns} dataSource={rows} pagination={false} size="small" />
      </Card>
    </Space>
  );
}

