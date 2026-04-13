import { Button, Card, Input, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/client";
import type { ScheduleRun } from "../api/types";

type RunsResp = {
  items: ScheduleRun[];
  total: number;
  page: number;
  page_size: number;
};

export function RunsListPage() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<RunsResp | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [profile, setProfile] = useState<string>("");
  const [acceptance, setAcceptance] = useState<string>("");
  const [failureMode, setFailureMode] = useState<string>("");

  async function load() {
    setLoading(true);
    try {
      const resp = await api.get<RunsResp>("/runs", {
        params: {
          page,
          page_size: pageSize,
          profile: profile || undefined,
          acceptance: acceptance || undefined,
          failure_mode: failureMode || undefined,
          sort_by: "created_at",
          sort_order: "desc",
        },
      });
      setData(resp.data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, pageSize]);

  const columns: ColumnsType<ScheduleRun> = useMemo(
    () => [
      {
        title: "run_code",
        dataIndex: "run_code",
        render: (v: string, r) => (
          <Space>
            <Link to={`/runs/${r.id}`}>
              <Typography.Text className="mono">{v}</Typography.Text>
            </Link>
            {r.analysis_only ? <Tag color="orange">ANALYSIS</Tag> : <Tag color="green">OFFICIAL</Tag>}
          </Space>
        ),
      },
      { title: "profile", dataIndex: "profile_name" },
      { title: "acceptance", dataIndex: "acceptance" },
      { title: "failure_mode", dataIndex: "failure_mode" },
      { title: "scheduled_orders", dataIndex: "scheduled_orders" },
      { title: "dropped_orders", dataIndex: "dropped_orders" },
      { title: "total_run_seconds", dataIndex: "total_run_seconds", render: (v) => (v == null ? "" : v.toFixed?.(3) ?? v) },
      { title: "created_at", dataIndex: "created_at" },
      {
        title: "actions",
        key: "actions",
        render: (_, r) => (
          <Space>
            <Link to={`/runs/${r.id}`}>
              <Button size="small">详情</Button>
            </Link>
          </Space>
        ),
      },
    ],
    []
  );

  const pagination: TablePaginationConfig = {
    current: page,
    pageSize,
    total: data?.total ?? 0,
    showSizeChanger: true,
    onChange: (p, ps) => {
      setPage(p);
      if (ps !== pageSize) setPageSize(ps);
    },
  };

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card>
        <Space wrap>
          <Input placeholder="profile" value={profile} onChange={(e) => setProfile(e.target.value)} style={{ width: 220 }} />
          <Input
            placeholder="acceptance"
            value={acceptance}
            onChange={(e) => setAcceptance(e.target.value)}
            style={{ width: 240 }}
          />
          <Input
            placeholder="failure_mode"
            value={failureMode}
            onChange={(e) => setFailureMode(e.target.value)}
            style={{ width: 260 }}
          />
          <Button
            type="primary"
            onClick={() => {
              setPage(1);
              void load();
            }}
          >
            查询
          </Button>
        </Space>
      </Card>

      <Card title="排程运行列表">
        <Table
          rowKey="id"
          loading={loading}
          columns={columns}
          dataSource={data?.items ?? []}
          pagination={pagination}
          size="middle"
        />
      </Card>
    </Space>
  );
}

