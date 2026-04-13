import { Button, Card, Input, Select, Space, Switch, Table, Typography } from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";

type OrderRow = {
  order_id: string;
  line?: string | null;
  slot_no?: number | null;
  candidate_position?: number | null;
  tons?: number | null;
  width?: number | null;
  thickness?: number | null;
  steel_group?: string | null;
  line_capability?: string | null;
  candidate_status?: string | null;
  drop_flag?: boolean | null;
  slot_unroutable_flag?: boolean | null;
  slot_route_risk_score?: number | null;
  analysis_only?: boolean | null;
};

type OrdersResp = {
  items: OrderRow[];
  total: number;
  page: number;
  page_size: number;
};

export function RunOrdersPage() {
  const { runId } = useParams();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<OrdersResp | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [line, setLine] = useState<string | undefined>(undefined);
  const [dropOnly, setDropOnly] = useState(false);
  const [searchOrderId, setSearchOrderId] = useState("");

  async function load() {
    if (!runId) return;
    setLoading(true);
    try {
      const resp = await api.get<OrdersResp>(`/runs/${runId}/orders`, {
        params: {
          page,
          page_size: pageSize,
          line,
          drop_flag: dropOnly ? true : undefined,
          order_id: searchOrderId || undefined,
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
  }, [runId, page, pageSize]);

  const columns: ColumnsType<OrderRow> = useMemo(
    () => [
      { title: "order_id", dataIndex: "order_id", render: (v) => <Typography.Text className="mono">{v}</Typography.Text> },
      { title: "line", dataIndex: "line" },
      { title: "slot_no", dataIndex: "slot_no" },
      { title: "pos", dataIndex: "candidate_position" },
      { title: "tons", dataIndex: "tons" },
      { title: "width", dataIndex: "width" },
      { title: "thickness", dataIndex: "thickness" },
      { title: "steel_group", dataIndex: "steel_group" },
      { title: "capability", dataIndex: "line_capability" },
      { title: "status", dataIndex: "candidate_status" },
      { title: "dropped", dataIndex: "drop_flag" },
      { title: "slot_unroutable", dataIndex: "slot_unroutable_flag" },
      { title: "slot_risk", dataIndex: "slot_route_risk_score" },
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
          <Link to={`/runs/${runId}`}>
            <Button>返回总览</Button>
          </Link>
          <Select
            allowClear
            placeholder="line"
            value={line}
            style={{ width: 160 }}
            options={[
              { value: "big_roll", label: "big_roll" },
              { value: "small_roll", label: "small_roll" },
            ]}
            onChange={(v) => setLine(v)}
          />
          <Input
            placeholder="搜索 order_id"
            value={searchOrderId}
            onChange={(e) => setSearchOrderId(e.target.value)}
            style={{ width: 260 }}
          />
          <Space>
            <span>只看剔除</span>
            <Switch checked={dropOnly} onChange={setDropOnly} />
          </Space>
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

      <Card title="订单明细（候选/正式）">
        <Table rowKey="order_id" loading={loading} columns={columns} dataSource={data?.items ?? []} pagination={pagination} size="small" />
      </Card>
    </Space>
  );
}

