import { Button, Card, Select, Space, Switch, Table } from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";

type SlotRow = {
  line: string;
  slot_no: number;
  slot_order_count?: number | null;
  slot_tons?: number | null;
  order_count_over_cap?: number | null;
  template_coverage_ratio?: number | null;
  missing_template_edge_count?: number | null;
  zero_in_orders?: number | null;
  zero_out_orders?: number | null;
  width_span?: number | null;
  thickness_span?: number | null;
  steel_group_count?: number | null;
  pair_gap_proxy?: number | null;
  span_risk?: number | null;
  degree_risk?: number | null;
  isolated_order_penalty?: number | null;
  slot_route_risk_score?: number | null;
  dominant_unroutable_reason?: string | null;
};

type SlotsResp = { items: SlotRow[]; total: number; page: number; page_size: number };

export function RunSlotsPage() {
  const { runId } = useParams();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<SlotsResp | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(200);
  const [line, setLine] = useState<string | undefined>(undefined);
  const [unroutableOnly, setUnroutableOnly] = useState(false);

  async function load() {
    if (!runId) return;
    setLoading(true);
    try {
      const resp = await api.get<SlotsResp>(`/runs/${runId}/slots`, {
        params: {
          page,
          page_size: pageSize,
          line,
          slot_unroutable_only: unroutableOnly ? true : false,
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

  const columns: ColumnsType<SlotRow> = useMemo(
    () => [
      { title: "line", dataIndex: "line" },
      { title: "slot_no", dataIndex: "slot_no" },
      { title: "orders", dataIndex: "slot_order_count" },
      { title: "tons", dataIndex: "slot_tons" },
      { title: "risk", dataIndex: "slot_route_risk_score" },
      { title: "pair_gap", dataIndex: "pair_gap_proxy" },
      { title: "span_risk", dataIndex: "span_risk" },
      { title: "degree_risk", dataIndex: "degree_risk" },
      { title: "isolated_pen", dataIndex: "isolated_order_penalty" },
      { title: "width_span", dataIndex: "width_span" },
      { title: "thk_span", dataIndex: "thickness_span" },
      { title: "group_cnt", dataIndex: "steel_group_count" },
      { title: "unroutable_reason", dataIndex: "dominant_unroutable_reason" },
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
          <Space>
            <span>只看不可路由</span>
            <Switch checked={unroutableOnly} onChange={setUnroutableOnly} />
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

      <Card title="槽位汇总">
        <Table rowKey={(r) => `${r.line}-${r.slot_no}`} loading={loading} columns={columns} dataSource={data?.items ?? []} pagination={pagination} size="small" />
      </Card>
    </Space>
  );
}

