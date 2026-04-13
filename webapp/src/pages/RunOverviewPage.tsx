import { Alert, Button, Card, Descriptions, Space, Table, Tag, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import type { LineSummaryRow, ScheduleRun, ViolationSummary } from "../api/types";

type RunDetailResp = {
  run: ScheduleRun;
  violation_summary?: ViolationSummary | null;
  bridge_drop_summary?: Record<string, unknown> | null;
  line_summary?: LineSummaryRow[];
};

export function RunOverviewPage() {
  const { runId } = useParams();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<RunDetailResp | null>(null);

  async function load() {
    if (!runId) return;
    setLoading(true);
    try {
      const resp = await api.get<RunDetailResp>(`/runs/${runId}`);
      setData(resp.data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const lineCols: ColumnsType<LineSummaryRow> = useMemo(
    () => [
      { title: "line", dataIndex: "line" },
      { title: "assigned_orders", dataIndex: "assigned_orders" },
      { title: "assigned_tons", dataIndex: "assigned_tons", render: (v) => (v == null ? "" : v.toFixed?.(3) ?? v) },
      { title: "slot_count", dataIndex: "slot_count" },
      { title: "avg_slot_order_count", dataIndex: "avg_slot_order_count", render: (v) => (v == null ? "" : v.toFixed?.(2) ?? v) },
      { title: "max_slot_order_count", dataIndex: "max_slot_order_count" },
      { title: "unroutable_slot_count", dataIndex: "unroutable_slot_count" },
      { title: "dropped_orders", dataIndex: "dropped_orders" },
    ],
    []
  );

  const run = data?.run;
  const vio = data?.violation_summary;

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card>
        <Space wrap>
          <Link to={`/runs/${runId}`}>
            <Button type="primary">总览</Button>
          </Link>
          <Link to={`/runs/${runId}/orders`}>
            <Button>订单明细</Button>
          </Link>
          <Link to={`/runs/${runId}/slots`}>
            <Button>槽位分析</Button>
          </Link>
          <Link to={`/runs/${runId}/violations`}>
            <Button>违规分析</Button>
          </Link>
          <Link to={`/runs/${runId}/charts`}>
            <Button>图表分析</Button>
          </Link>
        </Space>
      </Card>

      <Card loading={loading} title="运行详情">
        {!run ? (
          <Alert type="warning" message="未找到该 run" />
        ) : (
          <>
            <Space direction="vertical" style={{ width: "100%" }} size={16}>
              <Space wrap>
                {run.analysis_only ? <Tag color="orange">ANALYSIS_ONLY</Tag> : <Tag color="green">OFFICIAL</Tag>}
                <Tag color={run.routing_feasible ? "green" : "red"}>{String(run.routing_feasible)}</Tag>
                <Typography.Text className="mono">{run.run_code}</Typography.Text>
              </Space>

              <Descriptions bordered size="small" column={2}>
                <Descriptions.Item label="profile">{run.profile_name}</Descriptions.Item>
                <Descriptions.Item label="acceptance">{run.acceptance}</Descriptions.Item>
                <Descriptions.Item label="failure_mode">{run.failure_mode}</Descriptions.Item>
                <Descriptions.Item label="evidence_level">{run.evidence_level}</Descriptions.Item>
                <Descriptions.Item label="scheduled_orders">{run.scheduled_orders}</Descriptions.Item>
                <Descriptions.Item label="unscheduled_orders">{run.unscheduled_orders}</Descriptions.Item>
                <Descriptions.Item label="dropped_orders">{run.dropped_orders}</Descriptions.Item>
                <Descriptions.Item label="total_run_seconds">{run.total_run_seconds}</Descriptions.Item>
                <Descriptions.Item label="template_build_seconds">{run.template_build_seconds}</Descriptions.Item>
                <Descriptions.Item label="joint_master_seconds">{run.joint_master_seconds}</Descriptions.Item>
                <Descriptions.Item label="fallback_total_seconds">{run.fallback_total_seconds}</Descriptions.Item>
                <Descriptions.Item label="best_candidate_type">{run.best_candidate_type}</Descriptions.Item>
              </Descriptions>

              <Card size="small" title="违规汇总（run 级）">
                <Space wrap>
                  <Tag>direct_reverse: {String(vio?.direct_reverse_step_violation_count ?? "")}</Tag>
                  <Tag>virtual_attach_reverse: {String(vio?.virtual_attach_reverse_violation_count ?? "")}</Tag>
                  <Tag>period_reverse: {String(vio?.period_reverse_count_violation_count ?? "")}</Tag>
                  <Tag>invalid_virtual_spec: {String(vio?.invalid_virtual_spec_count ?? "")}</Tag>
                  <Tag color="red">unroutable_slots: {String(vio?.candidate_unroutable_slot_count ?? "")}</Tag>
                </Space>
              </Card>

              <Card size="small" title="产线汇总（由 order_result 聚合）">
                <Table rowKey="line" columns={lineCols} dataSource={data?.line_summary ?? []} pagination={false} size="small" />
              </Card>
            </Space>
          </>
        )}
      </Card>
    </Space>
  );
}

