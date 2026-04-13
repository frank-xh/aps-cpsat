import { Button, Card, Descriptions, Space } from "antd";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";
import type { ViolationSummary } from "../api/types";

export function RunViolationsPage() {
  const { runId } = useParams();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ViolationSummary | null>(null);

  async function load() {
    if (!runId) return;
    setLoading(true);
    try {
      const resp = await api.get(`/runs/${runId}/violations`);
      setData(resp.data?.available === false ? null : resp.data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card>
        <Space wrap>
          <Link to={`/runs/${runId}`}>
            <Button>返回总览</Button>
          </Link>
          <Button type="primary" onClick={() => void load()}>
            刷新
          </Button>
        </Space>
      </Card>

      <Card title="违规汇总" loading={loading}>
        <Descriptions bordered size="small" column={2}>
          <Descriptions.Item label="direct_reverse_step_violation_count">{String(data?.direct_reverse_step_violation_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="virtual_attach_reverse_violation_count">{String(data?.virtual_attach_reverse_violation_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="period_reverse_count_violation_count">{String(data?.period_reverse_count_violation_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="bridge_count_violation_count">{String(data?.bridge_count_violation_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="invalid_virtual_spec_count">{String(data?.invalid_virtual_spec_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="candidate_unroutable_slot_count">{String(data?.candidate_unroutable_slot_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="candidate_bad_slot_count">{String(data?.candidate_bad_slot_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="candidate_zero_in_order_count">{String(data?.candidate_zero_in_order_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="candidate_zero_out_order_count">{String(data?.candidate_zero_out_order_count ?? "")}</Descriptions.Item>
          <Descriptions.Item label="hard_cap_not_enforced">{String(data?.hard_cap_not_enforced ?? "")}</Descriptions.Item>
        </Descriptions>
      </Card>
    </Space>
  );
}

