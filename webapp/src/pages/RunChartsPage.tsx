import { Button, Card, Empty, Select, Space } from "antd";
import ReactECharts from "echarts-for-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api/client";

type SlotRow = { line: string; slot_no: number };
type SlotsResp = { items: SlotRow[] };
type MetricRow = {
  line: string;
  slot_no: number;
  seq_no: number;
  width_jump?: number | null;
  thickness_jump?: number | null;
  temp_jump?: number | null;
  reverse_flag?: boolean | null;
};
type MetricsResp = { items: MetricRow[] };

export function RunChartsPage() {
  const { runId } = useParams();
  const [slots, setSlots] = useState<SlotRow[]>([]);
  const [line, setLine] = useState<string | undefined>("big_roll");
  const [slotNo, setSlotNo] = useState<number | undefined>(undefined);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [loading, setLoading] = useState(false);

  async function loadSlots() {
    if (!runId) return;
    const resp = await api.get<SlotsResp>(`/runs/${runId}/slots`, { params: { page: 1, page_size: 1000 } });
    const items = resp.data.items ?? [];
    setSlots(items);
    if (items.length && slotNo == null) {
      const first = items.find((s) => s.line === (line ?? "big_roll")) ?? items[0];
      setLine(first.line);
      setSlotNo(first.slot_no);
    }
  }

  async function loadMetrics() {
    if (!runId || !line || slotNo == null) return;
    setLoading(true);
    try {
      const resp = await api.get<MetricsResp>(`/runs/${runId}/transition-metrics`, { params: { line, slot_no: slotNo, limit: 50000 } });
      setMetrics(resp.data.items ?? []);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadSlots();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  useEffect(() => {
    void loadMetrics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, line, slotNo]);

  const slotOptions = useMemo(() => {
    const filtered = slots.filter((s) => !line || s.line === line);
    return filtered.map((s) => ({ value: s.slot_no, label: String(s.slot_no) }));
  }, [slots, line]);

  const x = metrics.map((m) => m.seq_no);
  const widthJump = metrics.map((m) => m.width_jump ?? null);
  const thkJump = metrics.map((m) => m.thickness_jump ?? null);
  const tempJump = metrics.map((m) => m.temp_jump ?? null);
  const reverse = metrics.map((m) => (m.reverse_flag ? 1 : 0));

  const widthOption = useMemo(
    () => ({
      title: { text: "宽度跳变 (abs)" },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: x },
      yAxis: { type: "value" },
      series: [{ type: "line", data: widthJump, smooth: true }],
    }),
    [x, widthJump]
  );

  const thkOption = useMemo(
    () => ({
      title: { text: "厚度跳变 (abs)" },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: x },
      yAxis: { type: "value" },
      series: [{ type: "line", data: thkJump, smooth: true }],
    }),
    [x, thkJump]
  );

  const reverseOption = useMemo(
    () => ({
      title: { text: "逆宽标记 (to_width < from_width)" },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: x },
      yAxis: { type: "value", min: 0, max: 1 },
      series: [{ type: "bar", data: reverse }],
    }),
    [x, reverse]
  );

  const tempOption = useMemo(
    () => ({
      title: { text: "温度跳变 (abs mid diff)" },
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: x },
      yAxis: { type: "value" },
      series: [{ type: "line", data: tempJump, smooth: true }],
    }),
    [x, tempJump]
  );

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card>
        <Space wrap>
          <Link to={`/runs/${runId}`}>
            <Button>返回总览</Button>
          </Link>
          <Select
            value={line}
            style={{ width: 160 }}
            options={[
              { value: "big_roll", label: "big_roll" },
              { value: "small_roll", label: "small_roll" },
            ]}
            onChange={(v) => {
              setLine(v);
              setSlotNo(undefined);
            }}
          />
          <Select value={slotNo} style={{ width: 120 }} options={slotOptions} onChange={(v) => setSlotNo(v)} />
          <Button onClick={() => void loadMetrics()} loading={loading}>
            刷新图表
          </Button>
        </Space>
      </Card>

      {metrics.length === 0 ? (
        <Card>
          <Empty description="没有 transition-metrics 数据（可能未导入或该 run 无候选/正式顺序）" />
        </Card>
      ) : (
        <>
          <Card>
            <ReactECharts option={widthOption} style={{ height: 320 }} />
          </Card>
          <Card>
            <ReactECharts option={thkOption} style={{ height: 320 }} />
          </Card>
          <Card>
            <ReactECharts option={reverseOption} style={{ height: 280 }} />
          </Card>
          <Card>
            <ReactECharts option={tempOption} style={{ height: 280 }} />
          </Card>
        </>
      )}
    </Space>
  );
}
