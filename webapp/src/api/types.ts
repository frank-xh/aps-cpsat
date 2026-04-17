export type ScheduleRun = {
  id: number;
  run_code: string;
  profile_name?: string | null;
  result_file_path?: string | null;
  acceptance?: string | null;
  failure_mode?: string | null;
  routing_feasible?: boolean | null;
  analysis_only?: boolean | null;
  official_exported?: boolean | null;
  analysis_exported?: boolean | null;
  total_orders?: number | null;
  scheduled_orders?: number | null;
  unscheduled_orders?: number | null;
  dropped_orders?: number | null;
  scheduled_tons?: number | null;
  unscheduled_tons?: number | null;
  template_build_seconds?: number | null;
  joint_master_seconds?: number | null;
  fallback_total_seconds?: number | null;
  total_run_seconds?: number | null;
  evidence_level?: string | null;
  best_candidate_available?: boolean | null;
  best_candidate_type?: string | null;
  best_candidate_objective?: number | null;
  best_candidate_unroutable_slot_count?: number | null;
  created_at?: string;
};

export type LineSummaryRow = {
  line: string;
  assigned_orders: number;
  assigned_tons: number;
  slot_count: number;
  avg_slot_order_count: number;
  max_slot_order_count: number;
  unroutable_slot_count: number;
  dropped_orders: number;
};

export type ViolationSummary = {
  direct_reverse_step_violation_count?: number | null;
  virtual_attach_reverse_violation_count?: number | null;
  period_reverse_count_violation_count?: number | null;
  bridge_count_violation_count?: number | null;
  invalid_virtual_spec_count?: number | null;
  candidate_unroutable_slot_count?: number | null;
  candidate_bad_slot_count?: number | null;
  candidate_zero_in_order_count?: number | null;
  candidate_zero_out_order_count?: number | null;
  hard_cap_not_enforced?: boolean | null;
};

