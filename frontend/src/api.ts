export type Role = "scientist" | "reviewer" | "admin";
export interface User {
  id: string;
  email: string;
  name: string;
  role: Role;
  role_label: string;
  permissions: string[];
}
export interface WorkspaceUser extends User {
  active: boolean;
}
export interface WorkspaceSettings {
  workspace_mode: string;
  authentication: string;
  session_hours: number;
  user_provisioning: string;
  password_management: string;
  roles: {
    role: Role;
    label: string;
    permissions: string[];
  }[];
  guardrails: string[];
}
export interface WorkspaceUsersResponse {
  items: WorkspaceUser[];
}
export interface Run {
  run_id: string;
  started_at: string | null;
  status: string | null;
  objective: string | null;
}
export interface Overview {
  runs_count: number;
  review_count: number;
  failed_count: number;
  production_available: boolean;
  recent_runs: Run[];
}
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      "X-AutoDS-Client": "web",
      ...options.headers,
    },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : "The request could not be completed. Please try again.";
    throw new ApiError(response.status, detail);
  }
  return payload as T;
}

export interface DatasetItem {
  dataset_name: string;
  display_name: string;
  uploaded_at: string | null;
  size_bytes: number;
  rows: number | null;
  columns: number | null;
  quality_status: "passed" | "warnings" | "not_profiled";
}
export interface DatasetList {
  items: DatasetItem[];
  total: number;
  offset: number;
  limit: number;
}
export interface DatasetColumn {
  name: string;
  inferred_type: string;
  pandas_dtype: string;
  unique_count: number;
  missing_count: number;
  missing_fraction: number;
  likely_identifier: boolean;
}
export interface DatasetReport {
  dataset_name: string;
  display_name: string;
  uploaded_at: string | null;
  uploaded_by: string | null;
  size_bytes: number;
  profile: {
    rows: number;
    columns: DatasetColumn[];
    data_hash: string;
    likely_targets: string[];
  };
  quality: {
    status: "passed" | "warnings";
    missing_cells: number;
    total_cells: number;
    completeness: number;
    duplicate_rows: number;
    checks: {
      code: string;
      label: string;
      status: "passed" | "warning" | "info";
      message: string;
      count: number;
      columns: string[];
    }[];
  };
}
export interface DatasetPreview {
  columns: string[];
  rows: (string | number | boolean | null)[][];
  total_rows: number;
  offset: number;
  limit: number;
  cell_character_limit: number;
}

export interface MLPlan {
  objective: string;
  target_column: string;
  task_type:
    "binary_classification" | "multiclass_classification" | "regression";
  metric: string;
  acceptance_threshold: number;
  candidate_models: string[];
  preprocessing: {
    numeric_imputation: string;
    categorical_imputation: string;
    scale_numeric: boolean;
    one_hot_encode: boolean;
    excluded_columns: string[];
  };
  max_experiments: number;
  test_size: number;
  random_state: number;
  reasoning: string[];
}
export interface AgentDecision {
  action: string;
  model_name: string | null;
  strategy: string;
  reason: string;
}
export interface Experiment {
  experiment_id: string;
  model_name: string;
  metric: string;
  score: number | null;
  metrics: Record<string, number>;
  status: string;
  elapsed_seconds: number;
  strategy: string;
  error: string | null;
}
export interface RunEvent extends Partial<AgentDecision> {
  at: string;
  kind: string;
  message?: string;
  experiment_id?: string;
  metric?: string;
  score?: number | null;
  elapsed_seconds?: number;
  error?: string | null;
  proposal_id?: string;
  model_name?: string | null;
  status?: string;
  stop_reason?: string;
  feature_count?: number;
  training_rows?: number;
  validation_rows?: number;
  holdout_rows?: number;
}
export interface RunSummary {
  run_id: string;
  status: string;
  profile: {
    rows: number;
    columns: unknown[];
    data_hash: string;
  };
  plan: MLPlan | null;
  experiments: Experiment[];
  decision_trace: AgentDecision[];
  selected_experiment_id: string | null;
  final_evaluation: Experiment | null;
  promotion_proposal_id: string | null;
  stop_reason: string;
}
export interface StoredRun {
  run_id: string;
  started_at: string;
  completed_at?: string;
  objective: string;
  dataset: {
    sha256: string;
    rows: number;
    columns: number;
  };
  plan: MLPlan | null;
  events: RunEvent[];
  experiments: Experiment[];
  final_evaluation: Experiment | null;
  status: string;
  summary?: {
    selected_experiment_id: string | null;
    final_evaluation_id: string | null;
    promotion_proposal_id: string | null;
    stop_reason: string;
  };
}
export type ProposalStatus =
  | "pending_human_approval"
  | "approved_for_deployment"
  | "rejected"
  | "deployed"
  | "promoted";
export interface InputSchemaField {
  name: string;
  inferred_type: "numeric" | "categorical" | string;
  pandas_dtype: string;
}
export interface DeploymentDetails {
  version: string;
  deployed_by: string;
  deployed_at: string;
  deployment_note: string;
  feature_columns: string[];
  input_schema: InputSchemaField[];
  task_type: string;
  metric: string;
  class_labels: string[];
}
export interface ProposalAuditEvent {
  at: string;
  action: string;
  actor: string;
  reason: string;
}
export interface ModelProposal {
  proposal_id: string;
  status: ProposalStatus | string;
  created_at: string;
  metadata: {
    run_id?: string;
    experiment_id?: string;
    final_evaluation_id?: string;
    model_name?: string;
    metric?: string;
    validation_score?: number | null;
    validation_metrics?: Record<string, number>;
    holdout_score?: number | null;
    holdout_metrics?: Record<string, number>;
    dataset_sha256?: string;
    plan?: MLPlan;
    input_schema?: InputSchemaField[];
  };
  audit_events?: ProposalAuditEvent[];
  review?: {
    decision: "approve" | "reject";
    reviewed_by: string;
    reviewed_at: string;
    reason: string;
  };
  deployment?: DeploymentDetails;
}
export interface DeploymentStatus {
  endpoint: string;
  method: "POST" | string;
  state: "ready" | "not_deployed" | string;
  active: ModelProposal | null;
  releases: ModelProposal[];
}
export interface DriftReportData {
  current_hash: string;
  numeric_psi: Record<string, number | null>;
  categorical_tvd: Record<string, number | null>;
  drifted_columns: string[];
  missing_columns: string[];
  threshold: number;
  drift_detected: boolean;
}
export interface RetrainingRequest {
  state: string;
  source_run_id: string | null;
  reason: string;
  drifted_columns?: string[];
  automatic_retraining?: boolean;
}
export interface MonitoringReportSummary {
  report_id: string;
  created_at: string | null;
  dataset_name: string | null;
  monitored_by: string | null;
  model_version: string | null;
  monitoring_scope: "feature_drift" | string;
  drift_detected: boolean;
  drifted_columns: string[];
  missing_columns: string[];
  threshold: number | null;
  retraining_state: string | null;
}
export interface MonitoringReport extends MonitoringReportSummary {
  report: DriftReportData;
  retraining_request: RetrainingRequest;
}
export interface AlertItem {
  alert_id: string;
  source: "monitoring" | "workflow" | "governance" | string;
  severity: "critical" | "warning" | "info" | string;
  state: "open" | string;
  title: string;
  message: string;
  occurred_at: string | null;
  resource_id: string;
  destination: string;
}
export interface AuditLogItem {
  event_id: string;
  source: "governance" | "monitoring" | "workflow" | "workspace" | string;
  action: string;
  occurred_at: string;
  actor: string | null;
  resource_type:
    | "model_candidate"
    | "monitoring_report"
    | "workflow_run"
    | "workspace_user"
    | string;
  resource_id: string;
  summary: string;
  reason: string | null;
  destination: string;
}
export interface AuditLogResponse {
  items: AuditLogItem[];
  total: number;
}
export interface PlanPreview {
  dataset: {
    dataset_name: string;
    rows: number;
    columns: number;
    data_hash: string;
  };
  planning_adapter: "heuristic" | "openai_structured";
  plan: MLPlan;
  initial_decision: {
    action: string;
    model_name: string | null;
    strategy: string;
    reason: string;
  };
  guardrails: string[];
}
