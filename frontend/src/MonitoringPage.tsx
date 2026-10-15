import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BadgeCheck,
  BellRing,
  Check,
  CircleAlert,
  Clock3,
  Database,
  Eye,
  Gauge,
  History,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { api, ApiError } from "./api";
import type {
  DatasetItem,
  DatasetList,
  DeploymentStatus,
  MonitoringReport,
  MonitoringReportSummary,
  User,
} from "./api";
import "./monitoring.css";

function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value))
    return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function datasetLabel(
  item: DatasetItem | undefined,
  datasetName: string | null,
) {
  if (item) return item.display_name;
  return datasetName
    ? `Uploaded batch ${datasetName.slice(0, 8)}`
    : "Unknown batch";
}

function DriftStatus({ detected }: { detected: boolean }) {
  return (
    <span className={`monitoring-status ${detected ? "alert" : "normal"}`}>
      <span /> {detected ? "Drift detected" : "Within threshold"}
    </span>
  );
}

function MetricTable({
  title,
  subtitle,
  metricName,
  values,
  report,
}: {
  title: string;
  subtitle: string;
  metricName: string;
  values: Record<string, number | null>;
  report: MonitoringReport;
}) {
  const rows = Object.entries(values).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  return (
    <article className="panel monitoring-metric-panel">
      <header className="panel-header">
        <div>
          <span className="eyebrow">{metricName} CHECKS</span>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <Activity size={20} />
      </header>
      {!rows.length ? (
        <div className="monitoring-empty-inline">
          No compatible {metricName} features were present in the saved
          baseline.
        </div>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>FEATURE</th>
                <th>{metricName}</th>
                <th>THRESHOLD</th>
                <th>RESULT</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([column, value]) => {
                const drifted = report.report.drifted_columns.includes(column);
                const missing = report.report.missing_columns.includes(column);
                return (
                  <tr key={column} className={drifted ? "drift-row" : ""}>
                    <td>
                      <strong>{column}</strong>
                      {missing && <small>Missing from current batch</small>}
                    </td>
                    <td>{formatScore(value)}</td>
                    <td>{formatScore(report.report.threshold)}</td>
                    <td>
                      <span
                        className={`metric-result ${drifted ? "alert" : "normal"}`}
                      >
                        {drifted ? "Investigate" : "Normal"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

export function MonitoringPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const canMonitor = user.permissions.includes("monitoring:write");
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [deployment, setDeployment] = useState<DeploymentStatus | null>(null);
  const [history, setHistory] = useState<MonitoringReportSummary[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedReportId, setSelectedReportId] = useState("");
  const [selectedReport, setSelectedReport] = useState<MonitoringReport | null>(
    null,
  );
  const [threshold, setThreshold] = useState("0.20");
  const [loading, setLoading] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const handleError = useCallback(
    (cause: unknown, fallback: string) => {
      if (cause instanceof ApiError && cause.status === 401) {
        onSessionExpired();
        return;
      }
      setError(cause instanceof Error ? cause.message : fallback);
    },
    [onSessionExpired],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [datasetResponse, deploymentResponse, reports] = await Promise.all([
        api<DatasetList>("/datasets?limit=50"),
        api<DeploymentStatus>("/deployments"),
        api<MonitoringReportSummary[]>("/monitoring/reports"),
      ]);
      setDatasets(datasetResponse.items);
      setDeployment(deploymentResponse);
      setHistory(reports);
      setSelectedDataset((current) =>
        datasetResponse.items.some((item) => item.dataset_name === current)
          ? current
          : datasetResponse.items[0]?.dataset_name || "",
      );
      setSelectedReportId((current) =>
        reports.some((report) => report.report_id === current)
          ? current
          : reports[0]?.report_id || "",
      );
    } catch (cause) {
      handleError(cause, "Monitoring data could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  const loadReport = useCallback(
    async (reportId: string) => {
      if (!reportId) {
        setSelectedReport(null);
        return;
      }
      setLoadingReport(true);
      try {
        setSelectedReport(
          await api<MonitoringReport>(`/monitoring/reports/${reportId}`),
        );
      } catch (cause) {
        handleError(
          cause,
          "The selected monitoring report could not be loaded.",
        );
      } finally {
        setLoadingReport(false);
      }
    },
    [handleError],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    void loadReport(selectedReportId);
  }, [loadReport, selectedReportId]);

  const datasetByName = useMemo(
    () => new Map(datasets.map((item) => [item.dataset_name, item])),
    [datasets],
  );
  const driftAlerts = history.filter((report) => report.drift_detected).length;
  const active = deployment?.active || null;
  const hasDeployment = deployment?.state === "ready" && Boolean(active);

  async function runCheck() {
    const parsedThreshold = Number(threshold);
    if (
      !selectedDataset ||
      !Number.isFinite(parsedThreshold) ||
      parsedThreshold <= 0 ||
      parsedThreshold > 1
    ) {
      setError(
        "Select an uploaded batch and use a threshold greater than 0 and at most 1.",
      );
      return;
    }
    if (!canMonitor || !hasDeployment) return;
    setRunning(true);
    setError("");
    try {
      const report = await api<MonitoringReport>("/monitor/drift", {
        method: "POST",
        body: JSON.stringify({
          current_dataset_name: selectedDataset,
          drift_threshold: parsedThreshold,
        }),
      });
      setSelectedReport(report);
      setSelectedReportId(report.report_id);
      await refresh();
    } catch (cause) {
      handleError(cause, "The drift check could not be completed.");
    } finally {
      setRunning(false);
    }
  }

  const selectedSummary = history.find(
    (item) => item.report_id === selectedReportId,
  );
  const reportDataset = selectedReport
    ? datasetLabel(
        datasetByName.get(selectedReport.dataset_name || ""),
        selectedReport.dataset_name,
      )
    : "No report selected";

  return (
    <div className="monitoring-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STAGE 08 · MONITORING & DRIFT</div>
          <h1>Observe change. Keep the release decision human.</h1>
          <p>
            Compare incoming feature batches with the deployed model baseline
            and record a controlled follow-up when drift appears.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading || running}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          monitoring
        </button>
      </div>

      <section
        className={`monitoring-hero ${hasDeployment ? "ready" : "empty"}`}
      >
        <span className="monitoring-hero-icon">
          <Gauge size={24} />
        </span>
        <div>
          <span className="eyebrow">FEATURE DRIFT WATCH</span>
          <h2>
            {hasDeployment
              ? "A production baseline is ready to compare."
              : "Monitoring needs a deployed model baseline."}
          </h2>
          <p>
            {hasDeployment
              ? `${active?.deployment?.version || "Active release"} is monitored with saved numeric and categorical feature summaries.`
              : "Deploy a reviewer-approved candidate first. The monitor intentionally will not compare an arbitrary dataset without a model baseline."}
          </p>
        </div>
        <span
          className={`monitoring-state ${hasDeployment ? "ready" : "empty"}`}
        >
          <span /> {hasDeployment ? "Baseline ready" : "Waiting for release"}
        </span>
      </section>

      {!canMonitor && (
        <section className="monitoring-readonly">
          <Eye size={17} /> You can inspect monitoring evidence and alerts. Only
          an MLOps Administrator can run a new drift check.
        </section>
      )}
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section
        className="monitoring-status-grid"
        aria-label="Monitoring status"
      >
        <article>
          <span className="monitoring-stat-icon blue">
            <Gauge size={18} />
          </span>
          <small>Monitored release</small>
          <strong>{active?.deployment?.version || "None"}</strong>
          <p>
            {hasDeployment
              ? "Feature baseline connected"
              : "No active baseline"}
          </p>
        </article>
        <article>
          <span className="monitoring-stat-icon teal">
            <History size={18} />
          </span>
          <small>Recorded checks</small>
          <strong>{history.length}</strong>
          <p>Persisted drift evidence</p>
        </article>
        <article>
          <span className="monitoring-stat-icon amber">
            <BellRing size={18} />
          </span>
          <small>Drift alerts</small>
          <strong>{driftAlerts}</strong>
          <p>Require investigation</p>
        </article>
        <article>
          <span className="monitoring-stat-icon purple">
            <Target size={18} />
          </span>
          <small>Outcome feedback</small>
          <strong>Not configured</strong>
          <p>No labels or production KPIs yet</p>
        </article>
      </section>

      <section className="monitoring-layout">
        <article className="panel monitoring-run-panel">
          <header className="panel-header">
            <div>
              <span className="eyebrow">CONTROLLED CHECK</span>
              <h2>Compare an uploaded data batch</h2>
              <p>
                Runs PSI for numeric features and TVD for categorical features
                against the active release baseline.
              </p>
            </div>
            <SlidersHorizontal size={20} />
          </header>
          <div className="monitoring-run-body">
            <label>
              <span>Incoming dataset batch</span>
              <select
                value={selectedDataset}
                onChange={(event) => setSelectedDataset(event.target.value)}
                disabled={
                  !canMonitor || !hasDeployment || running || !datasets.length
                }
              >
                {!datasets.length && (
                  <option value="">No uploaded datasets available</option>
                )}
                {datasets.map((dataset) => (
                  <option
                    key={dataset.dataset_name}
                    value={dataset.dataset_name}
                  >
                    {dataset.display_name} · {dataset.rows ?? "—"} rows
                  </option>
                ))}
              </select>
            </label>
            <label className="threshold-field">
              <span>
                Drift threshold <small>0–1</small>
              </span>
              <input
                type="number"
                min="0.01"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
                disabled={!canMonitor || !hasDeployment || running}
              />
            </label>
            <button
              className="primary-button monitoring-run-button"
              onClick={() => void runCheck()}
              disabled={
                !canMonitor || !hasDeployment || !selectedDataset || running
              }
            >
              {running ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Activity size={16} />
              )}{" "}
              {running ? "Checking batch" : "Run drift check"}
            </button>
          </div>
          <div className="monitoring-guardrail">
            <ShieldCheck size={16} /> A check writes evidence and may request
            retraining. It never starts training, promotes a model, or changes
            the endpoint.
          </div>
        </article>

        <article className="panel monitoring-scope-panel">
          <header className="panel-header">
            <div>
              <span className="eyebrow">HONEST SCOPE</span>
              <h2>What this prototype observes</h2>
            </div>
            <ShieldCheck size={20} />
          </header>
          <ul className="monitoring-scope-list">
            <li>
              <span className="scope-check">
                <Check size={13} />
              </span>
              <div>
                <strong>Data / feature drift</strong>
                <p>
                  Numeric PSI and categorical TVD, compared with the model’s
                  saved training baseline.
                </p>
              </div>
            </li>
            <li>
              <span className="scope-check">
                <Check size={13} />
              </span>
              <div>
                <strong>Missing expected features</strong>
                <p>
                  A missing monitored column is treated as a drift signal and
                  displayed in the report.
                </p>
              </div>
            </li>
            <li>
              <span className="scope-pending">
                <Clock3 size={13} />
              </span>
              <div>
                <strong>Prediction drift & model performance</strong>
                <p>
                  Not measured yet: they require stored production predictions
                  and delayed ground-truth outcomes.
                </p>
              </div>
            </li>
          </ul>
        </article>
      </section>

      <section className="monitoring-section-heading">
        <div>
          <span className="eyebrow">ALERT EVIDENCE</span>
          <h2>Monitoring reports</h2>
        </div>
        <span>
          {history.length} saved {history.length === 1 ? "check" : "checks"}
        </span>
      </section>
      <section className="monitoring-report-layout">
        <article className="panel monitoring-history-panel">
          <header className="panel-header">
            <div>
              <h2>Recent batch comparisons</h2>
              <p>Select a saved report to inspect its metric tables.</p>
            </div>
            <History size={20} />
          </header>
          {!history.length ? (
            <div className="monitoring-history-empty">
              <History size={22} /> Your recorded feature-drift checks will
              appear here.
            </div>
          ) : (
            <div className="monitoring-history-list">
              {history.map((report) => (
                <button
                  key={report.report_id}
                  className={`monitoring-history-row ${selectedReportId === report.report_id ? "selected" : ""}`}
                  onClick={() => setSelectedReportId(report.report_id)}
                >
                  <span
                    className={`history-alert ${report.drift_detected ? "alert" : "normal"}`}
                  >
                    {report.drift_detected ? (
                      <AlertTriangle size={15} />
                    ) : (
                      <BadgeCheck size={15} />
                    )}
                  </span>
                  <span>
                    <strong>
                      {datasetLabel(
                        datasetByName.get(report.dataset_name || ""),
                        report.dataset_name,
                      )}
                    </strong>
                    <small>
                      {formatDate(report.created_at)} ·{" "}
                      {report.monitored_by || "Unknown actor"}
                    </small>
                  </span>
                  <DriftStatus detected={report.drift_detected} />
                </button>
              ))}
            </div>
          )}
        </article>

        <article className="panel monitoring-detail-panel">
          {loadingReport ? (
            <div className="monitoring-detail-loading">
              <LoaderCircle className="spin" size={18} /> Loading report
              evidence…
            </div>
          ) : selectedReport ? (
            <>
              <header className="panel-header">
                <div>
                  <span className="eyebrow">SELECTED REPORT</span>
                  <h2>{reportDataset}</h2>
                  <p>
                    {formatDate(selectedReport.created_at)} ·{" "}
                    {selectedReport.monitored_by || "Unknown actor"} ·{" "}
                    {selectedReport.model_version || "Release not recorded"}
                  </p>
                </div>
                <DriftStatus detected={selectedReport.report.drift_detected} />
              </header>
              <div
                className={`monitoring-outcome ${selectedReport.report.drift_detected ? "alert" : "normal"}`}
              >
                {selectedReport.report.drift_detected ? (
                  <AlertTriangle size={21} />
                ) : (
                  <BadgeCheck size={21} />
                )}
                <div>
                  <strong>
                    {selectedReport.report.drift_detected
                      ? "Investigation is required."
                      : "No feature exceeded the configured threshold."}
                  </strong>
                  <p>
                    {selectedReport.report.drifted_columns.length
                      ? `Flagged: ${selectedReport.report.drifted_columns.join(", ")}.`
                      : "The compared features remain within the chosen feature-drift threshold."}
                  </p>
                </div>
              </div>
              <div className="monitoring-detail-facts">
                <span>
                  <small>THRESHOLD</small>
                  <strong>
                    {formatScore(selectedReport.report.threshold)}
                  </strong>
                </span>
                <span>
                  <small>DRIFTED FEATURES</small>
                  <strong>
                    {selectedReport.report.drifted_columns.length}
                  </strong>
                </span>
                <span>
                  <small>MISSING FEATURES</small>
                  <strong>
                    {selectedReport.report.missing_columns.length}
                  </strong>
                </span>
              </div>
            </>
          ) : (
            <div className="monitoring-detail-loading">
              <Database size={21} /> Select a saved check to inspect its
              evidence.
            </div>
          )}
        </article>
      </section>

      {selectedReport && (
        <>
          <section className="monitoring-metric-grid">
            <MetricTable
              title="Numeric feature drift"
              subtitle="Population Stability Index (PSI) by monitored numeric feature."
              metricName="PSI"
              values={selectedReport.report.numeric_psi}
              report={selectedReport}
            />
            <MetricTable
              title="Categorical feature drift"
              subtitle="Total Variation Distance (TVD) by monitored categorical feature."
              metricName="TVD"
              values={selectedReport.report.categorical_tvd}
              report={selectedReport}
            />
          </section>
          <section
            className={`monitoring-retraining ${selectedReport.report.drift_detected ? "alert" : "normal"}`}
          >
            <span>
              {selectedReport.report.drift_detected ? (
                <AlertTriangle size={21} />
              ) : (
                <BadgeCheck size={21} />
              )}
            </span>
            <div>
              <span className="eyebrow">CONTROLLED RETRAINING GATE</span>
              <h3>
                {selectedReport.retraining_request.state ===
                "retraining_requested_pending_human_approval"
                  ? "Retraining requested — awaiting human approval."
                  : "No retraining request was created."}
              </h3>
              <p>{selectedReport.retraining_request.reason}</p>
            </div>
            <a className="text-button" href="#planning">
              Return to planning <ArrowRight size={14} />
            </a>
          </section>
        </>
      )}
      {selectedSummary && (
        <span className="sr-only">
          Selected monitoring report {selectedSummary.report_id}
        </span>
      )}
    </div>
  );
}
