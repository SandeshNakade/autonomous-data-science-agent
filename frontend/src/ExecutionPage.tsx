import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  BadgeCheck,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleAlert,
  ClipboardCheck,
  Clock3,
  Database,
  Eye,
  FlaskConical,
  Gauge,
  LoaderCircle,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import { api, ApiError } from "./api";
import type {
  DatasetList,
  DatasetReport,
  Experiment,
  Run,
  RunEvent,
  RunSummary,
  StoredRun,
  User,
} from "./api";
import "./execution.css";

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  linear_regression: "Linear Regression",
  random_forest: "Random Forest",
  ridge: "Ridge Regression",
  xgboost: "XGBoost",
};

const METRIC_NAMES: Record<string, string> = {
  f1: "F1 score",
  f1_macro: "Macro F1",
  accuracy: "Accuracy",
  roc_auc: "ROC-AUC",
  roc_auc_ovr: "ROC-AUC (OvR)",
  r2: "R²",
  rmse: "RMSE",
};

function titleCase(value: string) {
  return value.replaceAll("_", " ");
}

function modelName(value: string | null | undefined) {
  return MODEL_NAMES[value || ""] || titleCase(value || "Not recorded");
}

function metricName(value: string | null | undefined) {
  return METRIC_NAMES[value || ""] || titleCase(value || "Metric");
}

function statusLabel(value: string | null | undefined) {
  const labels: Record<string, string> = {
    needs_review: "Awaiting approval",
    stopped: "Stopped safely",
    failed: "Needs attention",
    accepted: "Accepted",
    completed: "Completed",
    running: "Running",
  };
  return labels[value || ""] || titleCase(value || "Not recorded");
}

function scoreValue(score: number | null | undefined) {
  if (score === null || score === undefined || !Number.isFinite(score))
    return "—";
  return score.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function thresholdLabel(metric: string, value: number) {
  return metric === "rmse" ? `≤ ${value}` : `≥ ${value}`;
}

function storedFromSummary(summary: RunSummary): StoredRun {
  return {
    run_id: summary.run_id,
    started_at: "",
    objective: summary.plan?.objective || "Controlled workflow",
    dataset: {
      sha256: summary.profile.data_hash,
      rows: summary.profile.rows,
      columns: summary.profile.columns.length,
    },
    plan: summary.plan,
    events: summary.decision_trace.map((decision, index) => ({
      at: String(index),
      kind: "decision",
      message: "Agent decision recorded.",
      ...decision,
    })),
    experiments: summary.experiments,
    final_evaluation: summary.final_evaluation,
    status: summary.status,
    summary: {
      selected_experiment_id: summary.selected_experiment_id,
      final_evaluation_id: summary.final_evaluation?.experiment_id || null,
      promotion_proposal_id: summary.promotion_proposal_id,
      stop_reason: summary.stop_reason,
    },
  };
}

function EventIcon({ kind }: { kind: string }) {
  if (kind === "dataset_profiled" || kind === "data_prepared")
    return <Database size={16} />;
  if (kind === "plan_validated" || kind === "decision")
    return <BrainCircuit size={16} />;
  if (kind.includes("experiment")) return <FlaskConical size={16} />;
  if (kind.includes("holdout")) return <Gauge size={16} />;
  if (kind === "candidate_submitted") return <ClipboardCheck size={16} />;
  if (kind === "workflow_failed") return <CircleAlert size={16} />;
  if (kind === "run_completed") return <Check size={16} />;
  return <Clock3 size={16} />;
}

function eventTitle(event: RunEvent) {
  const labels: Record<string, string> = {
    dataset_profiled: "Dataset profile created",
    plan_validated: "Plan passed policy validation",
    data_prepared: "Data prepared",
    experiment_started: "Experiment started",
    experiment_completed: "Experiment completed",
    experiment_failed: "Experiment did not complete",
    holdout_evaluation_started: "Final holdout evaluation started",
    holdout_evaluation_completed: "Final holdout evaluation completed",
    holdout_evaluation_failed: "Final holdout evaluation did not complete",
    candidate_submitted: "Candidate submitted for review",
    workflow_failed: "Workflow stopped",
    run_completed: "Workflow complete",
  };
  if (event.kind === "decision")
    return `Agent chose to ${titleCase(event.action || "continue")}`;
  return labels[event.kind] || titleCase(event.kind);
}

function RunConfirmation({
  datasetName,
  objective,
  budget,
  onCancel,
  onConfirm,
  submitting,
}: {
  datasetName: string;
  objective: string;
  budget: number;
  onCancel: () => void;
  onConfirm: () => void;
  submitting: boolean;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialogRef.current?.showModal();
    return () => dialogRef.current?.close();
  }, []);
  return (
    <dialog
      ref={dialogRef}
      className="dialog execution-confirm-dialog"
      aria-labelledby="execute-dialog-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!submitting) onCancel();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">
            <ShieldCheck size={13} /> CONFIRM CONTROLLED RUN
          </span>
          <h2 id="execute-dialog-title">Start this bounded workflow?</h2>
        </div>
        <button
          className="icon-button"
          aria-label="Cancel workflow execution"
          onClick={onCancel}
          disabled={submitting}
        >
          <X size={18} />
        </button>
      </header>
      <p className="dialog-intro">
        The server will create a fresh policy-validated plan and run no more
        than {budget} allowlisted experiment{budget === 1 ? "" : "s"}.
      </p>
      <div className="execution-confirm-details">
        <span>
          <Database size={15} /> {datasetName}
        </span>
        <span>
          <Target size={15} /> {objective}
        </span>
      </div>
      <div className="execution-confirm-guardrail">
        <BadgeCheck size={18} />
        <span>
          A successful candidate is only submitted for human review. This action
          cannot approve, deploy, or retrain a model.
        </span>
      </div>
      <footer className="execution-dialog-actions">
        <button
          className="secondary-button"
          onClick={onCancel}
          disabled={submitting}
        >
          Cancel
        </button>
        <button
          className="primary-button"
          onClick={onConfirm}
          disabled={submitting}
        >
          {submitting ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <Play size={15} />
          )}
          {submitting ? "Running controlled workflow" : "Start workflow"}
        </button>
      </footer>
    </dialog>
  );
}

function RunTimeline({ events }: { events: RunEvent[] }) {
  if (!events.length) {
    return (
      <div className="execution-empty-inline">
        <Clock3 size={18} /> Lifecycle events will appear here after the run is
        recorded.
      </div>
    );
  }
  return (
    <ol className="execution-timeline">
      {events.map((event, index) => (
        <li
          key={`${event.kind}-${event.at}-${index}`}
          className={`timeline-${event.kind}`}
        >
          <span className="timeline-icon">
            <EventIcon kind={event.kind} />
          </span>
          <div>
            <div className="timeline-heading">
              <strong>{eventTitle(event)}</strong>
              <time>{formatDate(event.at)}</time>
            </div>
            <p>
              {event.reason ||
                event.message ||
                "Recorded as part of this workflow."}
            </p>
            {event.kind === "data_prepared" && (
              <small>
                {event.feature_count ?? "—"} features ·{" "}
                {event.training_rows ?? "—"} train ·{" "}
                {event.validation_rows ?? "—"} validation ·{" "}
                {event.holdout_rows ?? "—"} holdout rows
              </small>
            )}
            {event.kind === "decision" &&
              (event.model_name || event.strategy) && (
                <small>
                  {event.model_name
                    ? modelName(event.model_name)
                    : "No model selected"}
                  {event.strategy ? ` · ${titleCase(event.strategy)}` : ""}
                </small>
              )}
            {event.experiment_id && event.kind !== "decision" && (
              <small>
                {event.metric
                  ? `${metricName(event.metric)} ${scoreValue(event.score)}`
                  : event.experiment_id}
                {event.elapsed_seconds !== undefined
                  ? ` · ${event.elapsed_seconds.toFixed(2)}s`
                  : ""}
              </small>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

function ExperimentTable({ experiments }: { experiments: Experiment[] }) {
  if (!experiments.length) {
    return (
      <div className="execution-empty-inline">
        <FlaskConical size={18} /> No experiment completed before this workflow
        stopped.
      </div>
    );
  }
  return (
    <div className="table-scroll execution-table-scroll">
      <table>
        <thead>
          <tr>
            <th>EXPERIMENT</th>
            <th>MODEL / STRATEGY</th>
            <th>RESULT</th>
            <th>TIME</th>
            <th>STATUS</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((experiment) => (
            <tr key={experiment.experiment_id}>
              <td>
                <strong>
                  {experiment.experiment_id.split("-").slice(-2).join(" · ")}
                </strong>
                <small>{experiment.experiment_id}</small>
              </td>
              <td>
                <strong>{modelName(experiment.model_name)}</strong>
                <small>{titleCase(experiment.strategy)}</small>
              </td>
              <td>
                <strong>{scoreValue(experiment.score)}</strong>
                <small>{metricName(experiment.metric)}</small>
              </td>
              <td>{experiment.elapsed_seconds.toFixed(2)}s</td>
              <td>
                <span className={`status-badge status-${experiment.status}`}>
                  <span /> {statusLabel(experiment.status)}
                </span>
                {experiment.error && (
                  <small className="experiment-error">{experiment.error}</small>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExecutionReport({ run }: { run: StoredRun }) {
  const { plan, summary } = run;
  const proposalId = summary?.promotion_proposal_id;
  return (
    <div className="execution-report" aria-live="polite">
      <section className={`execution-result-hero hero-${run.status}`}>
        <div>
          <span className="eyebrow">
            <Sparkles size={12} /> RECORDED WORKFLOW
          </span>
          <h2>{statusLabel(run.status)}</h2>
          <p>
            {summary?.stop_reason ||
              "The recorded workflow has no final status message."}
          </p>
        </div>
        <span className={`status-badge status-${run.status}`}>
          <span /> {statusLabel(run.status)}
        </span>
      </section>

      <section
        className="execution-summary-grid"
        aria-label="Workflow result summary"
      >
        <article>
          <span className="execution-stat-icon blue">
            <Target size={18} />
          </span>
          <small>Prediction target</small>
          <strong>{plan?.target_column || "Not planned"}</strong>
          <p>{plan ? titleCase(plan.task_type) : "No validated plan"}</p>
        </article>
        <article>
          <span className="execution-stat-icon purple">
            <Gauge size={18} />
          </span>
          <small>Quality gate</small>
          <strong>
            {plan
              ? thresholdLabel(plan.metric, plan.acceptance_threshold)
              : "—"}
          </strong>
          <p>{plan ? metricName(plan.metric) : "No metric recorded"}</p>
        </article>
        <article>
          <span className="execution-stat-icon teal">
            <FlaskConical size={18} />
          </span>
          <small>Experiments completed</small>
          <strong>
            {
              run.experiments.filter((item) => item.status === "completed")
                .length
            }{" "}
            of {plan?.max_experiments ?? "—"}
          </strong>
          <p>
            {run.dataset.rows} rows · {run.dataset.columns} columns
          </p>
        </article>
        <article>
          <span className="execution-stat-icon amber">
            <ClipboardCheck size={18} />
          </span>
          <small>Approval state</small>
          <strong>
            {proposalId ? "Awaiting reviewer" : "No candidate submitted"}
          </strong>
          <p>
            {proposalId
              ? "Human decision required"
              : "No promotion action taken"}
          </p>
        </article>
      </section>

      {proposalId && (
        <section className="execution-approval-callout">
          <span>
            <ShieldCheck size={22} />
          </span>
          <div>
            <h3>Promotion is waiting for a human decision.</h3>
            <p>
              Candidate <strong>{proposalId.slice(0, 12)}</strong> passed the
              run gate, but this workflow did not approve or deploy it.
            </p>
          </div>
          <span className="stage-next-badge">Compare models next</span>
        </section>
      )}

      <section className="execution-detail-grid">
        <article className="panel execution-panel timeline-panel">
          <header className="panel-header">
            <div>
              <h2>Execution timeline</h2>
              <p>Actual lifecycle events saved by the controlled backend.</p>
            </div>
            <span className="neutral-badge">{run.events.length} events</span>
          </header>
          <RunTimeline events={run.events} />
        </article>
        <article className="panel execution-panel agent-loop-panel">
          <header className="panel-header">
            <div>
              <h2>Agent feedback loop</h2>
              <p>
                Each choice is declarative; deterministic tools do the work.
              </p>
            </div>
            <BrainCircuit size={20} />
          </header>
          <div className="agent-loop-list">
            {run.events.filter((event) => event.kind === "decision").length ? (
              run.events
                .filter((event) => event.kind === "decision")
                .map((event, index) => (
                  <div className="agent-loop-item" key={`${event.at}-${index}`}>
                    <span>{index + 1}</span>
                    <div>
                      <strong>{titleCase(event.action || "decision")}</strong>
                      <p>{event.reason || "Agent decision recorded."}</p>
                      {(event.model_name || event.strategy) && (
                        <small>
                          {event.model_name
                            ? modelName(event.model_name)
                            : "No model"}
                          {event.strategy
                            ? ` · ${titleCase(event.strategy)}`
                            : ""}
                        </small>
                      )}
                    </div>
                  </div>
                ))
            ) : (
              <div className="execution-empty-inline">
                <BrainCircuit size={18} /> No decision events were saved.
              </div>
            )}
          </div>
          {run.final_evaluation && (
            <div className="holdout-result">
              <span>
                <BadgeCheck size={17} />
              </span>
              <div>
                <small>FINAL HOLDOUT RESULT</small>
                <strong>
                  {metricName(run.final_evaluation.metric)}{" "}
                  {scoreValue(run.final_evaluation.score)}
                </strong>
                <p>
                  {modelName(run.final_evaluation.model_name)} · untouched
                  evaluation split
                </p>
              </div>
            </div>
          )}
        </article>
      </section>

      <section className="panel execution-panel">
        <header className="panel-header">
          <div>
            <h2>Experiment results</h2>
            <p>Metrics and timings from each executed allowlisted recipe.</p>
          </div>
          <span className="neutral-badge">Run {run.run_id.slice(0, 12)}</span>
        </header>
        <ExperimentTable experiments={run.experiments} />
      </section>
    </div>
  );
}

export function ExecutionPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const canRun = user.permissions.includes("runs:write");
  const [datasets, setDatasets] = useState<DatasetList | null>(null);
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [objective, setObjective] = useState("");
  const [target, setTarget] = useState("");
  const [threshold, setThreshold] = useState("");
  const [budget, setBudget] = useState(3);
  const [activeRun, setActiveRun] = useState<StoredRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingRun, setLoadingRun] = useState(false);
  const [running, setRunning] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState("");

  const handleError = useCallback(
    (cause: unknown) => {
      if (cause instanceof ApiError && cause.status === 401) {
        onSessionExpired();
        return;
      }
      setError(
        cause instanceof Error
          ? cause.message
          : "The request could not be completed.",
      );
    },
    [onSessionExpired],
  );

  const refreshWorkspace = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [datasetResponse, runsResponse] = await Promise.all([
        api<DatasetList>("/datasets?offset=0&limit=50"),
        api<Run[]>("/runs"),
      ]);
      setDatasets(datasetResponse);
      setRuns(runsResponse);
      setSelectedDataset((current) =>
        datasetResponse.items.some((item) => item.dataset_name === current)
          ? current
          : datasetResponse.items[0]?.dataset_name || "",
      );
    } catch (cause) {
      handleError(cause);
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  useEffect(() => {
    void refreshWorkspace();
  }, [refreshWorkspace]);

  useEffect(() => {
    setReport(null);
    setTarget("");
    if (!selectedDataset) return;
    const controller = new AbortController();
    api<DatasetReport>(`/datasets/${selectedDataset}`, {
      signal: controller.signal,
    })
      .then(setReport)
      .catch((cause) => {
        if (!controller.signal.aborted) handleError(cause);
      });
    return () => controller.abort();
  }, [handleError, selectedDataset]);

  async function openRun(runId: string) {
    setLoadingRun(true);
    setError("");
    try {
      setActiveRun(await api<StoredRun>(`/runs/${runId}`));
    } catch (cause) {
      handleError(cause);
    } finally {
      setLoadingRun(false);
    }
  }

  function requestRun(event: FormEvent) {
    event.preventDefault();
    if (!canRun || running) return;
    if (!selectedDataset || !report) {
      setError("Choose a profiled dataset before starting a workflow.");
      return;
    }
    if (objective.trim().length < 3) {
      setError("Describe the ML objective in at least three characters.");
      return;
    }
    const parsed = threshold.trim() ? Number(threshold) : undefined;
    if (parsed !== undefined && (!Number.isFinite(parsed) || parsed < 0)) {
      setError("Acceptance threshold must be a non-negative number.");
      return;
    }
    setError("");
    setConfirming(true);
  }

  async function executeRun() {
    const parsedThreshold = threshold.trim() ? Number(threshold) : undefined;
    setRunning(true);
    setError("");
    try {
      const summary = await api<RunSummary>("/runs", {
        method: "POST",
        body: JSON.stringify({
          dataset_name: selectedDataset,
          objective: objective.trim(),
          ...(target ? { target_column: target } : {}),
          ...(parsedThreshold !== undefined
            ? { acceptance_threshold: parsedThreshold }
            : {}),
          max_experiments: budget,
        }),
      });
      try {
        setActiveRun(await api<StoredRun>(`/runs/${summary.run_id}`));
      } catch {
        setActiveRun(storedFromSummary(summary));
      }
      await refreshWorkspace();
    } catch (cause) {
      handleError(cause);
    } finally {
      setRunning(false);
      setConfirming(false);
    }
  }

  const selectedName =
    datasets?.items.find((item) => item.dataset_name === selectedDataset)
      ?.display_name || "Selected dataset";
  const validTargets =
    report?.profile.columns.filter((column) => !column.likely_identifier) || [];

  return (
    <div className="execution-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STAGE 04 · CONTROLLED EXECUTION</div>
          <h1>Run the workflow with guardrails.</h1>
          <p>
            Execute a policy-validated plan, observe the agent loop, and keep
            promotion human-controlled.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refreshWorkspace()}
          disabled={loading || running}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} />
          Refresh records
        </button>
      </div>

      <section className="execution-intro">
        <span className="execution-intro-icon">
          <ShieldCheck size={22} />
        </span>
        <div>
          <h2>One deliberate action, a fully recorded loop.</h2>
          <p>
            When you start a workflow, the server creates a fresh plan, applies
            policy limits, runs deterministic tools, and records every decision.
            The agent never executes arbitrary code.
          </p>
        </div>
        <span className="execution-safety-pill">
          <Check size={13} /> Bounded experiments
        </span>
      </section>

      {!canRun && (
        <section className="execution-readonly">
          <Eye size={17} />
          <span>
            You have review-only access. You can inspect recorded workflows
            below, while a Data Scientist or Administrator starts new runs.
          </span>
        </section>
      )}
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section className="panel execution-form-panel">
        <header className="panel-header">
          <div>
            <h2>Start a controlled workflow</h2>
            <p>
              Execution creates a new plan from the selected dataset and
              objective; the Stage 03 preview remains review-only.
            </p>
          </div>
          <span className="neutral-badge">
            <Clock3 size={13} /> Server-side run
          </span>
        </header>
        <form onSubmit={requestRun}>
          <div className="execution-form-grid">
            <label className="execution-field">
              <span>
                Dataset <em>required</em>
              </span>
              <select
                value={selectedDataset}
                onChange={(event) => setSelectedDataset(event.target.value)}
                disabled={!canRun || loading || running}
              >
                {!datasets?.items.length && (
                  <option value="">No datasets available</option>
                )}
                {datasets?.items.map((dataset) => (
                  <option
                    key={dataset.dataset_name}
                    value={dataset.dataset_name}
                  >
                    {dataset.display_name}
                  </option>
                ))}
              </select>
              <small>
                {report
                  ? `${report.profile.rows} rows · ${report.profile.columns.length} columns · profile ready`
                  : "Load a profiled dataset to continue."}
              </small>
            </label>
            <label className="execution-field">
              <span>
                Prediction target <em>optional</em>
              </span>
              <select
                value={target}
                onChange={(event) => setTarget(event.target.value)}
                disabled={!canRun || !report || running}
              >
                <option value="">Let the agent identify it</option>
                {validTargets.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name} · {column.inferred_type}
                  </option>
                ))}
              </select>
              <small>
                Identifier-like columns are excluded from target choices.
              </small>
            </label>
          </div>
          <label className="execution-field objective-field">
            <span>
              ML objective <em>required</em>
            </span>
            <textarea
              value={objective}
              onChange={(event) => setObjective(event.target.value)}
              disabled={!canRun || running}
              placeholder="Example: “Classify churn using F1 at least 0.80.”"
            />
            <small>
              State the outcome and, if useful, a metric or acceptance
              threshold.
            </small>
          </label>
          <div className="execution-controls">
            <label className="execution-field">
              <span>
                Acceptance threshold <em>optional</em>
              </span>
              <input
                type="number"
                min="0"
                step="any"
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
                disabled={!canRun || running}
                placeholder="Agent default"
              />
            </label>
            <label className="execution-field">
              <span>Experiment budget</span>
              <select
                value={budget}
                onChange={(event) => setBudget(Number(event.target.value))}
                disabled={!canRun || running}
              >
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value} value={value}>
                    {value} controlled{" "}
                    {value === 1 ? "experiment" : "experiments"}
                  </option>
                ))}
              </select>
            </label>
            <div className="execution-submit-wrap">
              <button
                className="primary-button"
                type="submit"
                disabled={
                  !canRun || loading || running || !datasets?.items.length
                }
              >
                {running ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Play size={15} />
                )}
                {running ? "Running workflow" : "Review & start"}
              </button>
              <small>
                <ShieldCheck size={12} /> Human approval is still required for
                promotion
              </small>
            </div>
          </div>
        </form>
      </section>

      {running && (
        <section className="execution-running panel">
          <LoaderCircle className="spin" size={23} />
          <div>
            <strong>The bounded workflow is executing on the server.</strong>
            <p>
              The prototype returns the recorded timeline and results when the
              controlled run completes.
            </p>
          </div>
        </section>
      )}

      <section className="execution-history-heading">
        <div>
          <span className="eyebrow">SAVED ARTIFACTS</span>
          <h2>Inspect a recorded workflow</h2>
        </div>
        <span>
          {runs.length} saved run{runs.length === 1 ? "" : "s"}
        </span>
      </section>
      {loadingRun && (
        <div className="execution-loading">
          <LoaderCircle className="spin" size={17} /> Opening the recorded
          workflow…
        </div>
      )}
      {!loadingRun && runs.length > 0 && (
        <div className="execution-run-list" aria-label="Saved workflows">
          {runs.slice(0, 6).map((run) => (
            <button
              key={run.run_id}
              className={`execution-run-card ${activeRun?.run_id === run.run_id ? "selected" : ""}`}
              onClick={() => void openRun(run.run_id)}
            >
              <span
                className={`run-card-icon status-${run.status || "unknown"}`}
              >
                <FlaskConical size={16} />
              </span>
              <span className="run-card-copy">
                <strong>{run.objective || "Untitled workflow"}</strong>
                <small>
                  {formatDate(run.started_at)} · {run.run_id.slice(0, 10)}
                </small>
              </span>
              <span
                className={`status-badge status-${run.status || "unknown"}`}
              >
                <span /> {statusLabel(run.status)}
              </span>
              <ChevronRight size={16} />
            </button>
          ))}
        </div>
      )}
      {!loading && !runs.length && !activeRun && (
        <section className="panel execution-placeholder">
          <span>
            <FlaskConical size={25} />
          </span>
          <h2>No workflow results yet.</h2>
          <p>
            Choose a dataset and objective above to create your first fully
            recorded controlled run.
          </p>
        </section>
      )}
      {activeRun && <ExecutionReport run={activeRun} />}

      {confirming && (
        <RunConfirmation
          datasetName={selectedName}
          objective={objective.trim()}
          budget={budget}
          onCancel={() => setConfirming(false)}
          onConfirm={() => void executeRun()}
          submitting={running}
        />
      )}
    </div>
  );
}
