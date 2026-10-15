import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BadgeCheck,
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  Clock3,
  Database,
  FlaskConical,
  History,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { Experiment, Run, StoredRun } from "./api";
import "./experiment-history.css";

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

function titleCase(value: string | null | undefined) {
  return (value || "Not recorded").replaceAll("_", " ");
}

function modelName(value: string | null | undefined) {
  return MODEL_NAMES[value || ""] || titleCase(value);
}

function metricName(value: string | null | undefined) {
  return METRIC_NAMES[value || ""] || titleCase(value || "Metric");
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

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value))
    return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function statusLabel(value: string | null | undefined) {
  const labels: Record<string, string> = {
    needs_review: "Awaiting approval",
    stopped: "Stopped safely",
    failed: "Needs attention",
    completed: "Completed",
    running: "Running",
  };
  return labels[value || ""] || titleCase(value);
}

function statusClass(value: string | null | undefined) {
  if (value === "failed") return "attention";
  if (value === "needs_review") return "review";
  if (value === "completed") return "completed";
  return "neutral";
}

function bestExperiment(run: StoredRun) {
  const metric = run.plan?.metric;
  const complete = run.experiments.filter(
    (experiment) =>
      experiment.status === "completed" &&
      experiment.score !== null &&
      experiment.score !== undefined,
  );
  if (!complete.length) return null;
  return complete.reduce((best, candidate) => {
    const bestScore = Number(best.score);
    const candidateScore = Number(candidate.score);
    return metric === "rmse"
      ? candidateScore < bestScore
        ? candidate
        : best
      : candidateScore > bestScore
        ? candidate
        : best;
  });
}

function RunStatus({ status }: { status: string | null | undefined }) {
  return (
    <span className={`history-status ${statusClass(status)}`}>
      <span /> {statusLabel(status)}
    </span>
  );
}

function DetailExperiments({ run }: { run: StoredRun }) {
  const experiments = run.experiments || [];
  const best = bestExperiment(run);
  return (
    <article className="panel history-experiment-panel">
      <header className="panel-header">
        <div>
          <span className="eyebrow">RUN EVIDENCE</span>
          <h2>Recorded experiments</h2>
          <p>Only recipes executed by the controlled backend appear here.</p>
        </div>
        <FlaskConical size={20} />
      </header>
      {!experiments.length ? (
        <div className="history-empty-inline">
          <CircleAlert size={18} /> This workflow did not record an experiment
          result.
        </div>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>MODEL</th>
                <th>PRIMARY RESULT</th>
                <th>TIME</th>
                <th>STRATEGY</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment: Experiment) => {
                const isBest = best?.experiment_id === experiment.experiment_id;
                return (
                  <tr
                    key={experiment.experiment_id}
                    className={isBest ? "history-best-row" : ""}
                  >
                    <td>
                      <strong>{modelName(experiment.model_name)}</strong>
                      <small>
                        {isBest
                          ? "Best validation result"
                          : experiment.experiment_id}
                      </small>
                    </td>
                    <td>
                      <strong>{formatScore(experiment.score)}</strong>
                      <small>{metricName(experiment.metric)}</small>
                    </td>
                    <td>
                      {Number(experiment.elapsed_seconds || 0).toFixed(2)}s
                    </td>
                    <td>{titleCase(experiment.strategy)}</td>
                    <td>
                      <RunStatus status={experiment.status} />
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

export function ExperimentHistoryPage({
  onSessionExpired,
}: {
  onSessionExpired: () => void;
}) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selectedRun, setSelectedRun] = useState<StoredRun | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
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
      const nextRuns = await api<Run[]>("/runs");
      setRuns(nextRuns);
      setSelectedId((current) =>
        nextRuns.some((run) => run.run_id === current)
          ? current
          : nextRuns[0]?.run_id || "",
      );
    } catch (cause) {
      handleError(cause, "Experiment history could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  const loadRun = useCallback(
    async (runId: string) => {
      if (!runId) {
        setSelectedRun(null);
        return;
      }
      setLoadingDetail(true);
      try {
        setSelectedRun(await api<StoredRun>(`/runs/${runId}`));
      } catch (cause) {
        handleError(cause, "The selected workflow record could not be loaded.");
      } finally {
        setLoadingDetail(false);
      }
    },
    [handleError],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    void loadRun(selectedId);
  }, [loadRun, selectedId]);

  const visibleRuns = useMemo(() => {
    const query = search.trim().toLowerCase();
    return runs.filter((run) => {
      const matchesStatus = filter === "all" || run.status === filter;
      const matchesSearch =
        !query ||
        `${run.objective} ${run.run_id}`.toLowerCase().includes(query);
      return matchesStatus && matchesSearch;
    });
  }, [filter, runs, search]);
  const completed = runs.filter((run) => run.status === "completed").length;
  const review = runs.filter((run) => run.status === "needs_review").length;
  const failed = runs.filter((run) => run.status === "failed").length;
  const best = selectedRun ? bestExperiment(selectedRun) : null;

  return (
    <div className="experiment-history-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">SUPPORTING PAGE · EXPERIMENT HISTORY</div>
          <h1>Every controlled workflow, in one evidence trail.</h1>
          <p>
            Search saved runs, inspect their plans and results, then return to
            comparison or approval with the right context.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading || loadingDetail}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          history
        </button>
      </div>

      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}
      <section className="history-hero">
        <span className="history-hero-icon">
          <History size={24} />
        </span>
        <div>
          <span className="eyebrow">PERSISTED WORKFLOW RECORDS</span>
          <h2>Evidence is retained after each bounded run.</h2>
          <p>
            Run objectives, plans, controlled experiment results, holdout
            evidence, and candidate handoffs are read from saved backend
            artifacts.
          </p>
        </div>
        <span className="history-hero-state">
          <span /> Read-only evidence
        </span>
      </section>

      <section
        className="history-status-grid"
        aria-label="Experiment history status"
      >
        <article>
          <span className="history-stat-icon blue">
            <History size={18} />
          </span>
          <small>Recorded workflows</small>
          <strong>{runs.length}</strong>
          <p>Saved controlled runs</p>
        </article>
        <article>
          <span className="history-stat-icon teal">
            <CheckCircle2 size={18} />
          </span>
          <small>Completed</small>
          <strong>{completed}</strong>
          <p>Finished successfully</p>
        </article>
        <article>
          <span className="history-stat-icon purple">
            <BadgeCheck size={18} />
          </span>
          <small>Awaiting review</small>
          <strong>{review}</strong>
          <p>Candidate handoffs</p>
        </article>
        <article>
          <span className="history-stat-icon amber">
            <CircleAlert size={18} />
          </span>
          <small>Need attention</small>
          <strong>{failed}</strong>
          <p>Recorded failed runs</p>
        </article>
      </section>

      <section className="panel history-library-panel">
        <header className="panel-header">
          <div>
            <span className="eyebrow">WORKFLOW LIBRARY</span>
            <h2>Find a saved experiment run</h2>
            <p>
              Latest records are shown first. This page does not start, alter,
              or delete workflows.
            </p>
          </div>
          <SlidersHorizontal size={20} />
        </header>
        <div className="history-filters">
          <label className="history-search">
            <Search size={16} />
            <span className="sr-only">Search workflow history</span>
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search objective or run ID"
            />
          </label>
          <label>
            <span>Workflow status</span>
            <select
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="all">All statuses</option>
              <option value="completed">Completed</option>
              <option value="needs_review">Awaiting approval</option>
              <option value="failed">Needs attention</option>
              <option value="stopped">Stopped safely</option>
            </select>
          </label>
          <span>
            {visibleRuns.length} of {runs.length} shown
          </span>
        </div>
        {loading ? (
          <div className="history-empty-inline">
            <LoaderCircle className="spin" size={18} /> Reading saved workflows…
          </div>
        ) : !visibleRuns.length ? (
          <div className="history-empty-inline">
            <History size={18} /> No workflow record matches these filters.
          </div>
        ) : (
          <div className="table-scroll history-run-table">
            <table>
              <thead>
                <tr>
                  <th>WORKFLOW</th>
                  <th>OBJECTIVE</th>
                  <th>CREATED</th>
                  <th>STATUS</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {visibleRuns.map((run) => (
                  <tr
                    key={run.run_id}
                    className={
                      selectedId === run.run_id ? "history-selected-row" : ""
                    }
                  >
                    <td>
                      <strong>{run.run_id.slice(0, 12)}</strong>
                      <small>Saved workflow identifier</small>
                    </td>
                    <td>
                      <span className="history-objective">
                        {run.objective || "Objective not recorded"}
                      </span>
                    </td>
                    <td>{formatDate(run.started_at)}</td>
                    <td>
                      <RunStatus status={run.status} />
                    </td>
                    <td>
                      <button
                        className="history-open-run"
                        onClick={() => setSelectedId(run.run_id)}
                        aria-label={`View workflow ${run.run_id}`}
                      >
                        View <ArrowRight size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="history-section-heading">
        <div>
          <span className="eyebrow">SELECTED WORKFLOW</span>
          <h2>Run detail</h2>
        </div>
        {selectedRun && <RunStatus status={selectedRun.status} />}
      </section>
      {loadingDetail ? (
        <section className="panel history-detail-loading">
          <LoaderCircle className="spin" size={19} /> Loading saved run
          evidence…
        </section>
      ) : selectedRun ? (
        <>
          <section className="history-detail-hero">
            <div>
              <span className="eyebrow">
                {selectedRun.run_id.slice(0, 12)} · SAVED WORKFLOW
              </span>
              <h2>{selectedRun.objective}</h2>
              <p>
                Started {formatDate(selectedRun.started_at)}
                {selectedRun.completed_at
                  ? ` · Completed ${formatDate(selectedRun.completed_at)}`
                  : ""}
              </p>
            </div>
            <a className="text-button" href="#comparison">
              Open comparison <ArrowRight size={14} />
            </a>
          </section>
          <section className="history-detail-grid">
            <article className="panel history-plan-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">PLAN SNAPSHOT</span>
                  <h2>Decision context</h2>
                </div>
                <BrainCircuit size={20} />
              </header>
              {selectedRun.plan ? (
                <div className="history-plan-facts">
                  <span>
                    <small>TARGET</small>
                    <strong>{selectedRun.plan.target_column}</strong>
                  </span>
                  <span>
                    <small>TASK</small>
                    <strong>{titleCase(selectedRun.plan.task_type)}</strong>
                  </span>
                  <span>
                    <small>PRIMARY METRIC</small>
                    <strong>{metricName(selectedRun.plan.metric)}</strong>
                  </span>
                  <span>
                    <small>EXPERIMENT BUDGET</small>
                    <strong>{selectedRun.plan.max_experiments}</strong>
                  </span>
                </div>
              ) : (
                <div className="history-empty-inline">
                  <BrainCircuit size={18} /> No saved plan is available for this
                  run.
                </div>
              )}
            </article>
            <article className="panel history-dataset-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">DATASET SNAPSHOT</span>
                  <h2>Input fingerprint</h2>
                </div>
                <Database size={20} />
              </header>
              <div className="history-dataset-facts">
                <span>
                  <small>ROWS</small>
                  <strong>
                    {selectedRun.dataset?.rows?.toLocaleString() || "—"}
                  </strong>
                </span>
                <span>
                  <small>COLUMNS</small>
                  <strong>{selectedRun.dataset?.columns ?? "—"}</strong>
                </span>
                <span className="history-fingerprint">
                  <small>DATA HASH</small>
                  <strong>
                    {selectedRun.dataset?.sha256?.slice(0, 16) ||
                      "Not recorded"}
                  </strong>
                </span>
              </div>
            </article>
            <article className="panel history-outcome-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">FINAL OUTCOME</span>
                  <h2>Selection evidence</h2>
                </div>
                <Target size={20} />
              </header>
              <div className="history-outcome-body">
                <span
                  className={
                    selectedRun.final_evaluation
                      ? "history-outcome-icon good"
                      : "history-outcome-icon neutral"
                  }
                >
                  {selectedRun.final_evaluation ? (
                    <BadgeCheck size={18} />
                  ) : (
                    <Clock3 size={18} />
                  )}
                </span>
                <div>
                  <strong>
                    {selectedRun.final_evaluation
                      ? `${formatScore(selectedRun.final_evaluation.score)} ${metricName(selectedRun.final_evaluation.metric)}`
                      : best
                        ? `${modelName(best.model_name)} selected`
                        : "No final holdout result"}
                  </strong>
                  <p>
                    {selectedRun.summary?.promotion_proposal_id
                      ? "Candidate submitted to the human approval gate."
                      : selectedRun.summary?.stop_reason ||
                        "No promotion handoff was recorded."}
                  </p>
                </div>
              </div>
            </article>
          </section>
          <DetailExperiments run={selectedRun} />
          <section className="history-guardrail">
            <ShieldCheck size={17} />
            <div>
              <strong>Evidence-only workspace</strong>
              <p>
                This page reflects saved records. Model approval, deployment,
                and retraining remain separate controlled actions.
              </p>
            </div>
          </section>
        </>
      ) : (
        <section className="panel history-detail-loading">
          <History size={20} /> Select a workflow to inspect its persisted
          evidence.
        </section>
      )}
    </div>
  );
}
