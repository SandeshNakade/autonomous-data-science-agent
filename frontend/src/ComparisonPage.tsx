import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BadgeCheck,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleAlert,
  ClipboardCheck,
  Clock3,
  Eye,
  FlaskConical,
  Gauge,
  LoaderCircle,
  RefreshCw,
  Scale,
  ShieldCheck,
  Target,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { Experiment, Run, StoredRun } from "./api";
import "./comparison.css";

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

function humanize(value: string) {
  return value.replaceAll("_", " ");
}

function modelName(value: string | null | undefined) {
  return MODEL_NAMES[value || ""] || humanize(value || "Not recorded");
}

function metricName(value: string | null | undefined) {
  return METRIC_NAMES[value || ""] || humanize(value || "Metric");
}

function formatScore(score: number | null | undefined) {
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

function statusLabel(status: string | null | undefined) {
  const labels: Record<string, string> = {
    needs_review: "Awaiting approval",
    stopped: "Stopped safely",
    failed: "Needs attention",
    completed: "Completed",
    running: "Running",
  };
  return labels[status || ""] || humanize(status || "Not recorded");
}

function isHigherBetter(metric: string | null | undefined) {
  return metric !== "rmse";
}

function betterExperiment(
  experiments: Experiment[],
  metric: string | null | undefined,
) {
  const eligible = experiments.filter(
    (experiment) =>
      experiment.status === "completed" &&
      experiment.score !== null &&
      experiment.score !== undefined &&
      Number.isFinite(experiment.score),
  );
  if (!eligible.length) return null;
  return eligible.reduce((best, current) =>
    isHigherBetter(metric)
      ? Number(current.score) > Number(best.score)
        ? current
        : best
      : Number(current.score) < Number(best.score)
        ? current
        : best,
  );
}

function normalizedScore(
  experiment: Experiment,
  experiments: Experiment[],
  metric: string | null | undefined,
) {
  if (experiment.score === null || experiment.score === undefined) return 0;
  const values = experiments
    .map((item) => item.score)
    .filter((value): value is number => value !== null && value !== undefined);
  if (values.length < 2) return 100;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (maximum === minimum) return 100;
  const score = Number(experiment.score);
  const ratio = isHigherBetter(metric)
    ? (score - minimum) / (maximum - minimum)
    : (maximum - score) / (maximum - minimum);
  return Math.max(0, Math.min(100, 18 + ratio * 82));
}

function ComparisonScorecards({ run }: { run: StoredRun }) {
  const plan = run.plan;
  const experiments = run.experiments.filter(
    (item) => item.status === "completed" && item.score !== null,
  );
  const best = betterExperiment(experiments, plan?.metric);
  const sorted = [...experiments].sort((left, right) => {
    const leftScore = Number(left.score);
    const rightScore = Number(right.score);
    return isHigherBetter(plan?.metric)
      ? rightScore - leftScore
      : leftScore - rightScore;
  });

  if (!experiments.length) {
    return (
      <div className="comparison-empty-inline">
        <CircleAlert size={18} /> This workflow did not produce a completed
        experiment to compare.
      </div>
    );
  }
  return (
    <div className="comparison-scorecards">
      {sorted.map((experiment, index) => {
        const isBest = best?.experiment_id === experiment.experiment_id;
        return (
          <article
            className={`comparison-scorecard ${isBest ? "best" : ""}`}
            key={experiment.experiment_id}
          >
            <div className="comparison-card-topline">
              <span>#{index + 1}</span>
              {isBest ? (
                <span className="comparison-best-badge">
                  <BadgeCheck size={12} /> Best validation result
                </span>
              ) : (
                <span>{humanize(experiment.strategy)}</span>
              )}
            </div>
            <h3>{modelName(experiment.model_name)}</h3>
            <div className="comparison-score-line">
              <strong>{formatScore(experiment.score)}</strong>
              <span>{metricName(experiment.metric)}</span>
            </div>
            <div
              className="comparison-bar"
              aria-label={`${modelName(experiment.model_name)} relative score`}
            >
              <span
                style={{
                  width: `${normalizedScore(experiment, experiments, plan?.metric)}%`,
                }}
              />
            </div>
            <footer>
              <span>
                <Clock3 size={12} /> {experiment.elapsed_seconds.toFixed(2)}s
              </span>
              <span>
                {experiment.experiment_id.split("-").slice(-2).join(" · ")}
              </span>
            </footer>
          </article>
        );
      })}
    </div>
  );
}

function MetricTable({ run }: { run: StoredRun }) {
  const experiments = run.experiments.filter(
    (item) => item.status === "completed",
  );
  const metricColumns = useMemo(() => {
    const values = new Set<string>();
    experiments.forEach((experiment) => {
      values.add(experiment.metric);
      Object.keys(experiment.metrics || {}).forEach((metric) =>
        values.add(metric),
      );
    });
    const mainMetric = run.plan?.metric;
    return [...values].sort((left, right) => {
      if (left === mainMetric) return -1;
      if (right === mainMetric) return 1;
      return left.localeCompare(right);
    });
  }, [experiments, run.plan?.metric]);

  if (!experiments.length) {
    return (
      <div className="comparison-empty-inline">
        <FlaskConical size={18} /> No result metrics are available for this run.
      </div>
    );
  }
  return (
    <div className="table-scroll comparison-table-scroll">
      <table>
        <thead>
          <tr>
            <th>MODEL</th>
            {metricColumns.map((metric) => (
              <th key={metric}>{metricName(metric).toUpperCase()}</th>
            ))}
            <th>TIME</th>
            <th>STRATEGY</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((experiment) => (
            <tr key={experiment.experiment_id}>
              <td>
                <strong>{modelName(experiment.model_name)}</strong>
                <small>{experiment.experiment_id}</small>
              </td>
              {metricColumns.map((metric) => (
                <td
                  key={metric}
                  className={
                    metric === run.plan?.metric
                      ? "comparison-primary-metric"
                      : ""
                  }
                >
                  {formatScore(
                    metric === experiment.metric
                      ? experiment.score
                      : experiment.metrics?.[metric],
                  )}
                </td>
              ))}
              <td>{experiment.elapsed_seconds.toFixed(2)}s</td>
              <td>{humanize(experiment.strategy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonReport({ run }: { run: StoredRun }) {
  const plan = run.plan;
  const best = betterExperiment(run.experiments, plan?.metric);
  const proposalId = run.summary?.promotion_proposal_id;
  return (
    <div className="comparison-report" aria-live="polite">
      <section className="comparison-result-hero">
        <div>
          <span className="eyebrow">
            <Scale size={12} /> WITHIN-RUN COMPARISON
          </span>
          <h2>Compare the evidence, not just a score.</h2>
          <p>
            These results share the same dataset split, target, metric, and
            policy-validated plan. Scores from different workflows are kept
            separate.
          </p>
        </div>
        <span className="comparison-run-status">
          <span className="live-dot" /> {statusLabel(run.status)}
        </span>
      </section>

      <section
        className="comparison-summary-grid"
        aria-label="Comparison summary"
      >
        <article>
          <span className="comparison-stat-icon blue">
            <Target size={18} />
          </span>
          <small>Comparison target</small>
          <strong>{plan?.target_column || "Not recorded"}</strong>
          <p>{plan ? humanize(plan.task_type) : "No plan stored"}</p>
        </article>
        <article>
          <span className="comparison-stat-icon purple">
            <Gauge size={18} />
          </span>
          <small>Primary metric</small>
          <strong>{plan ? metricName(plan.metric) : "—"}</strong>
          <p>
            {plan
              ? `${isHigherBetter(plan.metric) ? "Higher" : "Lower"} is better`
              : "No comparison rule"}
          </p>
        </article>
        <article>
          <span className="comparison-stat-icon teal">
            <FlaskConical size={18} />
          </span>
          <small>Validation winner</small>
          <strong>
            {best ? modelName(best.model_name) : "No completed model"}
          </strong>
          <p>
            {best
              ? `${formatScore(best.score)} ${metricName(best.metric)}`
              : "No score available"}
          </p>
        </article>
        <article>
          <span className="comparison-stat-icon amber">
            <BadgeCheck size={18} />
          </span>
          <small>Final holdout</small>
          <strong>
            {run.final_evaluation
              ? formatScore(run.final_evaluation.score)
              : "Not reached"}
          </strong>
          <p>
            {run.final_evaluation
              ? metricName(run.final_evaluation.metric)
              : "No holdout evaluation"}
          </p>
        </article>
      </section>

      <section className="comparison-interpretation">
        <span>
          <BrainCircuit size={21} />
        </span>
        <div>
          <h3>How to read this comparison</h3>
          <p>
            {plan?.metric === "rmse"
              ? "For RMSE, lower values are better. The visual ordering reflects that rule."
              : "For this metric, higher values are better. The visual ordering reflects that rule."}{" "}
            Timing is supporting evidence, not an automatic promotion rule.
          </p>
        </div>
        <span className="comparison-plan-pill">
          {run.experiments.length} recorded recipes
        </span>
      </section>

      <section className="panel comparison-panel comparison-score-panel">
        <header className="panel-header">
          <div>
            <h2>Validation score comparison</h2>
            <p>
              Relative bars compare only the completed recipes in this workflow.
            </p>
          </div>
          <span className="neutral-badge">
            {plan ? metricName(plan.metric) : "Metric unavailable"}
          </span>
        </header>
        <ComparisonScorecards run={run} />
      </section>

      <section className="comparison-detail-grid">
        <article className="panel comparison-panel">
          <header className="panel-header">
            <div>
              <h2>Metric and timing table</h2>
              <p>
                Readable values from the saved deterministic evaluation results.
              </p>
            </div>
            <BarChart3 size={20} />
          </header>
          <MetricTable run={run} />
        </article>
        <article className="panel comparison-panel comparison-decision-panel">
          <header className="panel-header">
            <div>
              <h2>Selection context</h2>
              <p>Why this run stopped or moved to human review.</p>
            </div>
            <ShieldCheck size={20} />
          </header>
          <div className="comparison-context-list">
            <div>
              <span>
                <Check size={13} />
              </span>
              <p>
                <strong>Selected validation recipe</strong>
                {best
                  ? `${modelName(best.model_name)} · ${formatScore(best.score)} ${metricName(best.metric)}`
                  : "No completed recipe was selected."}
              </p>
            </div>
            {run.final_evaluation && (
              <div>
                <span>
                  <BadgeCheck size={13} />
                </span>
                <p>
                  <strong>Untouched holdout</strong>
                  {`${modelName(run.final_evaluation.model_name)} · ${formatScore(run.final_evaluation.score)} ${metricName(run.final_evaluation.metric)}`}
                </p>
              </div>
            )}
            <div>
              <span>
                <ClipboardCheck size={13} />
              </span>
              <p>
                <strong>Promotion state</strong>
                {proposalId
                  ? "Candidate exists, but a reviewer has not approved it here."
                  : "No candidate was submitted for promotion."}
              </p>
            </div>
          </div>
          <div className={`comparison-stop-note ${proposalId ? "ready" : ""}`}>
            <CircleAlert size={16} />
            <span>
              {run.summary?.stop_reason ||
                "No final decision note was recorded."}
            </span>
          </div>
          <a
            className="text-button comparison-execution-link"
            href="#execution"
          >
            Inspect execution timeline <ArrowRight size={14} />
          </a>
        </article>
      </section>

      <section className="comparison-next-stage">
        <span>
          <ClipboardCheck size={21} />
        </span>
        <div>
          <span className="eyebrow">STAGE 06 NEXT</span>
          <h3>Approval is a separate human decision.</h3>
          <p>
            Model comparison provides evidence. A reviewer will approve or
            reject a submitted candidate with a reason and audit trail.
          </p>
        </div>
        <span className="stage-next-badge">Approval center next</span>
      </section>
    </div>
  );
}

export function ComparisonPage({
  onSessionExpired,
}: {
  onSessionExpired: () => void;
}) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeRun, setActiveRun] = useState<StoredRun | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingRun, setLoadingRun] = useState(false);
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
          : "The comparison data could not be loaded.",
      );
    },
    [onSessionExpired],
  );

  const openRun = useCallback(
    async (runId: string) => {
      setLoadingRun(true);
      setError("");
      try {
        setActiveRun(await api<StoredRun>(`/runs/${runId}`));
      } catch (cause) {
        handleError(cause);
      } finally {
        setLoadingRun(false);
      }
    },
    [handleError],
  );

  const refreshRuns = useCallback(async () => {
    setLoadingList(true);
    setError("");
    try {
      const nextRuns = await api<Run[]>("/runs");
      setRuns(nextRuns);
      if (!activeRun && nextRuns[0]) await openRun(nextRuns[0].run_id);
    } catch (cause) {
      handleError(cause);
    } finally {
      setLoadingList(false);
    }
  }, [activeRun, handleError, openRun]);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  return (
    <div className="comparison-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STAGE 05 · EXPERIMENTS &amp; COMPARISON</div>
          <h1>Make the evidence easy to compare.</h1>
          <p>
            Review actual model quality, evaluation metrics, and timing within a
            single controlled workflow.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refreshRuns()}
          disabled={loadingList || loadingRun}
        >
          <RefreshCw size={15} className={loadingList ? "spin" : ""} /> Refresh
          records
        </button>
      </div>

      <section className="comparison-intro">
        <span className="comparison-intro-icon">
          <Scale size={22} />
        </span>
        <div>
          <h2>A score needs context before it earns trust.</h2>
          <p>
            Choose a saved workflow to compare its allowlisted recipes under the
            same target, split, metric, and acceptance criterion. This page
            reads saved artifacts; it never reruns training.
          </p>
        </div>
        <span className="comparison-safety-pill">
          <Eye size={13} /> Read-only evidence
        </span>
      </section>

      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section className="comparison-history-heading">
        <div>
          <span className="eyebrow">SAVED WORKFLOWS</span>
          <h2>Choose a run to compare</h2>
        </div>
        <span>
          {runs.length} recorded run{runs.length === 1 ? "" : "s"}
        </span>
      </section>
      {loadingList && (
        <div className="comparison-loading">
          <LoaderCircle className="spin" size={17} /> Reading saved workflow
          records…
        </div>
      )}
      {!loadingList && runs.length > 0 && (
        <div
          className="comparison-run-list"
          aria-label="Saved workflows for comparison"
        >
          {runs.slice(0, 8).map((run) => (
            <button
              key={run.run_id}
              className={`comparison-run-card ${activeRun?.run_id === run.run_id ? "selected" : ""}`}
              onClick={() => void openRun(run.run_id)}
            >
              <span
                className={`comparison-run-icon status-${run.status || "unknown"}`}
              >
                <FlaskConical size={16} />
              </span>
              <span className="comparison-run-copy">
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
      {!loadingList && !runs.length && (
        <section className="panel comparison-placeholder">
          <span>
            <FlaskConical size={25} />
          </span>
          <h2>No recorded workflows to compare yet.</h2>
          <p>
            Run a controlled workflow first. Its saved experiment evidence will
            appear here automatically.
          </p>
          <a className="text-button" href="#execution">
            Open workflow execution <ArrowRight size={14} />
          </a>
        </section>
      )}
      {loadingRun && (
        <div className="comparison-loading">
          <LoaderCircle className="spin" size={17} /> Opening comparison
          evidence…
        </div>
      )}
      {activeRun && !loadingRun && <ComparisonReport run={activeRun} />}
    </div>
  );
}
