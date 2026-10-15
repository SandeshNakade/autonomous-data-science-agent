import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  ArrowRight,
  BrainCircuit,
  Check,
  ChevronRight,
  Database,
  FlaskConical,
  Info,
  LoaderCircle,
  LockKeyhole,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Target,
  TriangleAlert,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { DatasetList, DatasetReport, PlanPreview, User } from "./api";
import "./planning.css";

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  linear_regression: "Linear Regression",
  random_forest: "Random Forest",
  ridge: "Ridge Regression",
  xgboost: "XGBoost",
};

function humanize(value: string) {
  return value.replaceAll("_", " ");
}

function taskLabel(value: PlanPreview["plan"]["task_type"]) {
  return value === "binary_classification"
    ? "Binary classification"
    : value === "multiclass_classification"
      ? "Multiclass classification"
      : "Regression";
}

function metricLabel(value: string) {
  const labels: Record<string, string> = {
    f1: "F1 score",
    f1_macro: "Macro F1",
    accuracy: "Accuracy",
    roc_auc: "ROC-AUC",
    roc_auc_ovr: "ROC-AUC (OvR)",
    r2: "R²",
    rmse: "RMSE",
  };
  return labels[value] || humanize(value);
}

function thresholdLabel(plan: PlanPreview["plan"]) {
  const value = plan.acceptance_threshold;
  if (plan.metric === "rmse") return `≤ ${value.toLocaleString()}`;
  return `≥ ${value <= 1 ? `${(value * 100).toFixed((value * 100) % 1 ? 1 : 0)}%` : value}`;
}

function prepLabel(prep: PlanPreview["plan"]["preprocessing"]): string[] {
  const steps = [
    `Numeric gaps: ${humanize(prep.numeric_imputation)}`,
    `Categorical gaps: ${humanize(prep.categorical_imputation)}`,
  ];
  if (prep.scale_numeric) steps.push("Scale numeric features");
  if (prep.one_hot_encode) steps.push("One-hot encode categories");
  return steps;
}

function PlanningResult({ result }: { result: PlanPreview }) {
  const { plan, initial_decision: decision } = result;
  return (
    <div className="planning-result" aria-live="polite">
      <section className="plan-result-hero">
        <div>
          <span className="eyebrow">
            <Sparkles size={12} /> AGENT RECOMMENDATION
          </span>
          <h2>Here’s the controlled plan.</h2>
          <p>
            The adapter recommended a bounded setup from the dataset profile and
            your objective. No experiment has been run.
          </p>
        </div>
        <span className="planning-adapter">
          <span className="live-dot" />
          {result.planning_adapter === "openai_structured"
            ? "Structured LLM adapter"
            : "Deterministic planning adapter"}
        </span>
      </section>

      <section className="plan-summary-grid" aria-label="Agent plan summary">
        <article>
          <span className="plan-stat-icon blue">
            <Target size={18} />
          </span>
          <small>Prediction target</small>
          <strong>{plan.target_column}</strong>
          <p>{taskLabel(plan.task_type)}</p>
        </article>
        <article>
          <span className="plan-stat-icon purple">
            <FlaskConical size={18} />
          </span>
          <small>Success metric</small>
          <strong>{metricLabel(plan.metric)}</strong>
          <p>Acceptance {thresholdLabel(plan)}</p>
        </article>
        <article>
          <span className="plan-stat-icon teal">
            <BrainCircuit size={18} />
          </span>
          <small>Experiment budget</small>
          <strong>{plan.max_experiments} controlled runs</strong>
          <p>{Math.round(plan.test_size * 100)}% held out for evaluation</p>
        </article>
      </section>

      <div className="plan-detail-grid">
        <section className="panel planning-panel">
          <header className="panel-header">
            <div>
              <h2>Why the agent chose this</h2>
              <p>
                Recommendations are based on profile metadata, not generated
                code.
              </p>
            </div>
            <span className="neutral-badge">Reviewable</span>
          </header>
          <ol className="reasoning-list">
            {plan.reasoning.map((reason, index) => (
              <li key={`${reason}-${index}`}>
                <span>{index + 1}</span>
                <p>{reason}</p>
              </li>
            ))}
          </ol>
        </section>
        <section className="panel planning-panel">
          <header className="panel-header">
            <div>
              <h2>Data preparation</h2>
              <p>Fixed preprocessing rules applied only in a later run.</p>
            </div>
          </header>
          <div className="prep-list">
            {prepLabel(plan.preprocessing).map((step) => (
              <div key={step}>
                <span className="prep-check">
                  <Check size={14} />
                </span>
                <span>{step}</span>
              </div>
            ))}
            {plan.preprocessing.excluded_columns.length > 0 && (
              <div className="prep-excluded">
                <span className="prep-check muted">
                  <LockKeyhole size={13} />
                </span>
                <span>
                  Exclude likely identifier
                  {plan.preprocessing.excluded_columns.length > 1
                    ? "s"
                    : ""}:{" "}
                  <strong>
                    {plan.preprocessing.excluded_columns.join(", ")}
                  </strong>
                </span>
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="panel candidate-panel">
        <header className="panel-header">
          <div>
            <h2>Allowlisted candidate models</h2>
            <p>
              The agent can choose only from approved, deterministic recipes.
            </p>
          </div>
          <span className="neutral-badge">
            {plan.candidate_models.length} candidates
          </span>
        </header>
        <div className="candidate-flow">
          {plan.candidate_models.map((model, index) => (
            <div className="candidate-item" key={model}>
              {index > 0 && (
                <ChevronRight className="candidate-arrow" size={18} />
              )}
              <span
                className={`candidate-number ${index === 0 ? "first" : ""}`}
              >
                {index + 1}
              </span>
              <div>
                <strong>{MODEL_NAMES[model] || humanize(model)}</strong>
                <small>
                  {index === 0 ? "Baseline first" : "Only if needed"}
                </small>
              </div>
            </div>
          ))}
        </div>
        <div className="first-decision">
          <span>
            <BrainCircuit size={18} />
          </span>
          <div>
            <small>FIRST CONTROLLED EXPERIMENT</small>
            <strong>
              {MODEL_NAMES[decision.model_name || ""] ||
                humanize(decision.model_name || "")}
            </strong>
            <p>{decision.reason}</p>
          </div>
          <span className="neutral-badge">Not run</span>
        </div>
      </section>

      <section className="plan-guardrails">
        <span className="plan-guard-icon">
          <ShieldCheck size={23} />
        </span>
        <div>
          <h3>Controlled autonomy is still in control.</h3>
          <ul>
            {result.guardrails.map((guardrail) => (
              <li key={guardrail}>{guardrail}</li>
            ))}
          </ul>
        </div>
        <span className="stage-next-badge">Stage 04 ready</span>
      </section>
    </div>
  );
}

export function PlanningPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const canPlan = user.permissions.includes("runs:write");
  const [datasets, setDatasets] = useState<DatasetList | null>(null);
  const [selected, setSelected] = useState("");
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [objective, setObjective] = useState("");
  const [target, setTarget] = useState("");
  const [threshold, setThreshold] = useState("");
  const [budget, setBudget] = useState(3);
  const [result, setResult] = useState<PlanPreview | null>(null);
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");

  const handleError = useCallback(
    (errorValue: unknown, setMessage: (value: string) => void) => {
      if (errorValue instanceof ApiError && errorValue.status === 401) {
        onSessionExpired();
        return;
      }
      if (errorValue instanceof Error && errorValue.name !== "AbortError") {
        setMessage(errorValue.message);
      }
    },
    [onSessionExpired],
  );

  const refreshDatasets = useCallback(() => {
    const controller = new AbortController();
    setLoadingDatasets(true);
    setError("");
    api<DatasetList>("/datasets?offset=0&limit=50", {
      signal: controller.signal,
    })
      .then((data) => {
        setDatasets(data);
        setSelected((current) =>
          data.items.some((item) => item.dataset_name === current)
            ? current
            : data.items[0]?.dataset_name || "",
        );
      })
      .catch((cause) => handleError(cause, setError))
      .finally(() => {
        if (!controller.signal.aborted) setLoadingDatasets(false);
      });
    return () => controller.abort();
  }, [handleError]);

  useEffect(() => refreshDatasets(), [refreshDatasets]);

  useEffect(() => {
    setReport(null);
    setDetailError("");
    setTarget("");
    setResult(null);
    if (!selected) return;
    const controller = new AbortController();
    api<DatasetReport>(`/datasets/${selected}`, { signal: controller.signal })
      .then(setReport)
      .catch((cause) => handleError(cause, setDetailError));
    return () => controller.abort();
  }, [selected, handleError]);

  async function generatePlan(event: FormEvent) {
    event.preventDefault();
    if (!canPlan || planning) return;
    if (!selected || !report) {
      setError("Choose a profiled dataset before generating a plan.");
      return;
    }
    if (objective.trim().length < 3) {
      setError("Describe the ML objective in at least three characters.");
      return;
    }
    const parsedThreshold = threshold.trim() ? Number(threshold) : undefined;
    if (
      parsedThreshold !== undefined &&
      (!Number.isFinite(parsedThreshold) || parsedThreshold < 0)
    ) {
      setError("Acceptance threshold must be a non-negative number.");
      return;
    }
    setPlanning(true);
    setError("");
    setResult(null);
    try {
      setResult(
        await api<PlanPreview>("/plans/preview", {
          method: "POST",
          body: JSON.stringify({
            dataset_name: selected,
            objective: objective.trim(),
            ...(target ? { target_column: target } : {}),
            ...(parsedThreshold !== undefined
              ? { acceptance_threshold: parsedThreshold }
              : {}),
            max_experiments: budget,
          }),
        }),
      );
    } catch (cause) {
      handleError(cause, setError);
    } finally {
      setPlanning(false);
    }
  }

  const validTargets =
    report?.profile.columns.filter((column) => !column.likely_identifier) || [];
  const chosenDataset = datasets?.items.find(
    (item) => item.dataset_name === selected,
  );

  return (
    <div className="planning-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">ML LIFECYCLE / 02</div>
          <h1>
            AI Agent Planning<span className="greeting-dot">.</span>
          </h1>
          <p>
            Turn your objective into a bounded, explainable experiment plan.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={refreshDatasets}
          disabled={loadingDatasets}
        >
          <RefreshCw size={15} className={loadingDatasets ? "spin" : ""} />
          Refresh data
        </button>
      </div>

      <section className="planning-intro">
        <span className="planning-intro-icon">
          <BrainCircuit size={25} />
        </span>
        <div>
          <h2>Decision layer, not a code execution layer.</h2>
          <p>
            The agent receives dataset profile metadata and your objective, then
            proposes an allowlisted plan. Deterministic tools remain responsible
            for every later ML operation.
          </p>
        </div>
        <span className="planning-safety-pill">
          <ShieldCheck size={14} /> No training on this page
        </span>
      </section>

      {!canPlan && (
        <div className="planning-readonly">
          <LockKeyhole size={16} /> Reviewer access: planning is visible for
          review, but generating a new plan requires a Scientist or
          Administrator account.
        </div>
      )}
      {error && (
        <div className="error-notice" role="alert">
          <TriangleAlert size={18} />
          {error}
        </div>
      )}
      {detailError && (
        <div className="error-notice" role="alert">
          <TriangleAlert size={18} />
          {detailError}
        </div>
      )}

      <form className="planning-form panel" onSubmit={generatePlan}>
        <header className="panel-header">
          <div>
            <h2>Frame the planning request</h2>
            <p>
              Only these explicit controls are passed to the decision adapter.
            </p>
          </div>
          <span className="neutral-badge">
            <ShieldCheck size={13} /> Plan only
          </span>
        </header>
        <div className="planning-form-grid">
          <label className="planning-field dataset-picker">
            <span>Dataset</span>
            <select
              value={selected}
              disabled={loadingDatasets || !datasets?.items.length || !canPlan}
              onChange={(event) => setSelected(event.target.value)}
            >
              {!datasets?.items.length && (
                <option value="">No uploaded datasets</option>
              )}
              {datasets?.items.map((dataset) => (
                <option key={dataset.dataset_name} value={dataset.dataset_name}>
                  {dataset.display_name}{" "}
                  {dataset.rows
                    ? `· ${dataset.rows.toLocaleString()} rows`
                    : ""}
                </option>
              ))}
            </select>
            <small>
              {chosenDataset
                ? `${chosenDataset.columns || "—"} columns · uploaded dataset ID ${chosenDataset.dataset_name.slice(0, 8)}`
                : "Upload and profile a CSV in Dataset Management first."}
            </small>
          </label>
          <label className="planning-field">
            <span>
              Target variable <em>optional override</em>
            </span>
            <select
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              disabled={!report || !canPlan}
            >
              <option value="">
                Let the agent identify it from the objective
              </option>
              {validTargets.map((column) => (
                <option key={column.name} value={column.name}>
                  {column.name}
                  {report?.profile.likely_targets.includes(column.name)
                    ? " · likely target"
                    : ""}
                </option>
              ))}
            </select>
            <small>
              Likely identifier columns are excluded from target choices.
            </small>
          </label>
        </div>
        <label className="planning-field objective-field">
          <span>Natural-language ML objective</span>
          <textarea
            value={objective}
            disabled={!canPlan}
            maxLength={2000}
            rows={3}
            placeholder="Example: Classify churn with F1 at least 0.85."
            onChange={(event) => setObjective(event.target.value)}
          />
          <small>
            Mention the prediction goal and, if useful, a metric or acceptance
            criterion.
          </small>
        </label>
        <div className="planning-controls">
          <label className="planning-field compact-field">
            <span>
              Acceptance threshold <em>optional override</em>
            </span>
            <input
              type="number"
              min="0"
              step="0.01"
              inputMode="decimal"
              value={threshold}
              disabled={!canPlan}
              placeholder="Agent derives from objective"
              onChange={(event) => setThreshold(event.target.value)}
            />
          </label>
          <label className="planning-field compact-field">
            <span>Experiment limit</span>
            <select
              value={budget}
              disabled={!canPlan}
              onChange={(event) => setBudget(Number(event.target.value))}
            >
              {[1, 2, 3, 4, 5].map((value) => (
                <option key={value} value={value}>
                  {value} {value === 1 ? "experiment" : "experiments"}
                </option>
              ))}
            </select>
          </label>
          <div className="planning-submit-wrap">
            <button
              className="primary-button"
              type="submit"
              disabled={!canPlan || !report || !selected || planning}
            >
              {planning ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <BrainCircuit size={17} />
              )}
              {planning ? "Planning…" : "Generate controlled plan"}
            </button>
            <small>
              <Info size={13} /> This does not train or save an experiment.
            </small>
          </div>
        </div>
      </form>

      {!datasets?.items.length && !loadingDatasets && (
        <section className="planning-empty panel">
          <span>
            <Database size={25} />
          </span>
          <div>
            <h2>Start with a profiled dataset.</h2>
            <p>
              The planning agent works from deterministic profile metadata.
              Upload a CSV first, then return here to set the objective.
            </p>
          </div>
          <a className="text-button" href="#datasets">
            Go to Dataset Management <ArrowRight size={14} />
          </a>
        </section>
      )}

      {planning && (
        <section className="planning-loading panel" role="status">
          <LoaderCircle className="spin" size={22} />
          <div>
            <strong>Checking the objective against the dataset profile…</strong>
            <p>
              Validating task, metric, preprocessing, model recipes, and budget.
            </p>
          </div>
        </section>
      )}

      {result ? (
        <PlanningResult result={result} />
      ) : (
        <section className="planning-placeholder panel">
          <span>
            <BrainCircuit size={28} />
          </span>
          <h2>Your reviewable plan will appear here.</h2>
          <p>
            Choose an uploaded dataset, describe the ML objective, and ask the
            controlled agent for its recommendation.
          </p>
          <div>
            <span>Target & task</span>
            <ChevronRight size={14} />
            <span>Metric & quality gate</span>
            <ChevronRight size={14} />
            <span>Allowlisted models</span>
          </div>
        </section>
      )}
    </div>
  );
}
