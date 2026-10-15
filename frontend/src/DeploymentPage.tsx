import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BadgeCheck,
  Check,
  CircleAlert,
  ClipboardCheck,
  Clock3,
  Eye,
  Gauge,
  History,
  LoaderCircle,
  Play,
  RefreshCw,
  Rocket,
  Send,
  ShieldCheck,
  Target,
  X,
} from "lucide-react";
import { api, ApiError } from "./api";
import type {
  DeploymentDetails,
  DeploymentStatus,
  InputSchemaField,
  ModelProposal,
  User,
} from "./api";
import "./deployment.css";

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
const EMPTY_SCHEMA: InputSchemaField[] = [];

function displayModel(modelName: string | undefined) {
  return (
    MODEL_NAMES[modelName || ""] || modelName?.replaceAll("_", " ") || "Model"
  );
}

function displayMetric(metric: string | undefined) {
  return METRIC_NAMES[metric || ""] || metric?.replaceAll("_", " ") || "Metric";
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value))
    return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatDate(value: string | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function activeDeployment(
  proposal: ModelProposal | null,
): DeploymentDetails | null {
  return proposal?.deployment || null;
}

function DeployDialog({
  proposal,
  busy,
  onClose,
  onDeploy,
}: {
  proposal: ModelProposal;
  busy: boolean;
  onClose: () => void;
  onDeploy: (note: string) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [note, setNote] = useState("");
  useEffect(() => {
    dialogRef.current?.showModal();
    return () => dialogRef.current?.close();
  }, []);
  return (
    <dialog
      ref={dialogRef}
      className="dialog deployment-dialog"
      aria-labelledby="deploy-dialog-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onClose();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">
            <Rocket size={13} /> RELEASE TO LOCAL ENDPOINT
          </span>
          <h2 id="deploy-dialog-title">Deploy this approved candidate?</h2>
        </div>
        <button
          className="icon-button"
          aria-label="Cancel deployment"
          onClick={onClose}
          disabled={busy}
        >
          <X size={18} />
        </button>
      </header>
      <p className="dialog-intro">
        You are about to activate{" "}
        <strong>{displayModel(proposal.metadata.model_name)}</strong> for the
        local <code>/predict</code> endpoint.
      </p>
      <div className="deployment-dialog-guardrail">
        <ShieldCheck size={18} />
        <span>
          This is an administrator-only action. It switches the active local
          model and creates a retained release record. It cannot bypass the
          reviewer decision.
        </span>
      </div>
      <label className="deployment-note-field">
        <span>
          Release note <em>required</em>
        </span>
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Explain the purpose or scope of this deployment."
          minLength={3}
          maxLength={1200}
          disabled={busy}
          autoFocus
        />
        <small>{note.trim().length}/1200 characters</small>
      </label>
      <footer className="deployment-dialog-actions">
        <button className="secondary-button" onClick={onClose} disabled={busy}>
          Cancel
        </button>
        <button
          className="primary-button deployment-confirm"
          onClick={() => onDeploy(note)}
          disabled={busy || note.trim().length < 3}
        >
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <Rocket size={16} />
          )}
          {busy ? "Deploying" : "Deploy approved model"}
        </button>
      </footer>
    </dialog>
  );
}

function PredictionTester({
  active,
  onSessionExpired,
}: {
  active: ModelProposal;
  onSessionExpired: () => void;
}) {
  const deployment = activeDeployment(active);
  const schema = deployment?.input_schema ?? EMPTY_SCHEMA;
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState<{
    value: unknown;
    probabilities?: number[] | null;
  } | null>(null);

  useEffect(() => {
    setValues(Object.fromEntries(schema.map((field) => [field.name, ""])));
    setPrediction(null);
    setError("");
  }, [active.proposal_id, schema]);

  if (!deployment) {
    return (
      <section className="panel deployment-panel prediction-panel deployment-legacy-note">
        <Eye size={23} />
        <div>
          <h2>Prediction test is unavailable for this legacy deployment.</h2>
          <p>
            This model predates the dedicated deployment record. It remains
            available through the protected API, but the browser form needs a
            recorded input schema.
          </p>
        </div>
      </section>
    );
  }
  if (!schema.length) {
    return (
      <section className="panel deployment-panel prediction-panel deployment-legacy-note">
        <Eye size={23} />
        <div>
          <h2>No input schema was recorded for this candidate.</h2>
          <p>
            New controlled workflows save their feature schema for form-based
            prediction testing. This older candidate can still be inspected
            through its release record.
          </p>
        </div>
      </section>
    );
  }

  const complete = schema.every((field) => values[field.name]?.trim().length);
  async function testPrediction() {
    setBusy(true);
    setError("");
    setPrediction(null);
    try {
      const record = Object.fromEntries(
        schema.map((field) => [
          field.name,
          field.inferred_type === "numeric"
            ? Number(values[field.name])
            : values[field.name].trim(),
        ]),
      );
      const result = await api<{
        predictions: unknown[];
        probabilities?: number[][] | null;
      }>("/predict", {
        method: "POST",
        body: JSON.stringify({
          records: [record],
          include_probabilities:
            deployment?.task_type.includes("classification") ?? false,
        }),
      });
      setPrediction({
        value: result.predictions[0],
        probabilities: result.probabilities?.[0] || null,
      });
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) {
        onSessionExpired();
        return;
      }
      setError(
        cause instanceof Error ? cause.message : "Prediction testing failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel deployment-panel prediction-panel">
      <header className="panel-header">
        <div>
          <span className="eyebrow">SAFE ENDPOINT CHECK</span>
          <h2>Test one prediction</h2>
          <p>
            Enter one record using the active model’s recorded input fields.
          </p>
        </div>
        <Play size={20} />
      </header>
      <div className="prediction-form-grid">
        {schema.map((field: InputSchemaField) => (
          <label key={field.name}>
            <span>
              {field.name}
              <small>
                {field.inferred_type === "numeric" ? "Number" : "Category"}
              </small>
            </span>
            <input
              type={field.inferred_type === "numeric" ? "number" : "text"}
              inputMode={field.inferred_type === "numeric" ? "decimal" : "text"}
              value={values[field.name] || ""}
              onChange={(event) =>
                setValues((current) => ({
                  ...current,
                  [field.name]: event.target.value,
                }))
              }
              placeholder={
                field.inferred_type === "numeric" ? "e.g. 42" : "Enter a value"
              }
              disabled={busy}
            />
          </label>
        ))}
      </div>
      {error && (
        <div className="error-notice deployment-inline-error">
          <CircleAlert size={16} /> {error}
        </div>
      )}
      <div className="prediction-action-row">
        <span>
          <ShieldCheck size={14} /> One record · protected endpoint · no dataset
          upload
        </span>
        <button
          className="primary-button"
          onClick={() => void testPrediction()}
          disabled={busy || !complete}
        >
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <Send size={16} />
          )}
          {busy ? "Testing" : "Run test prediction"}
        </button>
      </div>
      {prediction && (
        <section className="prediction-result" aria-live="polite">
          <span className="prediction-result-icon">
            <BadgeCheck size={19} />
          </span>
          <div>
            <small>PREDICTED RESULT</small>
            <strong>{String(prediction.value)}</strong>
          </div>
          {prediction.probabilities && (
            <div className="prediction-probabilities">
              {prediction.probabilities.map((probability, index) => (
                <span key={`${index}-${probability}`}>
                  {deployment.class_labels[index] || `Class ${index + 1}`}
                  <strong>{(probability * 100).toFixed(1)}%</strong>
                </span>
              ))}
            </div>
          )}
        </section>
      )}
    </section>
  );
}

export function DeploymentPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const canDeploy = user.permissions.includes("deployments:write");
  const [proposals, setProposals] = useState<ModelProposal[]>([]);
  const [status, setStatus] = useState<DeploymentStatus | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [showDeployDialog, setShowDeployDialog] = useState(false);
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
          : "Deployment records could not be loaded.",
      );
    },
    [onSessionExpired],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [nextProposals, nextStatus] = await Promise.all([
        api<ModelProposal[]>("/models/proposals"),
        api<DeploymentStatus>("/deployments"),
      ]);
      const approved = nextProposals.filter(
        (proposal) => proposal.status === "approved_for_deployment",
      );
      setProposals(nextProposals);
      setStatus(nextStatus);
      setSelectedId((current) =>
        approved.some((proposal) => proposal.proposal_id === current)
          ? current
          : approved[0]?.proposal_id || "",
      );
    } catch (cause) {
      handleError(cause);
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const approved = proposals.filter(
    (proposal) => proposal.status === "approved_for_deployment",
  );
  const selected =
    approved.find((proposal) => proposal.proposal_id === selectedId) || null;
  const active = status?.active || null;
  const deployment = activeDeployment(active);

  async function deploy(note: string) {
    if (!selected || !canDeploy) return;
    setDeploying(true);
    setError("");
    try {
      await api<ModelProposal>(`/deployments/${selected.proposal_id}`, {
        method: "POST",
        body: JSON.stringify({ deployment_note: note.trim() }),
      });
      setShowDeployDialog(false);
      await refresh();
    } catch (cause) {
      handleError(cause);
    } finally {
      setDeploying(false);
    }
  }

  return (
    <div className="deployment-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STAGE 07 · MODEL DEPLOYMENT</div>
          <h1>Release only what passed human review.</h1>
          <p>
            Activate an approved model deliberately, retain its release
            evidence, then test the protected endpoint.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading || deploying}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          deployment
        </button>
      </div>

      <section
        className={`deployment-hero ${status?.state === "ready" ? "ready" : "empty"}`}
      >
        <div className="deployment-hero-icon">
          <Rocket size={25} />
        </div>
        <div>
          <span className="eyebrow">LOCAL SERVING ENDPOINT</span>
          <h2>
            {status?.state === "ready"
              ? "Endpoint ready for controlled testing."
              : "No model is deployed yet."}
          </h2>
          <p>
            {status?.state === "ready"
              ? `${displayModel(active?.metadata.model_name)} is active at ${status?.method || "POST"} ${status?.endpoint || "/predict"}.`
              : "A Model Reviewer must approve a candidate before an Administrator can release it to the local endpoint."}
          </p>
        </div>
        <span
          className={`deployment-state ${status?.state === "ready" ? "ready" : "empty"}`}
        >
          <span /> {status?.state === "ready" ? "Ready" : "Not deployed"}
        </span>
      </section>

      {!canDeploy && (
        <section className="deployment-readonly">
          <Eye size={17} /> You can view endpoint and release information. Only
          an MLOps Administrator can deploy an approved candidate.
        </section>
      )}
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section
        className="deployment-status-grid"
        aria-label="Deployment status"
      >
        <article>
          <span className="deployment-stat-icon blue">
            <Rocket size={18} />
          </span>
          <small>Active version</small>
          <strong>{deployment?.version || "None"}</strong>
          <p>
            {deployment
              ? `Released ${formatDate(deployment.deployed_at)}`
              : "No active release"}
          </p>
        </article>
        <article>
          <span className="deployment-stat-icon teal">
            <BadgeCheck size={18} />
          </span>
          <small>Endpoint status</small>
          <strong>{status?.state === "ready" ? "Ready" : "Offline"}</strong>
          <p>
            {status?.state === "ready"
              ? "Protected local API"
              : "No model is serving"}
          </p>
        </article>
        <article>
          <span className="deployment-stat-icon purple">
            <History size={18} />
          </span>
          <small>Retained releases</small>
          <strong>{status?.releases.length ?? 0}</strong>
          <p>Immutable local release records</p>
        </article>
      </section>

      <section className="deployment-layout">
        <article className="panel deployment-panel deployment-active-card">
          <header className="panel-header">
            <div>
              <span className="eyebrow">ACTIVE PRODUCTION MODEL</span>
              <h2>
                {active
                  ? displayModel(active.metadata.model_name)
                  : "Waiting for an approved release"}
              </h2>
              <p>
                {active
                  ? "Current local endpoint configuration."
                  : "No production model is available yet."}
              </p>
            </div>
            <Gauge size={20} />
          </header>
          {active && deployment ? (
            <div className="deployment-active-details">
              <div>
                <small>PRIMARY METRIC</small>
                <strong>
                  {displayMetric(active.metadata.metric || deployment.metric)}
                </strong>
                <span>
                  {formatScore(active.metadata.holdout_score)} final holdout
                </span>
              </div>
              <div>
                <small>DEPLOYED BY</small>
                <strong>{deployment.deployed_by}</strong>
                <span>{formatDate(deployment.deployed_at)}</span>
              </div>
              <div>
                <small>PREDICTION TARGET</small>
                <strong>
                  {active.metadata.plan?.target_column || "Not recorded"}
                </strong>
                <span>{deployment.feature_columns.length} input fields</span>
              </div>
              <div className="deployment-note">
                <small>RELEASE NOTE</small>
                <p>{deployment.deployment_note}</p>
              </div>
            </div>
          ) : (
            <div className="deployment-empty-active">
              <Rocket size={24} />
              <p>
                Approve a candidate in the Approval Center, then return here to
                deploy it.
              </p>
              <a className="text-button" href="#approval">
                Open Approval Center <ArrowRight size={14} />
              </a>
            </div>
          )}
        </article>

        <article className="panel deployment-panel endpoint-card">
          <header className="panel-header">
            <div>
              <span className="eyebrow">SERVING CONTRACT</span>
              <h2>Protected prediction endpoint</h2>
              <p>Only a signed-in workspace role may test predictions.</p>
            </div>
            <ShieldCheck size={20} />
          </header>
          <div className="endpoint-contract">
            <span className="endpoint-method">POST</span>
            <code>{status?.endpoint || "/predict"}</code>
            <span className="endpoint-auth">
              <ShieldCheck size={13} /> Session protected
            </span>
          </div>
          <ul>
            <li>
              <Check size={14} /> Uses the active release only
            </li>
            <li>
              <Check size={14} /> Aligns request fields to the fitted model
              schema
            </li>
            <li>
              <Check size={14} /> Does not accept or execute arbitrary code
            </li>
          </ul>
        </article>
      </section>

      <section className="deployment-section-heading">
        <div>
          <span className="eyebrow">ADMINISTRATOR RELEASE</span>
          <h2>Deploy a reviewer-approved candidate</h2>
        </div>
        <span>
          {approved.length} candidate{approved.length === 1 ? "" : "s"} ready
        </span>
      </section>
      <section className="deployment-release-panel">
        <div className="deployment-release-copy">
          <span>
            <ClipboardCheck size={19} />
          </span>
          <div>
            <h3>Reviewer approval is required first.</h3>
            <p>
              Deployment retains a versioned release copy, records the
              administrator and release note, then switches the local active
              model.
            </p>
          </div>
        </div>
        {loading ? (
          <div className="deployment-loading">
            <LoaderCircle className="spin" size={17} /> Reading approved
            candidates…
          </div>
        ) : approved.length ? (
          <div className="deployment-release-controls">
            <label>
              <span>Approved candidate</span>
              <select
                value={selectedId}
                onChange={(event) => setSelectedId(event.target.value)}
                disabled={!canDeploy || deploying}
              >
                {approved.map((proposal) => (
                  <option
                    key={proposal.proposal_id}
                    value={proposal.proposal_id}
                  >
                    {displayModel(proposal.metadata.model_name)} ·{" "}
                    {formatScore(proposal.metadata.holdout_score)}{" "}
                    {displayMetric(proposal.metadata.metric)}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="primary-button"
              onClick={() => setShowDeployDialog(true)}
              disabled={!canDeploy || !selected || deploying}
            >
              <Rocket size={16} /> Deploy selected model
            </button>
          </div>
        ) : (
          <div className="deployment-no-approved">
            <Clock3 size={17} /> No reviewer-approved candidate is available.
          </div>
        )}
      </section>

      {active && (
        <PredictionTester active={active} onSessionExpired={onSessionExpired} />
      )}

      <section className="panel deployment-panel release-history">
        <header className="panel-header">
          <div>
            <span className="eyebrow">RELEASE HISTORY</span>
            <h2>Retained deployment versions</h2>
            <p>
              Each Stage 07 deployment preserves its associated release metadata
              and model artifact locally.
            </p>
          </div>
          <History size={20} />
        </header>
        {!status?.releases.length ? (
          <div className="deployment-history-empty">
            <History size={22} /> Your release history will appear here after
            the first deployment.
          </div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>VERSION</th>
                  <th>MODEL / TARGET</th>
                  <th>DEPLOYED BY</th>
                  <th>TIME</th>
                  <th>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {status.releases.map((release) => {
                  const detail = activeDeployment(release);
                  return (
                    <tr key={release.proposal_id}>
                      <td>
                        <strong>{detail?.version || "Legacy release"}</strong>
                        <small>{release.proposal_id.slice(0, 12)}</small>
                      </td>
                      <td>
                        <strong>
                          {displayModel(release.metadata.model_name)}
                        </strong>
                        <small>
                          {release.metadata.plan?.target_column ||
                            "Target not recorded"}
                        </small>
                      </td>
                      <td>{detail?.deployed_by || "Legacy record"}</td>
                      <td>{formatDate(detail?.deployed_at)}</td>
                      <td>
                        <span className="status-badge status-deployed">
                          <span /> Deployed
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="deployment-next-stage">
        <span>
          <Target size={21} />
        </span>
        <div>
          <span className="eyebrow">STAGE 08 NEXT</span>
          <h3>Observe drift before it becomes a model issue.</h3>
          <p>
            Monitoring now compares incoming feature data with the saved model
            baseline and records controlled retraining requests—never automatic
            replacement.
          </p>
        </div>
        <a href="#monitoring">
          Open monitoring <ArrowRight size={13} />
        </a>
      </section>

      {showDeployDialog && selected && (
        <DeployDialog
          proposal={selected}
          busy={deploying}
          onClose={() => setShowDeployDialog(false)}
          onDeploy={(note) => void deploy(note)}
        />
      )}
    </div>
  );
}
