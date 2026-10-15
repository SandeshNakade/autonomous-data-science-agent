import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Box,
  CheckCircle2,
  CircleAlert,
  ClipboardCheck,
  Clock3,
  Eye,
  FlaskConical,
  GitBranch,
  History,
  LoaderCircle,
  PackageCheck,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { api, ApiError } from "./api";
import type {
  DeploymentStatus,
  ModelProposal,
  ProposalAuditEvent,
} from "./api";
import "./model-registry.css";

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

function modelName(value: string | undefined) {
  return MODEL_NAMES[value || ""] || titleCase(value);
}

function metricName(value: string | undefined) {
  return METRIC_NAMES[value || ""] || titleCase(value || "Metric");
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value))
    return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
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

function statusLabel(status: string | undefined) {
  const labels: Record<string, string> = {
    pending_human_approval: "Awaiting review",
    approved_for_deployment: "Approved to deploy",
    rejected: "Rejected",
    deployed: "Deployed",
    promoted: "Legacy promoted",
  };
  return labels[status || ""] || titleCase(status);
}

function statusClass(status: string | undefined) {
  if (status === "rejected") return "rejected";
  if (status === "pending_human_approval") return "pending";
  if (status === "approved_for_deployment") return "approved";
  if (status === "deployed" || status === "promoted") return "deployed";
  return "neutral";
}

function RegistryStatus({ status }: { status: string | undefined }) {
  return (
    <span className={`registry-status ${statusClass(status)}`}>
      <span /> {statusLabel(status)}
    </span>
  );
}

function AuditEvent({ event }: { event: ProposalAuditEvent }) {
  return (
    <li>
      <span
        className={`registry-audit-dot ${event.action.includes("reject") ? "rejected" : ""}`}
      />
      <div>
        <strong>{titleCase(event.action)}</strong>
        <p>{event.reason || "No reason recorded."}</p>
        <small>
          {event.actor || "System"} · {formatDate(event.at)}
        </small>
      </div>
    </li>
  );
}

export function ModelRegistryPage({
  onSessionExpired,
}: {
  onSessionExpired: () => void;
}) {
  const [proposals, setProposals] = useState<ModelProposal[]>([]);
  const [deployment, setDeployment] = useState<DeploymentStatus | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
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
          : "The model registry could not be loaded.",
      );
    },
    [onSessionExpired],
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [nextProposals, nextDeployment] = await Promise.all([
        api<ModelProposal[]>("/models/proposals"),
        api<DeploymentStatus>("/deployments"),
      ]);
      setProposals(nextProposals);
      setDeployment(nextDeployment);
      setSelectedId((current) =>
        nextProposals.some((proposal) => proposal.proposal_id === current)
          ? current
          : nextDeployment.active?.proposal_id ||
            nextProposals[0]?.proposal_id ||
            "",
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

  const activeId = deployment?.active?.proposal_id;
  const visible = useMemo(() => {
    const query = search.trim().toLowerCase();
    return proposals.filter((proposal) => {
      const matchesFilter = filter === "all" || proposal.status === filter;
      const matchesSearch =
        !query ||
        `${proposal.metadata.model_name} ${proposal.proposal_id} ${proposal.metadata.run_id}`
          .toLowerCase()
          .includes(query);
      return matchesFilter && matchesSearch;
    });
  }, [filter, proposals, search]);
  const selected =
    proposals.find((proposal) => proposal.proposal_id === selectedId) || null;
  const pending = proposals.filter(
    (proposal) => proposal.status === "pending_human_approval",
  ).length;
  const approved = proposals.filter(
    (proposal) => proposal.status === "approved_for_deployment",
  ).length;
  const releases = deployment?.releases.length || 0;

  return (
    <div className="model-registry-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">SUPPORTING PAGE · MODEL REGISTRY</div>
          <h1>Trace every candidate from evidence to release.</h1>
          <p>
            Inspect candidate model states, reviewer decisions, retained
            releases, and serving lineage without accessing model artifacts.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          registry
        </button>
      </div>
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section className="registry-hero">
        <span className="registry-hero-icon">
          <Box size={24} />
        </span>
        <div>
          <span className="eyebrow">LOCAL MODEL REGISTRY</span>
          <h2>
            {deployment?.state === "ready"
              ? "A reviewed release is active."
              : "Model candidates remain under human control."}
          </h2>
          <p>
            {deployment?.state === "ready"
              ? `${deployment.active?.deployment?.version || "Active release"} is serving locally. Its candidate and release evidence are retained below.`
              : "Candidates become deployable only after an independent reviewer approval. The registry never silently promotes a model."}
          </p>
        </div>
        <span
          className={`registry-hero-state ${deployment?.state === "ready" ? "ready" : "waiting"}`}
        >
          <span />{" "}
          {deployment?.state === "ready"
            ? "Active release"
            : "No active release"}
        </span>
      </section>

      <section
        className="registry-status-grid"
        aria-label="Model registry status"
      >
        <article>
          <span className="registry-stat-icon blue">
            <Box size={18} />
          </span>
          <small>Registered candidates</small>
          <strong>{proposals.length}</strong>
          <p>Saved model proposals</p>
        </article>
        <article>
          <span className="registry-stat-icon purple">
            <ClipboardCheck size={18} />
          </span>
          <small>Awaiting review</small>
          <strong>{pending}</strong>
          <p>Reviewer decision required</p>
        </article>
        <article>
          <span className="registry-stat-icon teal">
            <CheckCircle2 size={18} />
          </span>
          <small>Ready to deploy</small>
          <strong>{approved}</strong>
          <p>Approved candidates</p>
        </article>
        <article>
          <span className="registry-stat-icon amber">
            <PackageCheck size={18} />
          </span>
          <small>Retained releases</small>
          <strong>{releases}</strong>
          <p>Immutable local records</p>
        </article>
      </section>

      <section className="panel registry-library-panel">
        <header className="panel-header">
          <div>
            <span className="eyebrow">MODEL CATALOG</span>
            <h2>Registered candidates and releases</h2>
            <p>
              Each record is created by the controlled workflow and retains its
              review and release trail.
            </p>
          </div>
          <SlidersHorizontal size={20} />
        </header>
        <div className="registry-filters">
          <label className="registry-search">
            <Search size={16} />
            <span className="sr-only">Search model registry</span>
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search model, proposal, or run ID"
            />
          </label>
          <label>
            <span>Registry state</span>
            <select
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="all">All states</option>
              <option value="pending_human_approval">Awaiting review</option>
              <option value="approved_for_deployment">
                Approved to deploy
              </option>
              <option value="deployed">Deployed</option>
              <option value="rejected">Rejected</option>
            </select>
          </label>
          <span>
            {visible.length} of {proposals.length} shown
          </span>
        </div>
        {loading ? (
          <div className="registry-empty-inline">
            <LoaderCircle className="spin" size={18} /> Reading registry
            records…
          </div>
        ) : !visible.length ? (
          <div className="registry-empty-inline">
            <Box size={18} /> No registry record matches these filters.
          </div>
        ) : (
          <div className="table-scroll registry-table">
            <table>
              <thead>
                <tr>
                  <th>MODEL CANDIDATE</th>
                  <th>VALIDATION</th>
                  <th>HOLDOUT</th>
                  <th>CREATED</th>
                  <th>STATE</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {visible.map((proposal) => (
                  <tr
                    key={proposal.proposal_id}
                    className={
                      selectedId === proposal.proposal_id
                        ? "registry-selected-row"
                        : ""
                    }
                  >
                    <td>
                      <strong>{modelName(proposal.metadata.model_name)}</strong>
                      <small>
                        {proposal.proposal_id.slice(0, 12)} ·{" "}
                        {metricName(proposal.metadata.metric)}
                      </small>
                    </td>
                    <td>{formatScore(proposal.metadata.validation_score)}</td>
                    <td>{formatScore(proposal.metadata.holdout_score)}</td>
                    <td>{formatDate(proposal.created_at)}</td>
                    <td>
                      <RegistryStatus status={proposal.status} />
                    </td>
                    <td>
                      <button
                        className="registry-open"
                        onClick={() => setSelectedId(proposal.proposal_id)}
                        aria-label={`View ${modelName(proposal.metadata.model_name)} registry record`}
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

      <section className="registry-section-heading">
        <div>
          <span className="eyebrow">SELECTED RECORD</span>
          <h2>Model lineage &amp; governance</h2>
        </div>
        {selected && <RegistryStatus status={selected.status} />}
      </section>
      {selected ? (
        <>
          <section
            className={`registry-detail-hero ${selected.proposal_id === activeId ? "active" : ""}`}
          >
            <div>
              <span className="eyebrow">
                {selected.proposal_id.slice(0, 12)} · CANDIDATE RECORD
              </span>
              <h2>{modelName(selected.metadata.model_name)}</h2>
              <p>
                {selected.proposal_id === activeId
                  ? "This candidate is the active local serving release."
                  : selected.status === "approved_for_deployment"
                    ? "This candidate passed review and is ready for an administrator release."
                    : "This record retains the evidence and governance state for its model candidate."}
              </p>
            </div>
            {selected.status === "pending_human_approval" ? (
              <a className="text-button" href="#approval">
                Open approval <ArrowRight size={14} />
              </a>
            ) : selected.status === "approved_for_deployment" ? (
              <a className="text-button" href="#deployment">
                Open deployment <ArrowRight size={14} />
              </a>
            ) : null}
          </section>
          <section className="registry-detail-grid">
            <article className="panel registry-evidence-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">EVALUATION EVIDENCE</span>
                  <h2>Stored model scores</h2>
                </div>
                <FlaskConical size={20} />
              </header>
              <div className="registry-evidence-facts">
                <span>
                  <small>PRIMARY METRIC</small>
                  <strong>{metricName(selected.metadata.metric)}</strong>
                </span>
                <span>
                  <small>VALIDATION</small>
                  <strong>
                    {formatScore(selected.metadata.validation_score)}
                  </strong>
                </span>
                <span>
                  <small>FINAL HOLDOUT</small>
                  <strong>
                    {formatScore(selected.metadata.holdout_score)}
                  </strong>
                </span>
              </div>
            </article>
            <article className="panel registry-lineage-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">EXPERIMENT LINEAGE</span>
                  <h2>Training provenance</h2>
                </div>
                <GitBranch size={20} />
              </header>
              <div className="registry-lineage-facts">
                <span>
                  <small>WORKFLOW RUN</small>
                  <strong>
                    {selected.metadata.run_id?.slice(0, 16) || "Not recorded"}
                  </strong>
                </span>
                <span>
                  <small>EXPERIMENT</small>
                  <strong>
                    {selected.metadata.experiment_id || "Not recorded"}
                  </strong>
                </span>
                <span>
                  <small>DATASET HASH</small>
                  <strong>
                    {selected.metadata.dataset_sha256?.slice(0, 16) ||
                      "Not recorded"}
                  </strong>
                </span>
              </div>
            </article>
            <article className="panel registry-release-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">RELEASE STATE</span>
                  <h2>Deployment evidence</h2>
                </div>
                <PackageCheck size={20} />
              </header>
              {selected.deployment ? (
                <div className="registry-release-body">
                  <span className="registry-release-icon">
                    <CheckCircle2 size={18} />
                  </span>
                  <div>
                    <strong>{selected.deployment.version}</strong>
                    <p>
                      Released by {selected.deployment.deployed_by} ·{" "}
                      {formatDate(selected.deployment.deployed_at)}
                    </p>
                    <small>{selected.deployment.deployment_note}</small>
                  </div>
                </div>
              ) : (
                <div className="registry-empty-inline">
                  <Clock3 size={18} /> No release has been recorded for this
                  candidate.
                </div>
              )}
            </article>
          </section>
          <section className="registry-governance-grid">
            <article className="panel registry-plan-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">SAVED PLAN</span>
                  <h2>Decision context</h2>
                </div>
                <Target size={20} />
              </header>
              {selected.metadata.plan ? (
                <div className="registry-plan-facts">
                  <span>
                    <small>TARGET</small>
                    <strong>{selected.metadata.plan.target_column}</strong>
                  </span>
                  <span>
                    <small>TASK</small>
                    <strong>
                      {titleCase(selected.metadata.plan.task_type)}
                    </strong>
                  </span>
                  <span>
                    <small>ACCEPTANCE GATE</small>
                    <strong>
                      {formatScore(selected.metadata.plan.acceptance_threshold)}
                    </strong>
                  </span>
                  <span>
                    <small>MODEL BUDGET</small>
                    <strong>{selected.metadata.plan.max_experiments}</strong>
                  </span>
                </div>
              ) : (
                <div className="registry-empty-inline">
                  <Target size={18} /> Plan details were not stored for this
                  record.
                </div>
              )}
            </article>
            <article className="panel registry-audit-card">
              <header className="panel-header">
                <div>
                  <span className="eyebrow">GOVERNANCE AUDIT</span>
                  <h2>Recorded decisions</h2>
                </div>
                <History size={20} />
              </header>
              {selected.audit_events?.length ? (
                <ol className="registry-audit-list">
                  {selected.audit_events.map((event, index) => (
                    <AuditEvent key={`${event.at}-${index}`} event={event} />
                  ))}
                </ol>
              ) : (
                <div className="registry-empty-inline">
                  <Eye size={18} /> No governance events have been recorded yet.
                </div>
              )}
            </article>
          </section>
          <section className="registry-guardrail">
            <ShieldCheck size={17} />
            <div>
              <strong>
                Registry records are evidence, not an execution surface.
              </strong>
              <p>
                Only the separate Approval Center and Deployment pages can alter
                a model’s lifecycle state, under their role permissions.
              </p>
            </div>
          </section>
        </>
      ) : (
        <section className="panel registry-detail-loading">
          <Box size={20} /> Select a model record to inspect its lineage and
          governance evidence.
        </section>
      )}
    </div>
  );
}
