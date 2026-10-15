import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BadgeCheck,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleAlert,
  ClipboardCheck,
  Clock3,
  Database,
  Eye,
  FileCheck2,
  FlaskConical,
  Gauge,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Target,
  ThumbsDown,
  ThumbsUp,
  UserRoundCheck,
  X,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { ModelProposal, ProposalAuditEvent, User } from "./api";
import "./approval.css";

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

function modelName(value: string | undefined) {
  return MODEL_NAMES[value || ""] || humanize(value || "Not recorded");
}

function metricName(value: string | undefined) {
  return METRIC_NAMES[value || ""] || humanize(value || "Metric");
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

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    pending_human_approval: "Awaiting review",
    approved_for_deployment: "Approved for deployment",
    rejected: "Rejected",
    deployed: "Deployed",
    promoted: "Promoted (legacy)",
  };
  return labels[status] || humanize(status);
}

function auditLabel(action: string) {
  const labels: Record<string, string> = {
    submitted_for_review: "Candidate submitted for review",
    approved_for_deployment: "Approved for deployment",
    rejected: "Candidate rejected",
    deployed: "Deployed to local endpoint",
  };
  return labels[action] || humanize(action);
}

function StatusIcon({ status }: { status: string }) {
  if (
    status === "approved_for_deployment" ||
    status === "deployed" ||
    status === "promoted"
  )
    return <BadgeCheck size={17} />;
  if (status === "rejected") return <ThumbsDown size={17} />;
  return <ClipboardCheck size={17} />;
}

function ReviewDialog({
  proposal,
  decision,
  submitting,
  onCancel,
  onSubmit,
}: {
  proposal: ModelProposal;
  decision: "approve" | "reject";
  submitting: boolean;
  onCancel: () => void;
  onSubmit: (reason: string) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [reason, setReason] = useState("");
  const isApproval = decision === "approve";
  useEffect(() => {
    dialogRef.current?.showModal();
    return () => dialogRef.current?.close();
  }, []);
  return (
    <dialog
      ref={dialogRef}
      className="dialog approval-dialog"
      aria-labelledby="review-dialog-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!submitting) onCancel();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">
            {isApproval ? <ThumbsUp size={13} /> : <ThumbsDown size={13} />}
            {isApproval ? "REVIEW APPROVAL" : "REVIEW REJECTION"}
          </span>
          <h2 id="review-dialog-title">
            {isApproval
              ? "Approve this candidate for deployment?"
              : "Reject this candidate?"}
          </h2>
        </div>
        <button
          className="icon-button"
          onClick={onCancel}
          disabled={submitting}
          aria-label="Cancel review"
        >
          <X size={18} />
        </button>
      </header>
      <p className="dialog-intro">
        You are reviewing{" "}
        <strong>{modelName(proposal.metadata.model_name)}</strong>. Your
        decision and reason will be recorded in its audit trail.
      </p>
      <div
        className={`approval-dialog-guardrail ${isApproval ? "approve" : "reject"}`}
      >
        <ShieldCheck size={18} />
        <span>
          {isApproval
            ? "This marks the candidate approved for a later deployment step. It does not deploy, replace, or serve a production model."
            : "This keeps the candidate out of the deployment path. A Data Scientist can revise and submit a later candidate."}
        </span>
      </div>
      <label className="approval-reason-field">
        <span>
          Decision reason <em>required</em>
        </span>
        <textarea
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          placeholder={
            isApproval
              ? "Explain why the evidence is sufficient for deployment review."
              : "Explain what evidence or limitation requires revision."
          }
          disabled={submitting}
          minLength={3}
          maxLength={1200}
          autoFocus
        />
        <small>{reason.trim().length}/1200 characters</small>
      </label>
      <footer className="approval-dialog-actions">
        <button
          className="secondary-button"
          onClick={onCancel}
          disabled={submitting}
        >
          Cancel
        </button>
        <button
          className={`primary-button ${isApproval ? "approval-confirm" : "reject-confirm"}`}
          onClick={() => onSubmit(reason)}
          disabled={submitting || reason.trim().length < 3}
        >
          {submitting ? (
            <LoaderCircle className="spin" size={16} />
          ) : isApproval ? (
            <ThumbsUp size={15} />
          ) : (
            <ThumbsDown size={15} />
          )}
          {submitting
            ? "Recording decision"
            : isApproval
              ? "Approve for deployment"
              : "Record rejection"}
        </button>
      </footer>
    </dialog>
  );
}

function AuditTrail({ events }: { events: ProposalAuditEvent[] }) {
  if (!events.length) {
    return (
      <div className="approval-empty-inline">
        <Clock3 size={18} /> No audit event was recorded for this legacy
        candidate.
      </div>
    );
  }
  return (
    <ol className="approval-audit-list">
      {events.map((event, index) => (
        <li key={`${event.at}-${event.action}-${index}`}>
          <span className={`approval-audit-icon audit-${event.action}`}>
            {event.action === "rejected" ? (
              <ThumbsDown size={15} />
            ) : event.action === "approved_for_deployment" ? (
              <BadgeCheck size={15} />
            ) : (
              <FileCheck2 size={15} />
            )}
          </span>
          <div>
            <div>
              <strong>{auditLabel(event.action)}</strong>
              <time>{formatDate(event.at)}</time>
            </div>
            <p>{event.reason || "No reason was recorded."}</p>
            <small>{event.actor || "Unknown actor"}</small>
          </div>
        </li>
      ))}
    </ol>
  );
}

function CandidateDetails({ proposal }: { proposal: ModelProposal }) {
  const metadata = proposal.metadata;
  const plan = metadata.plan;
  const primaryMetric = metadata.metric || plan?.metric;
  return (
    <div className="approval-report" aria-live="polite">
      <section className={`approval-result-hero status-${proposal.status}`}>
        <div>
          <span className="eyebrow">
            <UserRoundCheck size={12} /> HUMAN REVIEW RECORD
          </span>
          <h2>{statusLabel(proposal.status)}</h2>
          <p>
            {proposal.status === "pending_human_approval"
              ? "This candidate passed the controlled workflow gate and is waiting for an explicit review decision."
              : proposal.status === "approved_for_deployment"
                ? "A reviewer approved this candidate for a later deployment step. It is not a production model."
                : proposal.status === "rejected"
                  ? "The review decision has been recorded. The candidate stays outside the deployment path."
                  : proposal.status === "deployed"
                    ? "This reviewer-approved candidate was released through the dedicated deployment path."
                    : "This candidate was handled through the prior combined promotion flow."}
          </p>
        </div>
        <span className={`approval-status-pill status-${proposal.status}`}>
          <StatusIcon status={proposal.status} /> {statusLabel(proposal.status)}
        </span>
      </section>

      <section
        className="approval-summary-grid"
        aria-label="Candidate evidence summary"
      >
        <article>
          <span className="approval-stat-icon purple">
            <FlaskConical size={18} />
          </span>
          <small>Candidate model</small>
          <strong>{modelName(metadata.model_name)}</strong>
          <p>{metadata.experiment_id?.slice(-12) || "Recorded candidate"}</p>
        </article>
        <article>
          <span className="approval-stat-icon blue">
            <Target size={18} />
          </span>
          <small>Prediction target</small>
          <strong>{plan?.target_column || "Not recorded"}</strong>
          <p>{plan ? humanize(plan.task_type) : "Plan unavailable"}</p>
        </article>
        <article>
          <span className="approval-stat-icon teal">
            <Gauge size={18} />
          </span>
          <small>Validation evidence</small>
          <strong>{formatScore(metadata.validation_score)}</strong>
          <p>{metricName(primaryMetric)}</p>
        </article>
        <article>
          <span className="approval-stat-icon amber">
            <BadgeCheck size={18} />
          </span>
          <small>Final holdout</small>
          <strong>{formatScore(metadata.holdout_score)}</strong>
          <p>{metricName(primaryMetric)} · untouched split</p>
        </article>
      </section>

      <section className="approval-review-note">
        <span>
          <ShieldCheck size={21} />
        </span>
        <div>
          <h3>Approval is deliberately not deployment.</h3>
          <p>
            Reviewers decide whether the evidence supports a candidate
            progressing. A separate administrator deployment step is still
            required before any endpoint can use it.
          </p>
        </div>
        <a className="approval-review-pill" href="#deployment">
          Open deployment <ArrowRight size={12} />
        </a>
      </section>

      <section className="approval-detail-grid">
        <article className="panel approval-panel evidence-panel">
          <header className="panel-header">
            <div>
              <h2>Evidence available for review</h2>
              <p>
                Metric values from the saved workflow and its single untouched
                holdout evaluation.
              </p>
            </div>
            <Eye size={20} />
          </header>
          <div className="approval-evidence-list">
            <div>
              <span>
                <Gauge size={15} />
              </span>
              <p>
                <small>QUALITY GATE</small>
                <strong>
                  {plan
                    ? `${metricName(plan.metric)} ${plan.metric === "rmse" ? "≤" : "≥"} ${plan.acceptance_threshold}`
                    : "Not recorded"}
                </strong>
              </p>
            </div>
            <div>
              <span>
                <Check size={15} />
              </span>
              <p>
                <small>VALIDATION RESULT</small>
                <strong>
                  {formatScore(metadata.validation_score)}{" "}
                  {metricName(primaryMetric)}
                </strong>
              </p>
            </div>
            <div>
              <span>
                <BadgeCheck size={15} />
              </span>
              <p>
                <small>FINAL HOLDOUT RESULT</small>
                <strong>
                  {formatScore(metadata.holdout_score)}{" "}
                  {metricName(primaryMetric)}
                </strong>
              </p>
            </div>
            <div>
              <span>
                <Database size={15} />
              </span>
              <p>
                <small>RUN REFERENCE</small>
                <strong>
                  {metadata.run_id?.slice(0, 14) || "Not recorded"}
                </strong>
              </p>
            </div>
          </div>
          <a
            className="text-button approval-comparison-link"
            href="#comparison"
          >
            Compare workflow experiments <ArrowRight size={14} />
          </a>
        </article>
        <article className="panel approval-panel audit-panel">
          <header className="panel-header">
            <div>
              <h2>Decision audit trail</h2>
              <p>Who recorded each state change and why.</p>
            </div>
            <ClipboardCheck size={20} />
          </header>
          <AuditTrail events={proposal.audit_events || []} />
        </article>
      </section>

      {proposal.review && (
        <section
          className={`approval-recorded-decision ${proposal.review.decision}`}
        >
          {proposal.review.decision === "approve" ? (
            <ThumbsUp size={19} />
          ) : (
            <ThumbsDown size={19} />
          )}
          <div>
            <small>RECORDED REVIEW DECISION</small>
            <h3>
              {proposal.review.decision === "approve"
                ? "Approved for deployment review"
                : "Rejected for revision"}
            </h3>
            <p>“{proposal.review.reason}”</p>
          </div>
          <span>
            {proposal.review.reviewed_by}
            <br />
            {formatDate(proposal.review.reviewed_at)}
          </span>
        </section>
      )}

      <section className="approval-next-stage">
        <span>
          <BrainCircuit size={21} />
        </span>
        <div>
          <span className="eyebrow">STAGE 07 NEXT</span>
          <h3>Deploy only what a reviewer approved.</h3>
          <p>
            Deployment will be an administrator-controlled operation with model
            versions, endpoint status, and prediction checks.
          </p>
        </div>
        <a className="stage-next-badge" href="#deployment">
          Open deployment <ArrowRight size={12} />
        </a>
      </section>
    </div>
  );
}

export function ApprovalPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const canReview = user.permissions.includes("reviews:write");
  const [proposals, setProposals] = useState<ModelProposal[]>([]);
  const [selected, setSelected] = useState<ModelProposal | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [dialogDecision, setDialogDecision] = useState<
    "approve" | "reject" | null
  >(null);
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
          : "The approval data could not be loaded.",
      );
    },
    [onSessionExpired],
  );

  const refreshProposals = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const next = await api<ModelProposal[]>("/models/proposals");
      setProposals(next);
      setSelected(
        (current) =>
          next.find((item) => item.proposal_id === current?.proposal_id) ||
          next[0] ||
          null,
      );
    } catch (cause) {
      handleError(cause);
    } finally {
      setLoading(false);
    }
  }, [handleError]);

  useEffect(() => {
    void refreshProposals();
  }, [refreshProposals]);

  async function submitReview(reason: string) {
    if (!selected || !dialogDecision || !canReview) return;
    setSubmitting(true);
    setError("");
    try {
      const reviewed = await api<ModelProposal>(
        `/model-reviews/${selected.proposal_id}`,
        {
          method: "POST",
          body: JSON.stringify({
            decision: dialogDecision,
            reason: reason.trim(),
          }),
        },
      );
      setSelected(reviewed);
      setProposals((items) =>
        items.map((item) =>
          item.proposal_id === reviewed.proposal_id ? reviewed : item,
        ),
      );
      setDialogDecision(null);
    } catch (cause) {
      handleError(cause);
    } finally {
      setSubmitting(false);
    }
  }

  const pendingCount = proposals.filter(
    (item) => item.status === "pending_human_approval",
  ).length;
  const approvedCount = proposals.filter(
    (item) => item.status === "approved_for_deployment",
  ).length;
  const rejectedCount = proposals.filter(
    (item) => item.status === "rejected",
  ).length;
  const pendingSelected = selected?.status === "pending_human_approval";

  return (
    <div className="approval-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">STAGE 06 · MODEL APPROVAL CENTER</div>
          <h1>Keep the final model decision human.</h1>
          <p>
            Review candidate evidence, record an accountable decision, and keep
            deployment as a separate operation.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refreshProposals()}
          disabled={loading || submitting}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          proposals
        </button>
      </div>

      <section className="approval-intro">
        <span className="approval-intro-icon">
          <UserRoundCheck size={22} />
        </span>
        <div>
          <h2>Evidence informs the decision. A person records it.</h2>
          <p>
            Only a Model Reviewer or Administrator can approve or reject a
            candidate. Every decision requires a reason and remains separate
            from deployment.
          </p>
        </div>
        <span className="approval-safety-pill">
          <ShieldCheck size={13} /> Human-in-the-loop
        </span>
      </section>

      {!canReview && (
        <section className="approval-readonly">
          <Eye size={17} />{" "}
          <span>
            You have read-only access to candidate evidence. A Model Reviewer or
            Administrator records approval decisions.
          </span>
        </section>
      )}
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section
        className="approval-count-grid"
        aria-label="Candidate review status"
      >
        <article>
          <span className="approval-count-icon purple">
            <ClipboardCheck size={17} />
          </span>
          <small>Awaiting review</small>
          <strong>{pendingCount}</strong>
          <p>Candidate decisions required</p>
        </article>
        <article>
          <span className="approval-count-icon teal">
            <ThumbsUp size={17} />
          </span>
          <small>Approved</small>
          <strong>{approvedCount}</strong>
          <p>Ready for deployment review</p>
        </article>
        <article>
          <span className="approval-count-icon coral">
            <ThumbsDown size={17} />
          </span>
          <small>Rejected</small>
          <strong>{rejectedCount}</strong>
          <p>Revision feedback recorded</p>
        </article>
      </section>

      <section className="approval-section-heading">
        <div>
          <span className="eyebrow">CANDIDATE QUEUE</span>
          <h2>Review a submitted candidate</h2>
        </div>
        <span>
          {proposals.length} total candidate{proposals.length === 1 ? "" : "s"}
        </span>
      </section>
      {loading && (
        <div className="approval-loading">
          <LoaderCircle className="spin" size={17} /> Reading candidate review
          records…
        </div>
      )}
      {!loading && proposals.length > 0 && (
        <div
          className="approval-proposal-list"
          aria-label="Submitted candidates"
        >
          {proposals.map((proposal) => (
            <button
              key={proposal.proposal_id}
              className={`approval-proposal-card ${selected?.proposal_id === proposal.proposal_id ? "selected" : ""}`}
              onClick={() => setSelected(proposal)}
            >
              <span
                className={`approval-proposal-icon status-${proposal.status}`}
              >
                <StatusIcon status={proposal.status} />
              </span>
              <span className="approval-proposal-copy">
                <strong>{modelName(proposal.metadata.model_name)}</strong>
                <small>
                  {proposal.metadata.run_id?.slice(0, 10) ||
                    proposal.proposal_id.slice(0, 10)}{" "}
                  · {formatDate(proposal.created_at)}
                </small>
              </span>
              <span className={`status-badge status-${proposal.status}`}>
                {statusLabel(proposal.status)}
              </span>
              <ChevronRight size={16} />
            </button>
          ))}
        </div>
      )}
      {!loading && !proposals.length && (
        <section className="panel approval-placeholder">
          <span>
            <ClipboardCheck size={25} />
          </span>
          <h2>No candidate is waiting for review.</h2>
          <p>
            A workflow that meets its validation and final-holdout quality gate
            will submit a candidate here.
          </p>
          <a className="text-button" href="#execution">
            Open workflow execution <ArrowRight size={14} />
          </a>
        </section>
      )}

      {selected && pendingSelected && canReview && (
        <section className="approval-actions">
          <div>
            <span className="eyebrow">REVIEW ACTION</span>
            <h3>
              Record your decision for {modelName(selected.metadata.model_name)}
              .
            </h3>
            <p>Your reason becomes part of the permanent local audit trail.</p>
          </div>
          <div>
            <button
              className="secondary-button approval-reject-button"
              onClick={() => setDialogDecision("reject")}
              disabled={submitting}
            >
              <ThumbsDown size={15} /> Reject with feedback
            </button>
            <button
              className="primary-button approval-approve-button"
              onClick={() => setDialogDecision("approve")}
              disabled={submitting}
            >
              <ThumbsUp size={15} /> Approve for deployment
            </button>
          </div>
        </section>
      )}
      {selected && <CandidateDetails proposal={selected} />}
      {dialogDecision && selected && (
        <ReviewDialog
          proposal={selected}
          decision={dialogDecision}
          submitting={submitting}
          onCancel={() => setDialogDecision(null)}
          onSubmit={(reason) => void submitReview(reason)}
        />
      )}
    </div>
  );
}
