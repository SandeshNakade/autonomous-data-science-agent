import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  BookOpenCheck,
  CircleAlert,
  Clock3,
  Eye,
  FileCheck2,
  FlaskConical,
  LoaderCircle,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  UserRound,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { AuditLogItem, AuditLogResponse } from "./api";
import "./audit-logs.css";

function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function actionLabel(action: string) {
  const labels: Record<string, string> = {
    submitted_for_review: "Submitted for review",
    approved_for_deployment: "Approved for deployment",
    rejected: "Rejected",
    deployed: "Deployed",
    legacy_promoted: "Legacy promotion",
    feature_drift_checked: "Feature drift checked",
    workflow_completed: "Workflow completed",
    workflow_failed: "Workflow failed",
    workflow_needs_review: "Workflow awaiting review",
    workflow_stopped: "Workflow stopped",
  };
  return labels[action] || action.replaceAll("_", " ");
}

function sourceLabel(source: string) {
  const labels: Record<string, string> = {
    governance: "Model governance",
    monitoring: "Monitoring",
    workflow: "Workflow",
    workspace: "Workspace administration",
  };
  return labels[source] || source.replaceAll("_", " ");
}

function resourceLabel(type: string) {
  const labels: Record<string, string> = {
    model_candidate: "Model candidate",
    monitoring_report: "Monitoring report",
    workflow_run: "Workflow run",
    workspace_user: "Workspace user",
  };
  return labels[type] || type.replaceAll("_", " ");
}

function destinationLabel(destination: string) {
  const labels: Record<string, string> = {
    "#model-registry": "Open model registry",
    "#monitoring": "Open monitoring",
    "#experiment-history": "Open experiment history",
    "#settings": "Open settings & users",
  };
  return labels[destination] || "Open related workspace";
}

function SourceIcon({ source, size = 16 }: { source: string; size?: number }) {
  if (source === "monitoring") return <Activity size={size} />;
  if (source === "workflow") return <FlaskConical size={size} />;
  if (source === "workspace") return <Settings2 size={size} />;
  return <ShieldCheck size={size} />;
}

function SourceBadge({ source }: { source: string }) {
  return (
    <span className={`audit-source-badge ${source}`}>
      <SourceIcon source={source} size={12} /> {sourceLabel(source)}
    </span>
  );
}

export function AuditLogsPage({
  onSessionExpired,
}: {
  onSessionExpired: () => void;
}) {
  const [events, setEvents] = useState<AuditLogItem[]>([]);
  const [total, setTotal] = useState(0);
  const [selectedId, setSelectedId] = useState("");
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("all");
  const [action, setAction] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api<AuditLogResponse>("/audit-logs?limit=250");
      setEvents(response.items);
      setTotal(response.total);
      setSelectedId((current) =>
        response.items.some((event) => event.event_id === current)
          ? current
          : response.items[0]?.event_id || "",
      );
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) {
        onSessionExpired();
        return;
      }
      setError(
        cause instanceof Error
          ? cause.message
          : "Audit records could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }, [onSessionExpired]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const actions = useMemo(
    () => [...new Set(events.map((event) => event.action))].sort(),
    [events],
  );
  const visible = useMemo(() => {
    const search = query.trim().toLowerCase();
    return events.filter((event) => {
      const matchesSource = source === "all" || event.source === source;
      const matchesAction = action === "all" || event.action === action;
      const matchesSearch =
        !search ||
        `${event.summary} ${event.reason || ""} ${event.actor || ""} ${event.resource_id}`
          .toLowerCase()
          .includes(search);
      return matchesSource && matchesAction && matchesSearch;
    });
  }, [action, events, query, source]);
  const selected =
    visible.find((event) => event.event_id === selectedId) ||
    visible[0] ||
    null;
  const governanceCount = events.filter(
    (event) => event.source === "governance",
  ).length;
  const monitoringCount = events.filter(
    (event) => event.source === "monitoring",
  ).length;
  const workflowCount = events.filter(
    (event) => event.source === "workflow",
  ).length;

  return (
    <div className="audit-logs-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">SUPPORTING PAGE · AUDIT LOGS</div>
          <h1>Understand what changed, when, and from which record.</h1>
          <p>
            Review a safe, searchable audit view of saved governance,
            monitoring, and workflow evidence.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          records
        </button>
      </div>
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section className="audit-hero">
        <span className="audit-hero-icon">
          <BookOpenCheck size={24} />
        </span>
        <div>
          <span className="eyebrow">READ-ONLY EVIDENCE TRAIL</span>
          <h2>Every row starts with a saved operational record.</h2>
          <p>
            This prototype combines retained model-governance events,
            feature-drift checks, and workflow status records. It does not
            create, edit, or silently delete audit entries.
          </p>
        </div>
        <span className="audit-hero-state">
          <span /> {total} recorded event{total === 1 ? "" : "s"}
        </span>
      </section>

      <section className="audit-status-grid" aria-label="Audit event overview">
        <article>
          <span className="audit-stat-icon blue">
            <BookOpenCheck size={18} />
          </span>
          <small>Visible records</small>
          <strong>{events.length}</strong>
          <p>Up to the latest 250 events</p>
        </article>
        <article>
          <span className="audit-stat-icon purple">
            <ShieldCheck size={18} />
          </span>
          <small>Governance events</small>
          <strong>{governanceCount}</strong>
          <p>Review and release decisions</p>
        </article>
        <article>
          <span className="audit-stat-icon teal">
            <Activity size={18} />
          </span>
          <small>Monitoring checks</small>
          <strong>{monitoringCount}</strong>
          <p>Saved feature-drift reports</p>
        </article>
        <article>
          <span className="audit-stat-icon amber">
            <FlaskConical size={18} />
          </span>
          <small>Workflow records</small>
          <strong>{workflowCount}</strong>
          <p>Persisted run outcomes</p>
        </article>
      </section>

      <section className="panel audit-library-panel">
        <header className="panel-header">
          <div>
            <span className="eyebrow">AUDIT RECORDS</span>
            <h2>Search the evidence trail</h2>
            <p>
              Filter by source or event type, then open the related protected
              workspace to inspect its underlying evidence.
            </p>
          </div>
          <SlidersHorizontal size={20} />
        </header>
        <div className="audit-filters">
          <label className="audit-search">
            <Search size={16} />
            <span className="sr-only">Search audit records</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search action, actor, reason, or resource ID"
            />
          </label>
          <label>
            <span>Source</span>
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
            >
              <option value="all">All sources</option>
              <option value="governance">Model governance</option>
              <option value="monitoring">Monitoring</option>
              <option value="workflow">Workflow</option>
              <option value="workspace">Workspace administration</option>
            </select>
          </label>
          <label>
            <span>Action</span>
            <select
              value={action}
              onChange={(event) => setAction(event.target.value)}
            >
              <option value="all">All actions</option>
              {actions.map((value) => (
                <option key={value} value={value}>
                  {actionLabel(value)}
                </option>
              ))}
            </select>
          </label>
          <span>
            {visible.length} of {events.length} shown
          </span>
        </div>
        {loading ? (
          <div className="audit-empty-inline">
            <LoaderCircle className="spin" size={18} /> Reading saved audit
            evidence…
          </div>
        ) : !visible.length ? (
          <div className="audit-empty-inline">
            <FileCheck2 size={18} /> No audit record matches these filters.
          </div>
        ) : (
          <div className="table-scroll audit-table">
            <table>
              <thead>
                <tr>
                  <th>WHEN</th>
                  <th>ACTION</th>
                  <th>ACTOR</th>
                  <th>SOURCE</th>
                  <th>RESOURCE</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {visible.map((event) => (
                  <tr
                    key={event.event_id}
                    className={
                      selected?.event_id === event.event_id
                        ? "audit-selected-row"
                        : ""
                    }
                  >
                    <td>
                      <span className="audit-date">
                        <Clock3 size={13} /> {formatDate(event.occurred_at)}
                      </span>
                    </td>
                    <td>
                      <strong>{actionLabel(event.action)}</strong>
                      <small>{event.summary}</small>
                    </td>
                    <td>
                      <span className="audit-actor">
                        <UserRound size={13} /> {event.actor || "Not recorded"}
                      </span>
                    </td>
                    <td>
                      <SourceBadge source={event.source} />
                    </td>
                    <td>
                      <span className="audit-resource">
                        {resourceLabel(event.resource_type)}
                        <small>{event.resource_id.slice(0, 12)}</small>
                      </span>
                    </td>
                    <td>
                      <button
                        className="audit-open"
                        onClick={() => setSelectedId(event.event_id)}
                        aria-label={`View audit event: ${actionLabel(event.action)}`}
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

      <section className="audit-detail-heading">
        <div>
          <span className="eyebrow">SELECTED RECORD</span>
          <h2>Action context</h2>
        </div>
        {selected && <SourceBadge source={selected.source} />}
      </section>
      {selected ? (
        <section className="audit-detail-grid">
          <article className="panel audit-detail-card">
            <header className="panel-header">
              <div>
                <span className="eyebrow">{sourceLabel(selected.source)}</span>
                <h2>{actionLabel(selected.action)}</h2>
                <p>{formatDate(selected.occurred_at)}</p>
              </div>
              <SourceIcon source={selected.source} size={21} />
            </header>
            <div className="audit-detail-summary">
              <strong>{selected.summary}</strong>
              <p>
                {selected.reason ||
                  "No additional reason was retained for this record."}
              </p>
            </div>
            <div className="audit-detail-facts">
              <span>
                <small>ACTOR</small>
                <strong>{selected.actor || "Not recorded"}</strong>
              </span>
              <span>
                <small>RESOURCE TYPE</small>
                <strong>{resourceLabel(selected.resource_type)}</strong>
              </span>
              <span>
                <small>RESOURCE ID</small>
                <strong>{selected.resource_id.slice(0, 16)}</strong>
              </span>
            </div>
            <a className="audit-destination" href={selected.destination}>
              <span>
                <Eye size={17} />
              </span>
              <div>
                <strong>{destinationLabel(selected.destination)}</strong>
                <p>
                  Inspect the retained evidence and take action only in its
                  protected page.
                </p>
              </div>
              <ArrowRight size={17} />
            </a>
          </article>
          <aside className="audit-boundary-card">
            <span className="audit-boundary-icon">
              <ShieldCheck size={19} />
            </span>
            <div>
              <span className="eyebrow">SCOPE BOUNDARY</span>
              <h3>Evidence, not a simulated ledger.</h3>
              <p>
                The view derives from already-persisted model, monitoring, and
                workflow records. It does not claim login history, data-download
                history, or a centralized immutable audit store.
              </p>
            </div>
            <ul>
              <li>
                <FileCheck2 size={14} /> No audit event can be edited here.
              </li>
              <li>
                <FileCheck2 size={14} /> Missing actors remain visibly
                unrecorded.
              </li>
              <li>
                <FileCheck2 size={14} /> No local paths or model bytes are
                exposed.
              </li>
            </ul>
          </aside>
        </section>
      ) : (
        <section className="audit-empty-detail">
          <BookOpenCheck size={20} /> Select a record to inspect its retained
          context.
        </section>
      )}
    </div>
  );
}
