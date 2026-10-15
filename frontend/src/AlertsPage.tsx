import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BellRing,
  CircleAlert,
  CircleCheck,
  Database,
  Eye,
  FlaskConical,
  Info,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { AlertItem } from "./api";
import "./alerts.css";

function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "Not recorded";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function sourceLabel(source: string) {
  const labels: Record<string, string> = {
    monitoring: "Monitoring",
    workflow: "Workflow execution",
    governance: "Model governance",
  };
  return labels[source] || source.replaceAll("_", " ");
}

function severityLabel(severity: string) {
  const labels: Record<string, string> = {
    critical: "Critical",
    warning: "Warning",
    info: "Info",
  };
  return labels[severity] || severity;
}

function SeverityIcon({
  severity,
  size = 17,
}: {
  severity: string;
  size?: number;
}) {
  if (severity === "critical") return <CircleAlert size={size} />;
  if (severity === "warning") return <AlertTriangle size={size} />;
  return <Info size={size} />;
}

function AlertBadge({ severity }: { severity: string }) {
  return (
    <span className={`alert-badge ${severity}`}>
      <SeverityIcon severity={severity} size={12} /> {severityLabel(severity)}
    </span>
  );
}

function destinationLabel(destination: string) {
  const labels: Record<string, string> = {
    "#monitoring": "Open monitoring",
    "#experiment-history": "Open experiment history",
    "#approval": "Open approval center",
  };
  return labels[destination] || "Open related workspace";
}

export function AlertsPage({
  onSessionExpired,
}: {
  onSessionExpired: () => void;
}) {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [severity, setSeverity] = useState("all");
  const [source, setSource] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const nextAlerts = await api<AlertItem[]>("/alerts");
      setAlerts(nextAlerts);
      setSelectedId((current) =>
        nextAlerts.some((alert) => alert.alert_id === current)
          ? current
          : nextAlerts[0]?.alert_id || "",
      );
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) {
        onSessionExpired();
        return;
      }
      setError(
        cause instanceof Error ? cause.message : "Alerts could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }, [onSessionExpired]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const visible = useMemo(() => {
    const query = search.trim().toLowerCase();
    return alerts.filter((alert) => {
      const matchesSeverity = severity === "all" || alert.severity === severity;
      const matchesSource = source === "all" || alert.source === source;
      const matchesSearch =
        !query ||
        `${alert.title} ${alert.message} ${alert.resource_id}`
          .toLowerCase()
          .includes(query);
      return matchesSeverity && matchesSource && matchesSearch;
    });
  }, [alerts, search, severity, source]);
  const selected =
    alerts.find((alert) => alert.alert_id === selectedId) || null;
  const critical = alerts.filter(
    (alert) => alert.severity === "critical",
  ).length;
  const warning = alerts.filter((alert) => alert.severity === "warning").length;
  const informational = alerts.filter(
    (alert) => alert.severity === "info",
  ).length;

  return (
    <div className="alerts-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            SUPPORTING PAGE · ALERTS &amp; NOTIFICATIONS
          </div>
          <h1>Focus attention where the evidence says it belongs.</h1>
          <p>
            Review unresolved monitoring, workflow, and governance events from
            the controlled system.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          alerts
        </button>
      </div>
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section className={`alerts-hero ${alerts.length ? "active" : "clear"}`}>
        <span className="alerts-hero-icon">
          <BellRing size={24} />
        </span>
        <div>
          <span className="eyebrow">IN-APP SIGNAL CENTER</span>
          <h2>
            {alerts.length
              ? `${alerts.length} unresolved system signal${alerts.length === 1 ? "" : "s"}.`
              : "No unresolved system signals."}
          </h2>
          <p>
            Alerts are derived from saved evidence only. This academic prototype
            does not send email, SMS, push notifications, or fabricate delivery
            status.
          </p>
        </div>
        <span
          className={`alerts-hero-state ${alerts.length ? "active" : "clear"}`}
        >
          <span /> {alerts.length ? "Review needed" : "All clear"}
        </span>
      </section>

      <section className="alerts-status-grid" aria-label="Alert overview">
        <article>
          <span className="alerts-stat-icon blue">
            <BellRing size={18} />
          </span>
          <small>Open signals</small>
          <strong>{alerts.length}</strong>
          <p>Read-only in-app inbox</p>
        </article>
        <article>
          <span className="alerts-stat-icon red">
            <CircleAlert size={18} />
          </span>
          <small>Critical</small>
          <strong>{critical}</strong>
          <p>Failed workflow events</p>
        </article>
        <article>
          <span className="alerts-stat-icon amber">
            <AlertTriangle size={18} />
          </span>
          <small>Warnings</small>
          <strong>{warning}</strong>
          <p>Drift investigation signals</p>
        </article>
        <article>
          <span className="alerts-stat-icon purple">
            <Info size={18} />
          </span>
          <small>Informational</small>
          <strong>{informational}</strong>
          <p>Human review reminders</p>
        </article>
      </section>

      <section className="alerts-layout">
        <article className="panel alerts-library-panel">
          <header className="panel-header">
            <div>
              <span className="eyebrow">OPEN EVENTS</span>
              <h2>Alert inbox</h2>
              <p>
                Events remain listed while their underlying saved evidence
                remains unresolved.
              </p>
            </div>
            <SlidersHorizontal size={20} />
          </header>
          <div className="alerts-filters">
            <label className="alerts-search">
              <Search size={16} />
              <span className="sr-only">Search alerts</span>
              <input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search event or resource ID"
              />
            </label>
            <label>
              <span>Severity</span>
              <select
                value={severity}
                onChange={(event) => setSeverity(event.target.value)}
              >
                <option value="all">All severities</option>
                <option value="critical">Critical</option>
                <option value="warning">Warning</option>
                <option value="info">Info</option>
              </select>
            </label>
            <label>
              <span>Source</span>
              <select
                value={source}
                onChange={(event) => setSource(event.target.value)}
              >
                <option value="all">All sources</option>
                <option value="monitoring">Monitoring</option>
                <option value="workflow">Workflow</option>
                <option value="governance">Governance</option>
              </select>
            </label>
          </div>
          {loading ? (
            <div className="alerts-empty">
              <LoaderCircle className="spin" size={18} /> Reading unresolved
              events…
            </div>
          ) : !visible.length ? (
            <div className="alerts-empty">
              <CircleCheck size={18} /> No open event matches these filters.
            </div>
          ) : (
            <div className="alerts-list">
              {visible.map((alert) => (
                <button
                  key={alert.alert_id}
                  className={`alert-row ${selectedId === alert.alert_id ? "selected" : ""} ${alert.severity}`}
                  onClick={() => setSelectedId(alert.alert_id)}
                >
                  <span className={`alert-row-icon ${alert.severity}`}>
                    <SeverityIcon severity={alert.severity} />
                  </span>
                  <span className="alert-row-copy">
                    <strong>{alert.title}</strong>
                    <small>{alert.message}</small>
                    <em>
                      {sourceLabel(alert.source)} ·{" "}
                      {formatDate(alert.occurred_at)}
                    </em>
                  </span>
                  <AlertBadge severity={alert.severity} />
                </button>
              ))}
            </div>
          )}
        </article>

        <article className="panel alerts-detail-panel">
          {selected ? (
            <>
              <header className="panel-header">
                <div>
                  <span className="eyebrow">SELECTED EVENT</span>
                  <h2>{selected.title}</h2>
                  <p>
                    {sourceLabel(selected.source)} ·{" "}
                    {formatDate(selected.occurred_at)}
                  </p>
                </div>
                <AlertBadge severity={selected.severity} />
              </header>
              <div className={`alerts-detail-outcome ${selected.severity}`}>
                <span>
                  <SeverityIcon severity={selected.severity} size={21} />
                </span>
                <div>
                  <strong>{selected.message}</strong>
                  <p>
                    This event is based on a persisted{" "}
                    {sourceLabel(selected.source).toLowerCase()} record. Review
                    the related workspace for the underlying evidence and next
                    safe action.
                  </p>
                </div>
              </div>
              <div className="alerts-detail-facts">
                <span>
                  <small>SOURCE</small>
                  <strong>{sourceLabel(selected.source)}</strong>
                </span>
                <span>
                  <small>STATE</small>
                  <strong>
                    {selected.state === "open" ? "Open" : selected.state}
                  </strong>
                </span>
                <span>
                  <small>RESOURCE</small>
                  <strong>{selected.resource_id.slice(0, 14)}</strong>
                </span>
              </div>
              <a className="alerts-destination" href={selected.destination}>
                <span>
                  <Eye size={17} />
                </span>
                <div>
                  <strong>{destinationLabel(selected.destination)}</strong>
                  <p>
                    Inspect the evidence and take action only through the
                    protected workflow page.
                  </p>
                </div>
                <ArrowRight size={17} />
              </a>
            </>
          ) : (
            <div className="alerts-empty">
              <BellRing size={20} /> Select an alert to inspect its source and
              evidence link.
            </div>
          )}
        </article>
      </section>

      <section className="alerts-scope-grid">
        <article>
          <span>
            <Database size={18} />
          </span>
          <div>
            <strong>Monitoring signals</strong>
            <p>
              Created when saved feature-drift evidence detects drift above its
              configured threshold.
            </p>
          </div>
        </article>
        <article>
          <span>
            <FlaskConical size={18} />
          </span>
          <div>
            <strong>Workflow signals</strong>
            <p>
              Created only for persisted controlled workflows whose recorded
              status is failed.
            </p>
          </div>
        </article>
        <article>
          <span>
            <ShieldCheck size={18} />
          </span>
          <div>
            <strong>Governance signals</strong>
            <p>
              Created for actual model candidates still waiting for a human
              reviewer decision.
            </p>
          </div>
        </article>
      </section>
      <section className="alerts-guardrail">
        <ShieldCheck size={17} />
        <div>
          <strong>
            Notifications are not delivery channels in this prototype.
          </strong>
          <p>
            The page is a live, read-only in-app inbox. Acknowledgement
            workflows, email, SMS, push notifications, and escalation rules are
            intentionally future work.
          </p>
        </div>
      </section>
    </div>
  );
}
