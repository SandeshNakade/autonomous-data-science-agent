import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  Bell,
  BookOpen,
  Box,
  BrainCircuit,
  Check,
  CheckCheck,
  ChevronRight,
  CircleHelp,
  ClipboardCheck,
  Database,
  Eye,
  EyeOff,
  FlaskConical,
  GitBranch,
  History,
  LayoutDashboard,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Menu,
  RefreshCw,
  Rocket,
  Settings2,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Workflow,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { api, ApiError } from "./api";
import type { Overview, User } from "./api";
import { ComparisonPage } from "./ComparisonPage";
import { DatasetsPage } from "./DatasetsPage";
import { ExecutionPage } from "./ExecutionPage";
import { PlanningPage } from "./PlanningPage";
import { ApprovalPage } from "./ApprovalPage";
import { DeploymentPage } from "./DeploymentPage";
import { MonitoringPage } from "./MonitoringPage";
import { ExperimentHistoryPage } from "./ExperimentHistoryPage";
import { ModelRegistryPage } from "./ModelRegistryPage";
import { AlertsPage } from "./AlertsPage";
import { AuditLogsPage } from "./AuditLogsPage";
import { SettingsPage } from "./SettingsPage";

const lifecycle: { title: string; icon: LucideIcon; description: string }[] = [
  {
    title: "Dataset Management",
    icon: Database,
    description: "Upload a CSV, inspect its columns, and check data quality.",
  },
  {
    title: "AI Agent Planning",
    icon: BrainCircuit,
    description:
      "Review the target, task, preprocessing, metrics, and model choices.",
  },
  {
    title: "Workflow Execution",
    icon: Workflow,
    description:
      "Follow bounded training, evaluation, and improvement experiments.",
  },
  {
    title: "Model Comparison",
    icon: FlaskConical,
    description: "Compare metrics and execution time, then submit a candidate.",
  },
  {
    title: "Approval Center",
    icon: ClipboardCheck,
    description:
      "A reviewer approves or rejects the candidate with a recorded reason.",
  },
  {
    title: "Model Deployment",
    icon: Rocket,
    description:
      "An administrator deploys an approved version and tests predictions.",
  },
  {
    title: "Monitoring & Drift",
    icon: Activity,
    description:
      "Investigate alerts and request controlled retraining when needed.",
  },
];
function Brand({ light = false }: { light?: boolean }) {
  return (
    <div className={`brand ${light ? "brand-light" : ""}`}>
      <span className="brand-symbol" aria-hidden="true">
        <GitBranch size={23} strokeWidth={2.5} />
      </span>
      <span>
        Auto<span className="brand-ds">DS</span>
        <small>Autonomous Data Science</small>
      </span>
    </div>
  );
}

function FlowIllustration({ compact = false }: { compact?: boolean }) {
  return (
    <div
      className={`flow-illustration ${compact ? "compact" : ""}`}
      aria-label="Data to agent planning to model, with human approval"
    >
      <div className="flow-label">
        <span className="live-dot" /> A CONTROLLED JOURNEY
      </div>
      <div className="flow-nodes">
        <div className="flow-node">
          <Database />
          <span>Your data</span>
          <small>CSV dataset</small>
        </div>
        <div className="flow-connector">
          <ArrowRight size={16} />
        </div>
        <div className="flow-node agent-node">
          <BrainCircuit />
          <span>AI agent</span>
          <small>Plan · train · improve</small>
        </div>
        <div className="flow-connector">
          <ArrowRight size={16} />
        </div>
        <div className="flow-node">
          <Box />
          <span>Best model</span>
          <small>Ready for review</small>
        </div>
      </div>
      <div className="flow-guardrail">
        <ShieldCheck size={15} />
        <span>AI makes recommendations. You stay in control.</span>
      </div>
    </div>
  );
}

function ErrorNotice({ children }: { children: ReactNode }) {
  return (
    <div className="error-notice" role="alert">
      <TriangleAlert size={18} />
      <span>{children}</span>
    </div>
  );
}

function Login({ onLogin }: { onLogin: (user: User) => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [help, setHelp] = useState(false);
  useEffect(() => {
    api<{ configured: boolean }>("/auth/status")
      .then((data) => setConfigured(data.configured))
      .catch(() =>
        setError(
          "Cannot reach the workspace server. Check that FastAPI is running.",
        ),
      );
  }, []);
  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      onLogin(
        await api<User>("/auth/login", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        }),
      );
      setPassword("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to sign in.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="login-layout">
      <section className="welcome-panel">
        <Brand light />
        <div className="welcome-content">
          <span className="eyebrow light-eyebrow">
            <Sparkles size={14} /> INTELLIGENCE, WITH INTENTION
          </span>
          <h1>
            From raw data.
            <br />
            To real <span>decisions.</span>
          </h1>
          <p>
            Your complete machine learning journey, guided by an AI agent. Built
            for experimentation. Grounded in human oversight.
          </p>
          <FlowIllustration />
          <div className="welcome-benefits">
            <span>
              <CheckCheck size={17} /> Bounded experiments
            </span>
            <span>
              <ShieldCheck size={17} /> Human-approved models
            </span>
          </div>
        </div>
        <div className="welcome-footer">
          <span>Autonomous Data Science Agent</span>
          <span>Academic research prototype</span>
        </div>
      </section>
      <section className="login-panel">
        <div className="login-top">
          <span>YOUR RESEARCH WORKSPACE</span>
          <span className="version-pill">Stage 13 · Settings &amp; Users</span>
        </div>
        <div className="login-form-wrap">
          <div className="login-icon">
            <LockKeyhole size={25} />
          </div>
          <h2>Welcome back.</h2>
          <p className="login-subtitle">
            Sign in to move your next idea forward.
          </p>
          {configured === false && (
            <div className="setup-notice">
              <ShieldCheck size={18} />
              <div>
                <strong>Let’s set up your workspace.</strong>
                <p>
                  Create your first account using the local setup command below.
                  There are no default passwords.
                </p>
              </div>
            </div>
          )}
          <form onSubmit={submit}>
            <label htmlFor="email">Email address</label>
            <input
              id="email"
              type="email"
              autoComplete="username"
              placeholder="you@example.com"
              required
              maxLength={254}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={busy}
            />
            <div className="password-label">
              <label htmlFor="password">Password</label>
              <button
                type="button"
                className="text-button"
                onClick={() => setHelp(!help)}
              >
                Need access?
              </button>
            </div>
            <div className="password-input">
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                placeholder="Enter your password"
                required
                maxLength={128}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={busy}
              />
              <button
                type="button"
                aria-label={showPassword ? "Hide password" : "Show password"}
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            {error && <ErrorNotice>{error}</ErrorNotice>}
            <button
              type="submit"
              className="primary-button login-submit"
              disabled={busy || configured !== true}
            >
              {busy ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <>
                  Sign in to workspace <ArrowRight size={18} />
                </>
              )}
            </button>
            <p className="session-note">
              <ShieldCheck size={13} /> Your role and permissions are verified
              by the server.
            </p>
          </form>
          {(help || configured === false) && (
            <div className="access-help">
              <strong>Account setup</strong>
              <p>
                Ask your workspace administrator for access. For the first
                account, run this in the project terminal and choose your
                password at the prompt:
              </p>
              <code>
                python -m autods.auth --email you@example.com --name "Your Name"
                --role admin
              </code>
              <button
                className="text-button"
                onClick={() => window.location.reload()}
              >
                I’ve created my account <RefreshCw size={13} />
              </button>
            </div>
          )}
          <div className="role-intro">
            <span>ONE WORKSPACE. THREE PERSPECTIVES.</span>
          </div>
          <div className="login-roles">
            <div>
              <span className="role-icon blue">
                <FlaskConical size={17} />
              </span>
              <div>
                <strong>Data Scientist</strong>
                <small>Explore, plan & experiment</small>
              </div>
            </div>
            <div>
              <span className="role-icon purple">
                <ClipboardCheck size={17} />
              </span>
              <div>
                <strong>Model Reviewer</strong>
                <small>Evaluate & approve</small>
              </div>
            </div>
            <div>
              <span className="role-icon teal">
                <Rocket size={17} />
              </span>
              <div>
                <strong>MLOps Engineer</strong>
                <small>Deploy, monitor & manage</small>
              </div>
            </div>
          </div>
        </div>
        <footer className="login-footer">
          <LockKeyhole size={13} /> Local accounts · Protected sessions ·
          Role-based access
        </footer>
      </section>
    </main>
  );
}

function Dialog({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      className="dialog"
      aria-labelledby={titleId}
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">AUTODS WORKSPACE</span>
          <h2 id={titleId}>{title}</h2>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close dialog"
        >
          <X size={21} />
        </button>
      </header>
      {children}
    </dialog>
  );
}

function LifecycleDialog({ onClose }: { onClose: () => void }) {
  return (
    <Dialog title="One connected ML lifecycle" onClose={onClose}>
      <p className="dialog-intro">
        Each stage has its own focused page. Sign-in, roles, the workspace
        overview, Dataset Management, AI Agent Planning, Workflow Execution,
        Model Comparison, the Approval Center, Model Deployment, Monitoring &
        Drift, Experiment History, Model Registry, Alerts &amp; Notifications,
        Audit Logs, and Settings &amp; Users are available. Every planned
        workspace and supporting page is now connected.
      </p>
      <ol className="lifecycle-list">
        {lifecycle.map((step, index) => (
          <li key={step.title}>
            <span className="step-number">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div>
              <strong>{step.title}</strong>
              <p>{step.description}</p>
            </div>
            <step.icon size={19} />
          </li>
        ))}
      </ol>
      <div className="info-note">
        <GitBranch size={20} />
        <p>
          <strong>A feedback loop, not a one-way pipeline.</strong>
          <br />
          Rejected candidates return to planning with reviewer feedback. Drift
          triggers investigation and controlled retraining—not automatic
          deployment.
        </p>
      </div>
    </Dialog>
  );
}

function PermissionsDialog({
  user,
  onClose,
}: {
  user: User;
  onClose: () => void;
}) {
  const actions = [
    ["Read workspace & run results", "workspace:read"],
    ["Upload datasets", "datasets:write"],
    ["Start bounded experiments", "runs:write"],
    ["Test model predictions", "predictions:read"],
    ["Review candidate models", "reviews:write"],
    ["Deploy reviewer-approved models", "deployments:write"],
    ["Run drift checks", "monitoring:write"],
    ["Request controlled retraining", "retraining:write"],
    ["Manage local users and roles", "users:write"],
    ["Legacy approval + deployment", "legacy-promotion:write"],
  ];
  return (
    <Dialog title="Your workspace access" onClose={onClose}>
      <div className="permission-person">
        <span className="avatar">{initials(user.name)}</span>
        <div>
          <strong>{user.name}</strong>
          <p>{user.role_label}</p>
        </div>
        <ShieldCheck size={22} />
      </div>
      <p className="dialog-intro">
        These are your current API permissions, verified on every request. The
        workflow pages are being introduced in stages.
      </p>
      <div className="permission-list">
        {actions.map(([label, permission]) => (
          <div key={permission}>
            <span>{label}</span>
            {user.permissions.includes(permission) ? (
              <span className="permission-yes">
                <Check size={14} /> Allowed
              </span>
            ) : (
              <span className="permission-no">
                <LockKeyhole size={13} /> Restricted
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="info-note">
        <ShieldCheck size={20} />
        <p>
          Reviewer approval records a decision with a reason; it does not deploy
          a model. Administrator deployment is a separate,
          release-note-controlled action. The legacy combined operation remains
          administrator-only and still requires an approval token.
        </p>
      </div>
    </Dialog>
  );
}

function initials(name: string) {
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((word) => word[0])
    .join("")
    .toUpperCase();
}
function formatDate(value: string | null) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Not recorded"
    : date.toLocaleDateString(undefined, {
        day: "numeric",
        month: "short",
        year: "numeric",
      });
}
function statusLabel(value: string | null) {
  const labels: Record<string, string> = {
    needs_review: "Needs review",
    failed: "Failed",
    stopped: "Stopped",
    completed: "Completed",
    running: "Running",
  };
  return (
    labels[value || ""] || (value ? value.replaceAll("_", " ") : "Not recorded")
  );
}

function Workspace({ user, onLogout }: { user: User; onLogout: () => void }) {
  const [page, setPage] = useState<
    | "overview"
    | "datasets"
    | "planning"
    | "execution"
    | "comparison"
    | "approval"
    | "deployment"
    | "monitoring"
    | "history"
    | "registry"
    | "alerts"
    | "audit"
    | "settings"
  >(() => {
    if (window.location.hash === "#settings") return "settings";
    if (window.location.hash === "#audit-logs") return "audit";
    if (window.location.hash === "#alerts") return "alerts";
    if (window.location.hash === "#model-registry") return "registry";
    if (window.location.hash === "#experiment-history") return "history";
    if (window.location.hash === "#monitoring") return "monitoring";
    if (window.location.hash === "#deployment") return "deployment";
    if (window.location.hash === "#approval") return "approval";
    if (window.location.hash === "#comparison") return "comparison";
    if (window.location.hash === "#execution") return "execution";
    if (window.location.hash === "#planning") return "planning";
    if (window.location.hash === "#datasets") return "datasets";
    return "overview";
  });
  useEffect(() => {
    const navigate = () => {
      if (window.location.hash === "#datasets") setPage("datasets");
      else if (window.location.hash === "#planning") setPage("planning");
      else if (window.location.hash === "#execution") setPage("execution");
      else if (window.location.hash === "#comparison") setPage("comparison");
      else if (window.location.hash === "#approval") setPage("approval");
      else if (window.location.hash === "#deployment") setPage("deployment");
      else if (window.location.hash === "#monitoring") setPage("monitoring");
      else if (window.location.hash === "#experiment-history")
        setPage("history");
      else if (window.location.hash === "#model-registry") setPage("registry");
      else if (window.location.hash === "#alerts") setPage("alerts");
      else if (window.location.hash === "#audit-logs") setPage("audit");
      else if (window.location.hash === "#settings") setPage("settings");
      else if (!window.location.hash || window.location.hash === "#overview")
        setPage("overview");
    };
    window.addEventListener("hashchange", navigate);
    return () => window.removeEventListener("hashchange", navigate);
  }, []);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [signingOut, setSigningOut] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [dialog, setDialog] = useState<"lifecycle" | "permissions" | null>(
    null,
  );
  const sidebarRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!mobileNav) return;
    sidebarRef.current
      ?.querySelector<HTMLButtonElement>(".mobile-close")
      ?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNav(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      document.querySelector<HTMLButtonElement>(".mobile-menu")?.focus();
    };
  }, [mobileNav]);
  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setOverview(await api<Overview>("/workspace/overview"));
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        onLogout();
        return;
      }
      setError(
        "We couldn’t load the latest workspace data. Please try refreshing.",
      );
    } finally {
      setLoading(false);
    }
  }, [onLogout]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  async function signOut() {
    setSigningOut(true);
    try {
      await api("/auth/logout", { method: "POST" });
      onLogout();
    } catch {
      setError("Sign-out failed. Please try again.");
      setSigningOut(false);
    }
  }
  const firstName = user.name.split(" ")[0];
  const roleTitles = {
    scientist: "Scientist workspace",
    reviewer: "Reviewer workspace",
    admin: "Operations workspace",
  };
  const stats = [
    {
      title: "Recorded workflows",
      value: overview?.runs_count,
      icon: Workflow,
      color: "blue",
      note: "Saved experiment runs",
    },
    {
      title: "Awaiting review",
      value: overview?.review_count,
      icon: ClipboardCheck,
      color: "purple",
      note: "Candidates pending approval",
    },
    {
      title: "Failed workflows",
      value: overview?.failed_count,
      icon: TriangleAlert,
      color: "amber",
      note: "Recorded runs needing attention",
    },
    {
      title: "Production model",
      value: overview
        ? overview.production_available
          ? "Registered"
          : "Not deployed"
        : undefined,
      icon: Box,
      color: "teal",
      note: "Registry status · not a health check",
    },
  ];
  return (
    <div className="workspace-layout">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      {mobileNav && (
        <button
          className="nav-overlay"
          aria-label="Close navigation"
          onClick={() => setMobileNav(false)}
        />
      )}
      <aside
        ref={sidebarRef}
        className={`sidebar ${mobileNav ? "mobile-open" : ""}`}
        aria-label="Main navigation"
      >
        <div className="sidebar-brand">
          <Brand />
          <button
            className="icon-button mobile-close"
            aria-label="Close navigation"
            onClick={() => setMobileNav(false)}
          >
            <X size={20} />
          </button>
        </div>
        <div className="project-switcher">
          <span className="project-icon">
            <FlaskConical size={18} />
          </span>
          <div>
            <strong>Research workspace</strong>
            <small>
              <span className="live-dot" /> Local environment
            </small>
          </div>
        </div>
        <nav>
          <span className="nav-heading">WORKSPACE</span>
          <a
            href="#overview"
            className={`nav-link ${page === "overview" ? "active" : ""}`}
            aria-current={page === "overview" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <LayoutDashboard size={18} />
            <span>Overview</span>
            {page === "overview" && <span className="active-dot" />}
          </a>
          <div className="nav-heading">
            <span>ML LIFECYCLE</span>
            <span className="nav-later">STEP BY STEP</span>
          </div>
          <a
            href="#datasets"
            className={`nav-link ${page === "datasets" ? "active" : ""}`}
            aria-current={page === "datasets" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Database size={17} />
            <span>Dataset Management</span>
            {page === "datasets" && <span className="active-dot" />}
          </a>
          <a
            href="#planning"
            className={`nav-link ${page === "planning" ? "active" : ""}`}
            aria-current={page === "planning" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <BrainCircuit size={17} />
            <span>AI Agent Planning</span>
            {page === "planning" && <span className="active-dot" />}
          </a>
          <a
            href="#execution"
            className={`nav-link ${page === "execution" ? "active" : ""}`}
            aria-current={page === "execution" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Workflow size={17} />
            <span>Workflow Execution</span>
            {page === "execution" && <span className="active-dot" />}
          </a>
          <a
            href="#comparison"
            className={`nav-link ${page === "comparison" ? "active" : ""}`}
            aria-current={page === "comparison" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <FlaskConical size={17} />
            <span>Model Comparison</span>
            {page === "comparison" && <span className="active-dot" />}
          </a>
          <a
            href="#approval"
            className={`nav-link ${page === "approval" ? "active" : ""}`}
            aria-current={page === "approval" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <ClipboardCheck size={17} />
            <span>Approval Center</span>
            {page === "approval" && <span className="active-dot" />}
          </a>
          <a
            href="#deployment"
            className={`nav-link ${page === "deployment" ? "active" : ""}`}
            aria-current={page === "deployment" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Rocket size={17} />
            <span>Model Deployment</span>
            {page === "deployment" && <span className="active-dot" />}
          </a>
          <a
            href="#monitoring"
            className={`nav-link ${page === "monitoring" ? "active" : ""}`}
            aria-current={page === "monitoring" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Activity size={17} />
            <span>Monitoring &amp; Drift</span>
            {page === "monitoring" && <span className="active-dot" />}
          </a>
          {lifecycle.slice(7).map((step) => (
            <button
              key={step.title}
              className="nav-link upcoming"
              disabled
              title="This page will be built in a later stage"
            >
              <step.icon size={17} />
              <span>{step.title}</span>
            </button>
          ))}
          <span className="nav-heading tools-heading">
            PROJECT TOOLS <span className="nav-later">UPCOMING</span>
          </span>
          <a
            href="#experiment-history"
            className={`nav-link ${page === "history" ? "active" : ""}`}
            aria-current={page === "history" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <History size={17} />
            <span>Experiment History</span>
            {page === "history" && <span className="active-dot" />}
          </a>
          <a
            href="#model-registry"
            className={`nav-link ${page === "registry" ? "active" : ""}`}
            aria-current={page === "registry" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Box size={17} />
            <span>Model Registry</span>
            {page === "registry" && <span className="active-dot" />}
          </a>
          <a
            href="#alerts"
            className={`nav-link ${page === "alerts" ? "active" : ""}`}
            aria-current={page === "alerts" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Bell size={17} />
            <span>Alerts &amp; Notifications</span>
            {page === "alerts" && <span className="active-dot" />}
          </a>
          <a
            href="#audit-logs"
            className={`nav-link ${page === "audit" ? "active" : ""}`}
            aria-current={page === "audit" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <BookOpen size={17} />
            <span>Audit Logs</span>
            {page === "audit" && <span className="active-dot" />}
          </a>
          <a
            href="#settings"
            className={`nav-link ${page === "settings" ? "active" : ""}`}
            aria-current={page === "settings" ? "page" : undefined}
            onClick={() => setMobileNav(false)}
          >
            <Settings2 size={17} />
            <span>Settings &amp; Users</span>
            {page === "settings" && <span className="active-dot" />}
          </a>
        </nav>
        <div className="sidebar-bottom">
          <button
            className="build-progress"
            onClick={() => setDialog("lifecycle")}
          >
            <span>
              <Sparkles size={15} /> Building step by step{" "}
              <ChevronRight size={15} />
            </span>
            <small>Stage 13 · Settings &amp; Users</small>
            <span className="progress-track">
              <span />
            </span>
          </button>
          <div className="sidebar-user">
            <span className="avatar">{initials(user.name)}</span>
            <div>
              <strong>{user.name}</strong>
              <small>
                {user.role === "admin"
                  ? "Administrator"
                  : user.role === "reviewer"
                    ? "Model Reviewer"
                    : "Data Scientist"}
              </small>
            </div>
            <button
              className="icon-button"
              aria-label="Sign out"
              onClick={signOut}
              disabled={signingOut}
            >
              {signingOut ? (
                <LoaderCircle className="spin" size={17} />
              ) : (
                <LogOut size={17} />
              )}
            </button>
          </div>
        </div>
      </aside>
      <div className="workspace-body" inert={mobileNav}>
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMobileNav(true)}
            >
              <Menu size={21} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={14} />
            <strong>
              {page === "datasets"
                ? "Datasets"
                : page === "planning"
                  ? "AI Agent Planning"
                  : page === "execution"
                    ? "Workflow Execution"
                    : page === "comparison"
                      ? "Model Comparison"
                      : page === "approval"
                        ? "Approval Center"
                        : page === "deployment"
                          ? "Model Deployment"
                          : page === "monitoring"
                            ? "Monitoring & Drift"
                            : page === "history"
                              ? "Experiment History"
                              : page === "registry"
                                ? "Model Registry"
                                : page === "alerts"
                                  ? "Alerts & Notifications"
                                  : page === "audit"
                                    ? "Audit Logs"
                                    : page === "settings"
                                      ? "Settings & Users"
                                      : "Overview"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="environment-tag">
              <span className="live-dot" /> Local workspace
            </span>
            <span className="topbar-divider" />
            <button
              className="access-button"
              onClick={() => setDialog("permissions")}
            >
              <ShieldCheck size={17} />
              <span>My access</span>
            </button>
            <span className="avatar small">{initials(user.name)}</span>
          </div>
        </header>
        <main id="main-content" className="main-content" tabIndex={-1}>
          {page === "datasets" ? (
            <DatasetsPage user={user} onSessionExpired={onLogout} />
          ) : page === "planning" ? (
            <PlanningPage user={user} onSessionExpired={onLogout} />
          ) : page === "execution" ? (
            <ExecutionPage user={user} onSessionExpired={onLogout} />
          ) : page === "comparison" ? (
            <ComparisonPage onSessionExpired={onLogout} />
          ) : page === "approval" ? (
            <ApprovalPage user={user} onSessionExpired={onLogout} />
          ) : page === "deployment" ? (
            <DeploymentPage user={user} onSessionExpired={onLogout} />
          ) : page === "monitoring" ? (
            <MonitoringPage user={user} onSessionExpired={onLogout} />
          ) : page === "history" ? (
            <ExperimentHistoryPage onSessionExpired={onLogout} />
          ) : page === "registry" ? (
            <ModelRegistryPage onSessionExpired={onLogout} />
          ) : page === "alerts" ? (
            <AlertsPage onSessionExpired={onLogout} />
          ) : page === "audit" ? (
            <AuditLogsPage onSessionExpired={onLogout} />
          ) : page === "settings" ? (
            <SettingsPage user={user} onSessionExpired={onLogout} />
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">{roleTitles[user.role]}</div>
                  <h1>
                    Welcome back, {firstName}
                    <span className="greeting-dot">.</span>
                  </h1>
                  <p>A clear view of your data science journey.</p>
                </div>
                <button
                  className="secondary-button"
                  onClick={refresh}
                  disabled={loading}
                >
                  <RefreshCw size={15} className={loading ? "spin" : ""} />
                  {loading ? "Refreshing" : "Refresh data"}
                </button>
              </div>
              <section className="workspace-hero">
                <div className="hero-copy">
                  <span className="hero-tag">
                    <Sparkles size={13} /> YOUR DATA. YOUR DECISIONS.
                  </span>
                  <h2>
                    One workspace.
                    <br />
                    Every step of the ML lifecycle.
                  </h2>
                  <p>
                    From the first dataset to a model you can trust.
                    <br />
                    Let the agent explore. Keep the decisions in your hands.
                  </p>
                  <button onClick={() => setDialog("lifecycle")}>
                    Explore the lifecycle <ArrowRight size={16} />
                  </button>
                </div>
                <FlowIllustration compact />
              </section>
              <div className="section-heading">
                <h2>Workspace snapshot</h2>
                <span>
                  <span className="live-dot" />{" "}
                  {loading
                    ? "Loading saved records"
                    : error
                      ? "Refresh needed"
                      : "From your local artifacts"}
                </span>
              </div>
              {error && <ErrorNotice>{error}</ErrorNotice>}
              <section
                className="stats-grid"
                aria-label="Workspace statistics"
                aria-busy={loading}
              >
                {stats.map((stat) => (
                  <article className="stat-card" key={stat.title}>
                    <div className="stat-heading">
                      <span>{stat.title}</span>
                      <span className={`stat-icon ${stat.color}`}>
                        <stat.icon size={18} />
                      </span>
                    </div>
                    <strong
                      className={
                        typeof stat.value === "string" ? "stat-text" : ""
                      }
                    >
                      {stat.value ?? "—"}
                    </strong>
                    <p>{stat.note}</p>
                  </article>
                ))}
              </section>
              <div className="overview-grid">
                <section className="panel recent-panel">
                  <header className="panel-header">
                    <div>
                      <h2>Recent workflows</h2>
                      <p>Your most recent saved runs, in one place.</p>
                    </div>
                    <span className="neutral-badge">Latest 5</span>
                  </header>
                  {!overview ? (
                    <div className="empty-state">
                      <Database size={28} />
                      <h3>
                        {loading
                          ? "Loading your workspace…"
                          : "Workspace data unavailable"}
                      </h3>
                      <p>
                        {loading
                          ? "Reading saved experiment records."
                          : "Use Refresh data to try again."}
                      </p>
                    </div>
                  ) : overview.recent_runs.length === 0 ? (
                    <div className="empty-state">
                      <span className="empty-icon">
                        <FlaskConical size={25} />
                      </span>
                      <h3>Your next discovery starts here.</h3>
                      <p>
                        No saved workflows yet. Runs from the existing CLI or
                        API will appear here automatically.
                      </p>
                      <a className="text-button" href="#datasets">
                        Explore your dataset library <ArrowRight size={14} />
                      </a>
                    </div>
                  ) : (
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>OBJECTIVE / RUN</th>
                            <th>CREATED</th>
                            <th>STATUS</th>
                          </tr>
                        </thead>
                        <tbody>
                          {overview.recent_runs.map((run) => (
                            <tr key={run.run_id}>
                              <td>
                                <strong>
                                  {run.objective || "Untitled workflow"}
                                </strong>
                                <small>{run.run_id?.slice(0, 12)}</small>
                              </td>
                              <td className="date-cell">
                                {formatDate(run.started_at)}
                              </td>
                              <td>
                                <span
                                  className={`status-badge status-${run.status || "unknown"}`}
                                >
                                  <span />
                                  {statusLabel(run.status)}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <footer className="panel-footer">
                    <History size={14} />
                    <span>
                      Detailed history and model comparison have dedicated
                      pages.
                    </span>
                  </footer>
                </section>
                <section className="panel role-panel">
                  <div className="role-panel-icon">
                    <ShieldCheck size={24} />
                  </div>
                  <span className="eyebrow">HUMAN IN THE LOOP</span>
                  <h2>Your role matters.</h2>
                  <p>
                    {user.role === "scientist"
                      ? "Frame the problem, guide experiments, and build the evidence for a better model."
                      : user.role === "reviewer"
                        ? "Bring a critical eye to model quality. Every candidate deserves a thoughtful review."
                        : "Keep the journey from model to production controlled, observable, and accountable."}
                  </p>
                  <div className="role-summary">
                    <span className="avatar small">{initials(user.name)}</span>
                    <div>
                      <strong>
                        {user.role === "admin"
                          ? "MLOps & Administration"
                          : user.role === "reviewer"
                            ? "Model Review"
                            : "Data Science"}
                      </strong>
                      <small>Server-verified permissions</small>
                    </div>
                    <Check size={15} />
                  </div>
                  <button
                    className="text-button"
                    onClick={() => setDialog("permissions")}
                  >
                    View my permissions <ArrowRight size={15} />
                  </button>
                </section>
              </div>
              <section className="next-stage">
                <span className="next-stage-icon">
                  <Rocket size={21} />
                </span>
                <div>
                  <span className="eyebrow">NOW AVAILABLE · STAGE 07</span>
                  <h3>Deploy approved evidence, deliberately.</h3>
                  <p>
                    An MLOps Administrator can release a reviewer-approved
                    candidate with a recorded note, then test the protected
                    local prediction endpoint.
                  </p>
                </div>
                <a
                  href="#deployment"
                  className="text-button overview-dataset-link"
                >
                  Open deployment <ArrowRight size={15} />
                </a>
              </section>
            </>
          )}
          <footer className="workspace-footer">
            <span>
              AutoDS <span> / </span> Academic research prototype
            </span>
            <button onClick={() => setDialog("lifecycle")}>
              <CircleHelp size={14} /> About this workspace
            </button>
          </footer>
        </main>
      </div>
      {dialog === "lifecycle" && (
        <LifecycleDialog onClose={() => setDialog(null)} />
      )}
      {dialog === "permissions" && (
        <PermissionsDialog user={user} onClose={() => setDialog(null)} />
      )}
    </div>
  );
}

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);
  const logout = useCallback(() => setUser(null), []);
  useEffect(() => {
    api<User>("/auth/me")
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setChecking(false));
  }, []);
  if (checking)
    return (
      <div className="boot-screen">
        <Brand />
        <LoaderCircle size={24} className="spin" />
        <p>Opening your workspace…</p>
      </div>
    );
  return user ? (
    <Workspace user={user} onLogout={logout} />
  ) : (
    <Login onLogin={setUser} />
  );
}
