import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  CircleAlert,
  ClipboardCheck,
  Clock3,
  Code2,
  Database,
  FileKey2,
  LoaderCircle,
  LockKeyhole,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  UserCog,
  UserRoundCheck,
  UserRoundX,
  UsersRound,
  X,
} from "lucide-react";
import { api, ApiError } from "./api";
import type {
  Role,
  User,
  WorkspaceSettings,
  WorkspaceUser,
  WorkspaceUsersResponse,
} from "./api";
import "./settings.css";

const ROLE_NAMES: Record<Role, string> = {
  scientist: "Data Scientist",
  reviewer: "Model Reviewer",
  admin: "MLOps Administrator",
};

function humanize(value: string) {
  return value.replaceAll("_", " ");
}

function RoleIcon({ role, size = 17 }: { role: Role; size?: number }) {
  if (role === "reviewer") return <ClipboardCheck size={size} />;
  if (role === "admin") return <Settings2 size={size} />;
  return <Database size={size} />;
}

function RoleBadge({ role }: { role: Role }) {
  return (
    <span className={`workspace-role-badge ${role}`}>
      <RoleIcon role={role} size={12} /> {ROLE_NAMES[role]}
    </span>
  );
}

function UserChangeDialog({
  change,
  busy,
  onClose,
  onConfirm,
}: {
  change: { user: WorkspaceUser; role?: Role; active?: boolean };
  busy: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const roleChange = change.role !== undefined;
  const activeChange = change.active !== undefined;
  const title = roleChange
    ? `Change ${change.user.name}'s role?`
    : change.active
      ? `Reactivate ${change.user.name}'s account?`
      : `Deactivate ${change.user.name}'s account?`;
  useEffect(() => {
    ref.current?.showModal();
    return () => ref.current?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className="dialog settings-confirm-dialog"
      aria-labelledby="settings-confirm-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onClose();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">
            <UserCog size={13} /> ADMINISTRATOR ACTION
          </span>
          <h2 id="settings-confirm-title">{title}</h2>
        </div>
        <button
          className="icon-button"
          aria-label="Cancel account update"
          onClick={onClose}
          disabled={busy}
        >
          <X size={18} />
        </button>
      </header>
      <p className="dialog-intro">
        {roleChange
          ? `${change.user.name} will move from ${ROLE_NAMES[change.user.role]} to ${ROLE_NAMES[change.role!]}.`
          : change.active
            ? `${change.user.name} will be allowed to sign in again with their current role.`
            : `${change.user.name} will no longer be able to sign in until an administrator reactivates the account.`}
      </p>
      <div className="settings-confirm-guardrail">
        <ShieldCheck size={18} />
        <span>
          The user’s existing sessions will be revoked immediately. Passwords
          are never viewed, stored, or changed through this page.
        </span>
      </div>
      {activeChange && !change.active && (
        <div className="settings-confirm-warning">
          <CircleAlert size={17} /> This change is recorded in the workspace
          audit trail.
        </div>
      )}
      <footer className="settings-confirm-actions">
        <button className="secondary-button" onClick={onClose} disabled={busy}>
          Cancel
        </button>
        <button
          className="primary-button settings-confirm-button"
          onClick={onConfirm}
          disabled={busy}
        >
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <Check size={16} />
          )}
          Confirm change
        </button>
      </footer>
    </dialog>
  );
}

export function SettingsPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const [settings, setSettings] = useState<WorkspaceSettings | null>(null);
  const [users, setUsers] = useState<WorkspaceUser[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [change, setChange] = useState<{
    user: WorkspaceUser;
    role?: Role;
    active?: boolean;
  } | null>(null);
  const [updatingId, setUpdatingId] = useState("");
  const isAdmin = user.permissions.includes("users:write");

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
      const nextSettings = await api<WorkspaceSettings>("/auth/settings");
      setSettings(nextSettings);
      if (isAdmin) {
        const response = await api<WorkspaceUsersResponse>("/auth/users");
        setUsers(response.items);
      } else {
        setUsers([]);
      }
    } catch (cause) {
      handleError(cause, "Workspace settings could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, [handleError, isAdmin]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const visibleUsers = useMemo(() => {
    const search = query.trim().toLowerCase();
    if (!search) return users;
    return users.filter((workspaceUser) =>
      `${workspaceUser.name} ${workspaceUser.email} ${workspaceUser.role_label}`
        .toLowerCase()
        .includes(search),
    );
  }, [query, users]);
  const activeUsers = users.filter(
    (workspaceUser) => workspaceUser.active,
  ).length;
  const activeAdmins = users.filter(
    (workspaceUser) => workspaceUser.active && workspaceUser.role === "admin",
  ).length;

  async function applyChange() {
    if (!change) return;
    setUpdatingId(change.user.id);
    setError("");
    try {
      const result = await api<{ user: WorkspaceUser; changed: boolean }>(
        `/auth/users/${change.user.id}`,
        {
          method: "PATCH",
          body: JSON.stringify({ role: change.role, active: change.active }),
        },
      );
      setUsers((current) =>
        current.map((workspaceUser) =>
          workspaceUser.id === result.user.id ? result.user : workspaceUser,
        ),
      );
      setChange(null);
    } catch (cause) {
      handleError(cause, "The workspace account could not be updated.");
    } finally {
      setUpdatingId("");
    }
  }

  return (
    <div className="settings-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">SUPPORTING PAGE · SETTINGS &amp; USERS</div>
          <h1>Keep the workspace controlled, clear, and accountable.</h1>
          <p>
            Review the fixed local configuration and, for administrators, manage
            workspace roles and account access.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void refresh()}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
          settings
        </button>
      </div>
      {error && (
        <div className="error-notice">
          <CircleAlert size={16} /> {error}
        </div>
      )}

      <section className="settings-hero">
        <span className="settings-hero-icon">
          <Settings2 size={24} />
        </span>
        <div>
          <span className="eyebrow">LOCAL CONTROL PLANE</span>
          <h2>
            {isAdmin
              ? "You can manage access with server-enforced guardrails."
              : "Your access is set by a workspace administrator."}
          </h2>
          <p>
            Roles are verified by the server for every protected action. This
            page never exposes passwords, session tokens, API keys, or local
            secrets.
          </p>
        </div>
        <span className={`settings-hero-state ${isAdmin ? "admin" : "member"}`}>
          <span /> {isAdmin ? "Administrator access" : "Read-only settings"}
        </span>
      </section>

      <section
        className="settings-status-grid"
        aria-label="Workspace settings overview"
      >
        <article>
          <span className="settings-stat-icon blue">
            <LockKeyhole size={18} />
          </span>
          <small>Authentication</small>
          <strong>{settings ? humanize(settings.authentication) : "—"}</strong>
          <p>Protected browser sessions</p>
        </article>
        <article>
          <span className="settings-stat-icon purple">
            <Clock3 size={18} />
          </span>
          <small>Session window</small>
          <strong>{settings ? `${settings.session_hours} hours` : "—"}</strong>
          <p>Reauthentication required</p>
        </article>
        <article>
          <span className="settings-stat-icon teal">
            <UsersRound size={18} />
          </span>
          <small>Active accounts</small>
          <strong>{isAdmin ? activeUsers : "Protected"}</strong>
          <p>
            {isAdmin
              ? `${activeAdmins} active administrator${activeAdmins === 1 ? "" : "s"}`
              : "Visible to administrators"}
          </p>
        </article>
        <article>
          <span className="settings-stat-icon amber">
            <FileKey2 size={18} />
          </span>
          <small>Account provisioning</small>
          <strong>
            {settings ? humanize(settings.user_provisioning) : "—"}
          </strong>
          <p>Passwords stay off the web UI</p>
        </article>
      </section>

      <section className="settings-layout">
        <article className="panel workspace-config-card">
          <header className="panel-header">
            <div>
              <span className="eyebrow">WORKSPACE CONFIGURATION</span>
              <h2>Fixed prototype controls</h2>
              <p>
                These configuration values are intentionally visible but not
                weakened through the browser.
              </p>
            </div>
            <SlidersHorizontal size={20} />
          </header>
          {loading || !settings ? (
            <div className="settings-empty">
              <LoaderCircle className="spin" size={18} /> Reading configuration…
            </div>
          ) : (
            <>
              <div className="workspace-config-facts">
                <span>
                  <small>WORKSPACE MODE</small>
                  <strong>{humanize(settings.workspace_mode)}</strong>
                </span>
                <span>
                  <small>ACCOUNT CREATION</small>
                  <strong>{humanize(settings.user_provisioning)}</strong>
                </span>
                <span>
                  <small>PASSWORD CHANGES</small>
                  <strong>{humanize(settings.password_management)}</strong>
                </span>
              </div>
              <div className="workspace-guardrails">
                {settings.guardrails.map((guardrail) => (
                  <p key={guardrail}>
                    <ShieldCheck size={15} /> {guardrail}
                  </p>
                ))}
              </div>
            </>
          )}
        </article>

        <aside className="settings-provision-card">
          <span className="settings-provision-icon">
            <Code2 size={20} />
          </span>
          <div>
            <span className="eyebrow">SAFE ACCOUNT PROVISIONING</span>
            <h3>Create accounts locally.</h3>
            <p>
              New accounts and passwords are created only from the project
              terminal, never submitted through this web form.
            </p>
          </div>
          <code>
            python -m autods.auth --email you@example.com --name "Your Name"
            --role reviewer
          </code>
          <a className="text-button" href="#audit-logs">
            Open audit logs <ArrowRight size={14} />
          </a>
        </aside>
      </section>

      <section className="settings-roles-heading">
        <div>
          <span className="eyebrow">ROLE POLICY</span>
          <h2>Three focused workspace roles</h2>
        </div>
        <span>Permissions are enforced server-side.</span>
      </section>
      <section className="settings-role-grid">
        {(settings?.roles || []).map((role) => (
          <article
            key={role.role}
            className={`settings-role-card ${role.role}`}
          >
            <span className="settings-role-icon">
              <RoleIcon role={role.role} size={19} />
            </span>
            <div>
              <RoleBadge role={role.role} />
              <h3>{role.label}</h3>
            </div>
            <ul>
              {role.permissions.map((permission) => (
                <li key={permission}>
                  <Check size={13} /> {permission}
                </li>
              ))}
            </ul>
          </article>
        ))}
      </section>

      <section className="settings-users-heading">
        <div>
          <span className="eyebrow">USER MANAGEMENT</span>
          <h2>
            {isAdmin
              ? "Workspace accounts"
              : "Administrator-only account management"}
          </h2>
        </div>
        {isAdmin && (
          <span>
            {users.length} local account{users.length === 1 ? "" : "s"}
          </span>
        )}
      </section>
      {isAdmin ? (
        <section className="panel users-library-panel">
          <header className="panel-header">
            <div>
              <span className="eyebrow">LOCAL ACCOUNTS</span>
              <h2>Role and access status</h2>
              <p>
                Changing another account’s role or active state revokes that
                account’s current sessions and adds a workspace audit event.
              </p>
            </div>
            <UserCog size={20} />
          </header>
          <div className="users-filter">
            <label>
              <Search size={16} />
              <span className="sr-only">Search workspace users</span>
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search name, email, or role"
              />
            </label>
            <span>
              {visibleUsers.length} of {users.length} shown
            </span>
          </div>
          {loading ? (
            <div className="settings-empty">
              <LoaderCircle className="spin" size={18} /> Reading local
              accounts…
            </div>
          ) : !visibleUsers.length ? (
            <div className="settings-empty">
              <UsersRound size={18} /> No account matches this search.
            </div>
          ) : (
            <div className="table-scroll users-table">
              <table>
                <thead>
                  <tr>
                    <th>ACCOUNT</th>
                    <th>ROLE</th>
                    <th>STATUS</th>
                    <th>ROLE CHANGE</th>
                    <th>ACCESS</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleUsers.map((workspaceUser) => {
                    const isSelf = workspaceUser.id === user.id;
                    const busy = updatingId === workspaceUser.id;
                    return (
                      <tr
                        key={workspaceUser.id}
                        className={
                          !workspaceUser.active ? "workspace-user-inactive" : ""
                        }
                      >
                        <td>
                          <strong>{workspaceUser.name}</strong>
                          <small>
                            {workspaceUser.email}
                            {isSelf ? " · You" : ""}
                          </small>
                        </td>
                        <td>
                          <RoleBadge role={workspaceUser.role} />
                        </td>
                        <td>
                          <span
                            className={`workspace-status ${workspaceUser.active ? "active" : "inactive"}`}
                          >
                            {workspaceUser.active ? (
                              <UserRoundCheck size={13} />
                            ) : (
                              <UserRoundX size={13} />
                            )}
                            {workspaceUser.active ? "Active" : "Inactive"}
                          </span>
                        </td>
                        <td>
                          {isSelf ? (
                            <span className="users-self-lock">
                              <LockKeyhole size={13} /> Self protected
                            </span>
                          ) : (
                            <div className="users-role-action">
                              <select
                                value={workspaceUser.role}
                                disabled={busy}
                                onChange={(event) => {
                                  const nextRole = event.target.value as Role;
                                  if (nextRole !== workspaceUser.role)
                                    setChange({
                                      user: workspaceUser,
                                      role: nextRole,
                                    });
                                }}
                              >
                                <option value="scientist">
                                  Data Scientist
                                </option>
                                <option value="reviewer">Model Reviewer</option>
                                <option value="admin">
                                  MLOps Administrator
                                </option>
                              </select>
                            </div>
                          )}
                        </td>
                        <td>
                          {isSelf ? (
                            <span className="users-self-lock">
                              <LockKeyhole size={13} /> Self protected
                            </span>
                          ) : (
                            <button
                              className={`users-access-button ${workspaceUser.active ? "deactivate" : "activate"}`}
                              onClick={() =>
                                setChange({
                                  user: workspaceUser,
                                  active: !workspaceUser.active,
                                })
                              }
                              disabled={busy}
                            >
                              {busy ? (
                                <LoaderCircle className="spin" size={14} />
                              ) : workspaceUser.active ? (
                                <UserRoundX size={14} />
                              ) : (
                                <UserRoundCheck size={14} />
                              )}
                              {workspaceUser.active ? "Deactivate" : "Activate"}
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ) : (
        <section className="settings-access-limited">
          <span>
            <LockKeyhole size={19} />
          </span>
          <div>
            <strong>Account management is restricted.</strong>
            <p>
              Your current role is <RoleBadge role={user.role} />. Ask an MLOps
              Administrator to change roles or account access; the API rejects
              these actions for non-administrators.
            </p>
          </div>
        </section>
      )}
      {change && (
        <UserChangeDialog
          change={change}
          busy={Boolean(updatingId)}
          onClose={() => !updatingId && setChange(null)}
          onConfirm={() => void applyChange()}
        />
      )}
    </div>
  );
}
