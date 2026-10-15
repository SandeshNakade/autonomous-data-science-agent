# Custom web workspace — stages 01–13

The new React + TypeScript interface replaces Streamlit as the primary web entry
point. FastAPI serves its built assets and APIs on the same origin. The existing
ML engine, datasets, models, and experiment artifacts remain in place.

## What is implemented now

- Responsive welcome/sign-in page, inspired by the supplied light/blue references.
- Local accounts with three server-assigned roles; no public registration or default passwords.
- Hashed passwords, opaque HttpOnly session cookies, eight-hour expiry, revocable logout,
  same-origin request checks, and basic per-account failed-login throttling.
- Lifecycle sidebar with subsequent pages explicitly marked **Upcoming**.
- Read-only overview of real saved run counts, pending candidates, failures, and
  production registry presence. Recent runs are shown in a readable table, not JSON.
- Current-permissions dialog and a lifecycle guide explaining review/retraining gates.
- Dataset Management: immutable CSV uploads, a searchable library, paged row previews,
  searchable column profiles, and explained quality checks. Upload validation rejects
  malformed CSV files, duplicate or blank headers, oversized files, and datasets outside
  the configured row/column limits; successful files are saved unchanged and do not
  start training.
- AI Agent Planning: selected dataset + natural-language objective in, a reviewable
  policy-validated target/task/metric/preprocessing/model plan out. Operator target,
  threshold, and experiment-budget inputs are authoritative. Plan preview does not
  run preprocessing, fit a model, create an experiment record, or alter a dataset.
- Workflow Execution: an explicit confirmation starts the existing bounded server-side
  workflow. Its real persisted milestones (profiling, policy validation, data split,
  agent decisions, experiments, final holdout, approval submission, and completion)
  are presented as a timeline and readable result tables. A successful candidate is
  still only **submitted for human approval**; execution cannot approve, deploy, or
  retrain a model.
- Experiments & Model Comparison: users select a recorded workflow and compare only its
  completed recipes, using its own primary metric, supporting metrics, timings, target,
  and split context. The page treats RMSE as lower-is-better and keeps results from
  different workflows separate. It is a read-only evidence page and cannot submit,
  approve, deploy, or retrain a model.
- Model Approval Center: reviewers and administrators can inspect a candidate's saved
  validation/holdout evidence, approve or reject it with a required reason, and read
  its audit trail. An approval changes the state to **approved for deployment** only;
  it cannot deploy, promote, replace, or serve a production model.
- Model Deployment: only an administrator can activate a reviewer-approved candidate.
  The action requires a release note, stores an immutable local release record, switches
  the active local model, and exposes its protected single-record prediction test without
  asking users to type raw JSON. Legacy candidates can be read but cannot bypass the
  reviewer-approved deployment path.
- Monitoring & Drift: any signed-in role can read saved monitoring evidence; only an
  administrator can compare an uploaded batch to the active release baseline. The page
  renders numeric PSI and categorical TVD tables, missing expected features, alert
  history, and a controlled retraining request. It does not claim prediction drift or
  production performance without a stored prediction/outcome feedback feed, and it never
  automatically retrains, promotes, or replaces the active model.
- Experiment History: any signed-in role can search and inspect saved workflow records.
  Each selected record presents the persisted objective, plan summary, dataset fingerprint,
  controlled experiment results, final-holdout evidence, and candidate handoff without
  offering any action that changes a run, dataset, model, or approval state.
- Model Registry: any signed-in role can inspect every saved candidate's metric evidence,
  plan and experiment lineage, governance events, release metadata, and active-serving
  state. This is a read-only evidence surface; the separate Approval Center and Model
  Deployment pages retain their existing backend-enforced mutation permissions.
- Alerts & Notifications: any signed-in role can review an in-app inbox of unresolved
  persisted feature-drift reports, failed workflows, and candidate-review reminders. It
  links to the relevant evidence page without exposing local paths or creating a new
  model-lifecycle action. Email, SMS, push delivery, acknowledgement, and escalation
  rules are deliberately not simulated in this prototype.
- Audit Logs: any signed-in role can search a safe, read-only aggregation of retained
  governance audit events, feature-drift checks, and workflow outcomes. It preserves a
  missing actor as **Not recorded**, links to protected evidence pages, and does not claim
  a centralized append-only ledger, login history, download history, or hidden events.
- Settings & User Management: every signed-in role can inspect the fixed local workspace
  configuration and role policy. Only an administrator can list local accounts or change
  another account's role/active state; each change revokes that account's sessions and adds
  a password-free workspace administration event to Audit Logs. Administrators cannot
  modify their own access here, and at least one active administrator is always retained.
  Account creation and password management remain terminal-only.

All planned academic workspace pages are now connected. This remains a local research
prototype, not an enterprise control plane: the overview does not claim endpoint health or
show fabricated performance charts.

## Start locally (PowerShell)

Run from the repository root. On this computer use the approved interpreter:

```powershell
$agentPython = 'C:\Users\sande\AppData\Local\Python\pythoncore-3.14-64\python.exe'
```

On another computer use its working Python interpreter. Install the Python package
if not already installed: `python -m pip install -e ".[dev]"`.

Build the frontend once, and again after frontend changes (Node 22.12+ or 24):

```powershell
npm.cmd --prefix frontend install
npm.cmd --prefix frontend run build
```

Create your administrator account locally. The command asks for the password twice;
typing is intentionally invisible. Passwords must have 12–128 characters. An existing
account is never silently overwritten. Keep the terminal password private.

```powershell
& $agentPython -m autods.auth --email sandesh@example.com --name "Sandesh" --role admin
```

Use the same command with a different email and `--role scientist` or `--role reviewer`
to provision the other roles. Account creation does not send any email.

Start the API **after** building the frontend:

```powershell
& $agentPython -m uvicorn autods.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> and sign in with the account you just created. If an older
API instance already owns port 8000, stop that instance first or choose another port.
Accounts persist in `artifacts/workspace.sqlite3` (ignored by Git).

For frontend development, leave FastAPI on port 8000 and run `npm.cmd --prefix frontend
run dev` in a second terminal. Open the Vite URL, normally <http://127.0.0.1:5173>.
Vite proxies the API routes to FastAPI; no permissive CORS configuration is needed.
`npm run preview` alone does not serve the API—use FastAPI for integrated previews.

## Current API permissions

All workspace/data/run/model reads require a session. Health and authentication
status are public. The role sent by a browser is never trusted.

| Operation | Scientist | Reviewer | MLOps / Admin |
| --- | --- | --- | --- |
| Read runs, model metadata, overview | Yes | Yes | Yes |
| List datasets, inspect profiles, preview rows | Yes | Yes | Yes |
| Generate a plan-only agent recommendation | Yes | No | Yes |
| Upload CSV / start bounded runs | Yes | No | Yes |
| Test predictions | Yes | Yes | Yes |
| Review candidate, approve/reject with a reason | No | Yes | Yes |
| Deploy reviewer-approved candidate with a release note | No | No | Yes |
| Execute drift check / controlled retraining | No | No | Yes |
| List accounts; change another role or active state | No | No | Yes |
| Existing combined approve-and-deploy endpoint | No | No | Yes + approval token |

**Important migration detail:** `POST /model-reviews/{proposal_id}` records an
authenticated reviewer decision, reason, timestamp, and audit event. An approval
sets the candidate to `approved_for_deployment`; it does not deploy it.
`POST /deployments/{proposal_id}` accepts only that state, requires an administrator and
a release note, records the authenticated deployment actor, retains a local release copy,
and then switches the local production model. The existing registry `approve()` remains
a legacy combined promotion operation restricted to an administrator plus the
`X-Approval-Token` gate. The authenticated identity, not the request body's
`approved_by`, is recorded on legacy promotions. Retraining also still requires the
approval-token gate.

The APIs now require a session cookie. Sign in using `POST /auth/login` with JSON
`email` and `password`, retain the cookie, and include `X-AutoDS-Client: web` on
every mutating request. This header is a cross-site-request defense, not a secret.
Do not enable wildcard credentialed CORS. `/auth/me` returns the user's current
role and permissions; `/auth/logout` revokes the session.

## Scope and security boundaries

- These are local, single-workspace prototype accounts—not enterprise SSO or tenant isolation.
- The administrator provisions new users via the local CLI. The web page can change another
  account's role or active state, revokes its sessions, and records that administration
  action; it cannot create accounts, change passwords, recover passwords, or alter itself.
- Anyone with terminal/filesystem access already controls this local project. The
  CLI and legacy Streamlit surface are trusted local tools and do not use web RBAC.
  Do not expose the legacy dashboard alongside this as a protected multi-user service.
- Keep HTTP bound to loopback for the demonstration. For a remote deployment, use
  HTTPS, set `AUTODS_COOKIE_SECURE=true`, preserve the public Host/scheme at the
  reverse proxy, add infrastructure rate limiting, and review deployment security.
- `AUTODS_AUTH_DB` optionally sets a different SQLite file. Account provisioning and
  the API must use the same setting. `AUTODS_FRONTEND_DIR` can override the build path.
- No API keys are requested by, stored in, or sent to the frontend.
- Authentication state stays in an HttpOnly cookie, not localStorage.

## Implementation order

1. **Delivered:** welcome/sign-in, three-role API foundation, navigation, initial overview.
2. **Delivered:** dataset upload, library/search, row previews, column profiles, validation,
   completeness and duplicate counts, missing-value bars, and quality explanations.
3. **Delivered:** AI planning: target/task, metrics, preprocessing, allowlisted models,
   acceptance criteria, first controlled experiment recommendation, and explicit guardrails.
4. **Delivered:** Workflow Execution: explicit confirmation, bounded server-side run,
   persisted lifecycle events, agent decision loop, experiment result table, final-holdout
   evidence, and a human-approval handoff.
5. **Delivered:** Experiments & Model Comparison: saved-run selector, within-run score
   cards, metric/timing table, primary-metric ordering, and selection context without any
   promotion action.
6. **Delivered:** Model Approval Center: reviewer/admin-only approve or reject action,
   required rationale, immutable local audit events, and approved-for-deployment state
   without any production promotion.
7. **Delivered:** Model Deployment: administrator-only release of a reviewer-approved
   candidate, release note and actor audit, retained local version record, active endpoint
   status, and a form-based protected prediction test.
8. **Delivered:** Monitoring & Drift: authenticated report history, administrator-only
   data/feature-drift checks against the active release baseline, PSI/TVD evidence tables,
   drift alerts, and a human-gated retraining request. Prediction drift and model
   performance remain unmeasured until an outcome-feedback feed is added.
9. **Delivered:** Experiment History: searchable saved workflow records, run-plan/data
   snapshots, experiment result table, final selection evidence, and read-only handoff
   links to the comparison workspace.
10. **Delivered:** Model Registry: searchable candidate/release catalog, validation and
    holdout evidence, plan and run lineage, reviewer/deployment audit trail, and
    read-only active-release visibility.
11. **Delivered:** Alerts & Notifications: authenticated, read-only in-app signals derived
    only from persisted drift, failed-workflow, and pending-review evidence; filters,
    severity counts, and safe links to the source workflow; no delivery or acknowledgement
    simulation.
12. **Delivered:** Audit Logs: authenticated, searchable read-only aggregation of
    persisted governance events, monitoring checks, and workflow outcomes, with safe
    evidence links and explicit scope boundaries for absent actors and non-existent logs.
13. **Delivered:** Settings & User Management: safe fixed-configuration display, role
    policy visibility, administrator-only local-account listing and role/active-state
    changes, self-protection, last-active-admin protection, session revocation, and
    password-free workspace-administration audit events.

The lifecycle and supporting navigation are now fully connected. New account provisioning
and password operations intentionally remain local terminal commands rather than browser forms.

## Verification

```powershell
& $agentPython -m pytest
npm.cmd --prefix frontend run build
```

The backend tests cover sessions, expiration, revocation, role escalation attempts,
direct forbidden API requests, CSRF guards, authenticated attribution, dataset upload
validation, immutable storage, profiles, preview pagination, quality checks, agent-plan
policy validation, plan-only guarantees, and existing ML workflows. Browser checks should
cover sign-in errors, successful login, upload validation, dataset inspection, plan
generation, logout, and narrow-screen layout.

The backend suite covers the existing ML workflow, and the focused Stage 04 API workflow
test verifies that a completed run persists each lifecycle milestone rendered by the interface.
The Stage 04, Stage 05, Stage 06, Stage 07, Stage 08, Stage 09, Stage 10, Stage 11, Stage 12, and Stage 13 production frontend builds pass. Focused
Stage 06 API tests verify that a reviewer can approve or reject with an audit reason,
that neither action creates a production model, and that a scientist is forbidden from
reviewing. Focused Stage 07 tests verify that deployment remains unavailable to scientists
and reviewers, records the authenticated administrator, requires a reviewer-approved
candidate, retains a release artifact, and then serves the active model.
Focused Stage 08 tests verify that monitoring reports require authentication, a new report
records the authenticated administrator, and no response exposes a local report path.
Stage 09 is a frontend-only reader over the existing authenticated saved-run APIs; its
production frontend build passes without adding a new mutating endpoint.
Stage 10 is likewise a frontend-only reader over the authenticated model-proposal and
deployment-status APIs; it adds no model-file download or lifecycle-changing endpoint.
Stage 11 adds the authenticated `GET /alerts` reader and a focused API test that proves it
derives only safe open events from persisted monitoring, workflow, and governance records.
Stage 12 adds the authenticated, paginated `GET /audit-logs` reader and a focused API test
that verifies it safely aggregates persisted governance, monitoring, and workflow evidence
without local paths.
Stage 13 adds authenticated `GET /auth/settings`, administrator-only `GET /auth/users`, and
administrator-only `PATCH /auth/users/{user_id}`. Focused tests verify the non-admin block,
safe metadata-only account listing, self-protection, session revocation after an access
change, and the resulting workspace audit event.
Browser checks have covered dataset upload, preview, column-profile, and data-quality
views; Stage 03 has API coverage for the plan-only, policy-validated agent recommendation
and role enforcement. The execution, comparison, approval, and deployment interfaces
read persisted artifacts without rendering raw JSON. Screenshots are saved locally under
`artifacts/ui-proof/`. The Docker build configuration has been updated for the frontend,
but a container build was not verified on this workstation.
