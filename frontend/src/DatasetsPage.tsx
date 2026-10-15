import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { DragEvent, FormEvent } from "react";
import {
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  Columns3,
  Database,
  FileSpreadsheet,
  Fingerprint,
  Info,
  LoaderCircle,
  LockKeyhole,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Table2,
  TriangleAlert,
  UploadCloud,
  X,
} from "lucide-react";
import { api, ApiError } from "./api";
import type { DatasetList, DatasetPreview, DatasetReport, User } from "./api";
import "./datasets.css";

const MAX_BYTES = 200 * 1024 * 1024;
function sizeLabel(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
function number(value: number) {
  return value.toLocaleString();
}
function dateLabel(value: string | null) {
  return value
    ? new Date(value).toLocaleDateString(undefined, {
        day: "numeric",
        month: "short",
      })
    : "Existing upload";
}

function UploadDialog({
  onClose,
  onUploaded,
  onSessionExpired,
}: {
  onClose: () => void;
  onUploaded: (report: DatasetReport) => void;
  onSessionExpired: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const headingId = useId();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  function choose(files: FileList | null) {
    if (busy || !files?.length) return;
    setError("");
    if (files.length !== 1) {
      setError("Please upload one CSV at a time.");
      return;
    }
    const next = files[0];
    if (!next.name.toLowerCase().endsWith(".csv")) {
      setError("Choose a .csv file.");
      return;
    }
    if (!next.size || next.size > MAX_BYTES) {
      setError("Choose a non-empty CSV no larger than 200 MB.");
      return;
    }
    setFile(next);
  }
  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files);
  }
  async function upload(event: FormEvent) {
    event.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      onUploaded(
        await api<DatasetReport>("/datasets", { method: "POST", body }),
      );
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        onSessionExpired();
        return;
      }
      setError(
        e instanceof Error ? e.message : "Upload failed. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <dialog
      className="dialog dataset-upload-dialog"
      ref={ref}
      aria-labelledby={headingId}
      onCancel={(event) => {
        if (busy) event.preventDefault();
        else onClose();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">DATASET MANAGEMENT</span>
          <h2 id={headingId}>Bring your data to the workspace.</h2>
        </div>
        <button
          className="icon-button"
          aria-label="Close upload"
          onClick={onClose}
          disabled={busy}
        >
          <X size={20} />
        </button>
      </header>
      <p className="dialog-intro">
        We’ll validate your file and build a data-quality profile. Your CSV
        stays unchanged, and no model is trained.
      </p>
      <form onSubmit={upload}>
        <div
          className={`upload-dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            if (!busy) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={drop}
        >
          <span className="upload-symbol">
            <UploadCloud size={28} />
          </span>
          <strong>{file ? file.name : "Drop your CSV here"}</strong>
          <p>
            {file ? sizeLabel(file.size) : "or browse files on your computer"}
          </p>
          <input
            ref={input}
            id="dataset-file"
            type="file"
            accept=".csv,text/csv"
            aria-label="CSV dataset file"
            onChange={(event) => {
              choose(event.target.files);
              event.target.value = "";
            }}
            disabled={busy}
          />
          <button
            className="secondary-button"
            type="button"
            onClick={() => input.current?.click()}
            disabled={busy}
          >
            {file ? "Choose another file" : "Browse files"}
          </button>
          <small>UTF-8 CSV · up to 200 MB · 1,000,000 rows · 500 columns</small>
        </div>
        {error && (
          <div className="error-notice" role="alert">
            <TriangleAlert size={18} />
            {error}
          </div>
        )}
        {busy && (
          <div className="upload-progress" role="status">
            <LoaderCircle className="spin" size={18} />
            <div>
              <strong>Uploading and profiling your dataset…</strong>
              <p>
                Large files may take a little longer. Please keep this window
                open.
              </p>
            </div>
          </div>
        )}
        <div className="upload-safety">
          <ShieldCheck size={16} /> Saved as a new, immutable dataset. Existing
          files are never replaced.
        </div>
        <footer>
          <button
            className="secondary-button"
            type="button"
            onClick={onClose}
            disabled={busy}
          >
            Cancel
          </button>
          <button
            className="primary-button"
            type="submit"
            disabled={!file || busy}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <UploadCloud size={16} />
            )}
            {busy ? "Processing…" : "Upload & inspect"}
          </button>
        </footer>
      </form>
    </dialog>
  );
}

function QualityView({ report }: { report: DatasetReport }) {
  const missing = [...report.profile.columns]
    .filter((c) => c.missing_count > 0)
    .sort((a, b) => b.missing_count - a.missing_count)
    .slice(0, 8);
  return (
    <div className="quality-view">
      <div className="quality-summary">
        <span
          className={`quality-summary-icon ${report.quality.status === "passed" ? "teal" : "amber"}`}
        >
          {report.quality.status === "passed" ? (
            <ShieldCheck size={24} />
          ) : (
            <TriangleAlert size={24} />
          )}
        </span>
        <div>
          <h3>
            {report.quality.status === "passed"
              ? "Structural checks passed."
              : "A few things to review."}
          </h3>
          <p>
            {report.quality.status === "passed"
              ? "No warnings in these checks. Model readiness still depends on your objective, target, and evaluation design."
              : "Your dataset is valid, but these findings need consideration during planning. No values have been changed."}
          </p>
        </div>
      </div>
      <div className="quality-checks">
        {report.quality.checks.map((check) => (
          <article
            key={check.code}
            className={`quality-check check-${check.status}`}
          >
            <span className="check-icon">
              {check.status === "passed" ? (
                <Check size={16} />
              ) : check.status === "warning" ? (
                <TriangleAlert size={16} />
              ) : (
                <Info size={16} />
              )}
            </span>
            <div>
              <header>
                <h4>{check.label}</h4>
                <span className={`quality-badge quality-${check.status}`}>
                  {check.status === "passed"
                    ? "Passed"
                    : `${number(check.count)} ${check.status === "info" ? "to review" : "found"}`}
                </span>
              </header>
              <p>{check.message}</p>
              {check.columns.length > 0 && (
                <div className="column-tags">
                  {check.columns.slice(0, 8).map((name) => (
                    <span key={name}>{name}</span>
                  ))}
                  {check.columns.length > 8 && (
                    <span>+{check.columns.length - 8} more</span>
                  )}
                </div>
              )}
            </div>
          </article>
        ))}
      </div>
      {missing.length > 0 && (
        <section className="missing-values">
          <div className="section-heading">
            <h3>Missing values by column</h3>
            <span>Up to 8 columns with the most missing cells</span>
          </div>
          {missing.map((column) => (
            <div className="missing-bar-row" key={column.name}>
              <span>{column.name}</span>
              <div className="missing-bar">
                <span style={{ width: `${column.missing_fraction * 100}%` }} />
              </div>
              <strong>{(column.missing_fraction * 100).toFixed(1)}%</strong>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

export function DatasetsPage({
  user,
  onSessionExpired,
}: {
  user: User;
  onSessionExpired: () => void;
}) {
  const canUpload = user.permissions.includes("datasets:write");
  const [list, setList] = useState<DatasetList | null>(null);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [listOffset, setListOffset] = useState(0);
  const [selected, setSelected] = useState("");
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewOffset, setPreviewOffset] = useState(0);
  const [tab, setTab] = useState<"preview" | "columns" | "quality">("preview");
  const [columnSearch, setColumnSearch] = useState("");
  const [showUpload, setShowUpload] = useState(false);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [success, setSuccess] = useState("");
  const [revision, setRevision] = useState(0);
  const [listLoading, setListLoading] = useState(true);
  const handleError = useCallback(
    (e: unknown, setter: (message: string) => void) => {
      if (e instanceof ApiError && e.status === 401) {
        onSessionExpired();
        return;
      }
      if (e instanceof Error && e.name !== "AbortError") setter(e.message);
    },
    [onSessionExpired],
  );

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setListLoading(true);
    api<DatasetList>(
      `/datasets?search=${encodeURIComponent(query)}&offset=${listOffset}&limit=10`,
      { signal: controller.signal },
    )
      .then((result) => {
        setList(result);
        setSelected(
          (previous) => previous || result.items[0]?.dataset_name || "",
        );
      })
      .catch((e) => handleError(e, setError))
      .finally(() => {
        if (!controller.signal.aborted) setListLoading(false);
      });
    return () => controller.abort();
  }, [query, listOffset, revision, handleError]);

  useEffect(() => {
    setReport(null);
    setPreview(null);
    setPreviewOffset(0);
    setDetailError("");
    setColumnSearch("");
    if (!selected) return;
    const controller = new AbortController();
    api<DatasetReport>(`/datasets/${selected}`, { signal: controller.signal })
      .then(setReport)
      .catch((e) => handleError(e, setDetailError));
    return () => controller.abort();
  }, [selected, revision, handleError]);

  useEffect(() => {
    if (!report || report.dataset_name !== selected || tab !== "preview")
      return;
    const controller = new AbortController();
    setPreview(null);
    setPreviewError("");
    api<DatasetPreview>(
      `/datasets/${selected}/preview?offset=${previewOffset}&limit=20`,
      { signal: controller.signal },
    )
      .then(setPreview)
      .catch((e) => handleError(e, setPreviewError));
    return () => controller.abort();
  }, [report, selected, previewOffset, tab, handleError]);

  function uploaded(result: DatasetReport) {
    setShowUpload(false);
    setQuery("");
    setSearch("");
    setListOffset(0);
    setSelected(result.dataset_name);
    setTab("preview");
    setSuccess(
      `${result.display_name} was uploaded and validated. No training has started.`,
    );
    setRevision((value) => value + 1);
  }
  const columns =
    report?.profile.columns.filter((column) =>
      column.name.toLowerCase().includes(columnSearch.toLowerCase()),
    ) || [];
  return (
    <div className="datasets-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">ML LIFECYCLE / 01</div>
          <h1>
            Dataset Management<span className="greeting-dot">.</span>
          </h1>
          <p>Understand your data before the first experiment.</p>
        </div>
        <div className="dataset-heading-actions">
          <button
            className="secondary-button"
            aria-label="Refresh datasets"
            onClick={() => setRevision((value) => value + 1)}
            disabled={listLoading}
          >
            <RefreshCw size={16} className={listLoading ? "spin" : ""} />
          </button>
          {canUpload && (
            <button
              className="primary-button"
              onClick={() => setShowUpload(true)}
            >
              <Plus size={16} />
              Upload dataset
            </button>
          )}
        </div>
      </div>
      <section className="dataset-intro">
        <span className="dataset-intro-icon">
          <Database size={25} />
        </span>
        <div>
          <h2>Good models begin with understood data.</h2>
          <p>
            Upload a CSV, explore its structure, and review quality signals. The
            agent’s planning comes next.
          </p>
        </div>
        <span className="dataset-safety-pill">
          <ShieldCheck size={14} /> Original data preserved
        </span>
      </section>
      {!canUpload && (
        <div className="dataset-readonly">
          <LockKeyhole size={16} /> Reviewer access: inspect datasets and
          quality reports. Uploads require a Scientist or Administrator account.
        </div>
      )}
      {success && (
        <div className="dataset-success" role="status">
          <Check size={17} />
          <span>{success}</span>
          <button
            className="icon-button"
            aria-label="Dismiss upload confirmation"
            onClick={() => setSuccess("")}
          >
            <X size={16} />
          </button>
        </div>
      )}
      {error && (
        <div className="error-notice" role="alert">
          <TriangleAlert size={18} />
          {error}
        </div>
      )}
      <div className="dataset-layout">
        <section className="panel dataset-library">
          <header className="panel-header">
            <h2>Dataset library</h2>
            <span className="neutral-badge">
              {list ? number(list.total) : "—"}
            </span>
          </header>
          <form
            className="dataset-search"
            onSubmit={(event) => {
              event.preventDefault();
              setQuery(search);
              setListOffset(0);
              setSelected("");
            }}
          >
            <Search size={15} />
            <input
              aria-label="Search datasets"
              placeholder="Find a dataset…"
              value={search}
              maxLength={150}
              onChange={(event) => setSearch(event.target.value)}
            />
            <button type="submit" aria-label="Submit dataset search">
              <ArrowRight size={15} />
            </button>
          </form>
          <div
            className="dataset-list"
            aria-label="Saved datasets"
            aria-busy={listLoading}
          >
            {listLoading ? (
              <div className="library-empty">
                <LoaderCircle className="spin" size={21} />
                <p>Loading datasets…</p>
              </div>
            ) : list?.items.length ? (
              list.items.map((item) => (
                <button
                  className={`dataset-list-item ${selected === item.dataset_name ? "selected" : ""}`}
                  key={item.dataset_name}
                  onClick={() => {
                    setSelected(item.dataset_name);
                    setSuccess("");
                  }}
                  aria-pressed={selected === item.dataset_name}
                >
                  <span className="file-icon">
                    <FileSpreadsheet size={19} />
                  </span>
                  <span className="dataset-item-text">
                    <strong>{item.display_name}</strong>
                    <small>
                      {item.rows !== null ? `${number(item.rows)} rows · ` : ""}
                      {sizeLabel(item.size_bytes)}
                    </small>
                    <span
                      className={`quality-badge quality-${item.quality_status}`}
                    >
                      {item.quality_status === "passed"
                        ? "Checks passed"
                        : item.quality_status === "warnings"
                          ? "Review warnings"
                          : "Inspect to profile"}
                    </span>
                  </span>
                </button>
              ))
            ) : (
              <div className="library-empty">
                <Database size={24} />
                <h3>
                  {query ? "No matching datasets" : "Your library starts here"}
                </h3>
                <p>
                  {query
                    ? "Try another filename or dataset identifier."
                    : "Upload your first CSV to explore its structure and quality."}
                </p>
                {canUpload && !query && (
                  <button
                    className="text-button"
                    onClick={() => setShowUpload(true)}
                  >
                    Upload a CSV <ArrowRight size={14} />
                  </button>
                )}
              </div>
            )}
          </div>
          {list && list.total > 10 && (
            <div className="library-pagination">
              <button
                className="icon-button"
                aria-label="Previous datasets"
                disabled={listOffset === 0 || listLoading}
                onClick={() => setListOffset((value) => value - 10)}
              >
                <ChevronLeft size={16} />
              </button>
              <span>
                {listOffset + 1}–{Math.min(listOffset + 10, list.total)} of{" "}
                {number(list.total)}
              </span>
              <button
                className="icon-button"
                aria-label="Next datasets"
                disabled={listOffset + 10 >= list.total || listLoading}
                onClick={() => setListOffset((value) => value + 10)}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          )}
          <footer>
            <LockKeyhole size={13} />
            <p>
              Registered uploads only. For CSVs copied manually into the
              project, upload them here to add them to the library.
            </p>
          </footer>
        </section>
        <div
          className="dataset-detail"
          aria-busy={!!selected && !report && !detailError}
        >
          {detailError ? (
            <section className="panel dataset-detail-empty">
              <TriangleAlert size={30} />
              <h2>This dataset needs attention.</h2>
              <p role="alert">{detailError}</p>
              <button
                className="secondary-button"
                onClick={() => setRevision((value) => value + 1)}
              >
                Try again
              </button>
            </section>
          ) : !report ? (
            <section className="panel dataset-detail-empty">
              {selected ? (
                <LoaderCircle className="spin" size={29} />
              ) : (
                <span className="empty-icon">
                  <Table2 size={28} />
                </span>
              )}
              <h2>
                {selected
                  ? "Inspecting your dataset…"
                  : "A closer look at your data."}
              </h2>
              <p>
                {selected
                  ? "Calculating column types, completeness, and quality signals."
                  : "Choose a dataset from the library, or upload a CSV to see a readable preview and profile here."}
              </p>
              <div className="empty-detail-features">
                <span>
                  <Table2 size={15} /> Row previews
                </span>
                <span>
                  <Columns3 size={15} /> Column profiles
                </span>
                <span>
                  <ShieldCheck size={15} /> Quality checks
                </span>
              </div>
            </section>
          ) : (
            <>
              <section className="panel dataset-inspector">
                <header className="inspector-heading">
                  <span className="file-icon">
                    <FileSpreadsheet size={24} />
                  </span>
                  <div>
                    <h2>{report.display_name}</h2>
                    <p>
                      {sizeLabel(report.size_bytes)} <span>·</span>{" "}
                      {dateLabel(report.uploaded_at)} <span>·</span> CSV dataset
                    </p>
                  </div>
                  <span
                    className={`quality-badge quality-${report.quality.status}`}
                  >
                    {report.quality.status === "passed" ? (
                      <Check size={13} />
                    ) : (
                      <TriangleAlert size={13} />
                    )}
                    {report.quality.status === "passed"
                      ? "Checks passed"
                      : "Review warnings"}
                  </span>
                </header>
                <div className="dataset-metrics">
                  <div>
                    <span>Total rows</span>
                    <strong>{number(report.profile.rows)}</strong>
                  </div>
                  <div>
                    <span>Columns</span>
                    <strong>{number(report.profile.columns.length)}</strong>
                  </div>
                  <div>
                    <span>Completeness</span>
                    <strong>{report.quality.completeness}%</strong>
                    <small>Non-missing cells</small>
                  </div>
                  <div>
                    <span>Duplicate rows</span>
                    <strong>{number(report.quality.duplicate_rows)}</strong>
                    <small>After the first occurrence</small>
                  </div>
                </div>
                <div
                  className="dataset-tabs"
                  role="tablist"
                  aria-label="Dataset inspection"
                >
                  {(
                    [
                      { id: "preview", title: "Data preview", icon: Table2 },
                      {
                        id: "columns",
                        title: "Column profile",
                        icon: Columns3,
                      },
                      {
                        id: "quality",
                        title: "Data quality",
                        icon: ShieldCheck,
                      },
                    ] as const
                  ).map((item) => (
                    <button
                      key={item.id}
                      role="tab"
                      id={`tab-${item.id}`}
                      aria-controls={`panel-${item.id}`}
                      aria-selected={tab === item.id}
                      tabIndex={tab === item.id ? 0 : -1}
                      onKeyDown={(event) => {
                        const tabs = ["preview", "columns", "quality"] as const;
                        if (
                          !["ArrowLeft", "ArrowRight", "Home", "End"].includes(
                            event.key,
                          )
                        )
                          return;
                        event.preventDefault();
                        const index =
                          event.key === "Home"
                            ? 0
                            : event.key === "End"
                              ? 2
                              : (tabs.indexOf(tab) +
                                  (event.key === "ArrowRight" ? 1 : 2)) %
                                3;
                        setTab(tabs[index]);
                        document.getElementById(`tab-${tabs[index]}`)?.focus();
                      }}
                      onClick={() => setTab(item.id)}
                    >
                      <item.icon size={16} />
                      {item.title}
                      {item.id === "quality" &&
                        report.quality.status === "warnings" && (
                          <span className="warning-dot" />
                        )}
                    </button>
                  ))}
                </div>
                <div
                  role="tabpanel"
                  id={`panel-${tab}`}
                  aria-labelledby={`tab-${tab}`}
                >
                  {tab === "preview" && (
                    <>
                      {previewError ? (
                        <div className="error-notice" role="alert">
                          {previewError}
                          <button
                            className="text-button"
                            onClick={() => setRevision((value) => value + 1)}
                          >
                            Retry
                          </button>
                        </div>
                      ) : !preview ? (
                        <div className="preview-loading" role="status">
                          <LoaderCircle className="spin" size={21} />
                          Loading preview…
                        </div>
                      ) : (
                        <>
                          <div
                            className="data-preview-scroll"
                            tabIndex={0}
                            role="region"
                            aria-label="Dataset rows; scroll horizontally for more columns"
                          >
                            <table className="data-preview-table">
                              <thead>
                                <tr>
                                  <th scope="col">ROW</th>
                                  {preview.columns.map((column) => (
                                    <th key={column} scope="col">
                                      {column}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {preview.rows.map((row, rowIndex) => (
                                  <tr key={preview.offset + rowIndex}>
                                    <th scope="row">
                                      {number(preview.offset + rowIndex + 1)}
                                    </th>
                                    {row.map((value, index) => (
                                      <td key={index}>
                                        {value === null ? (
                                          <span className="missing-cell">
                                            Missing / non-finite
                                          </span>
                                        ) : (
                                          String(value)
                                        )}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <div className="preview-pagination">
                            <span>
                              Rows{" "}
                              {preview.rows.length
                                ? number(preview.offset + 1)
                                : "0"}
                              –{number(preview.offset + preview.rows.length)} of{" "}
                              {number(preview.total_rows)}
                            </span>
                            <div>
                              <button
                                className="secondary-button"
                                disabled={previewOffset === 0}
                                onClick={() =>
                                  setPreviewOffset((value) =>
                                    Math.max(0, value - 20),
                                  )
                                }
                              >
                                <ChevronLeft size={14} />
                                Previous
                              </button>
                              <button
                                className="secondary-button"
                                disabled={
                                  previewOffset + 20 >= preview.total_rows
                                }
                                onClick={() =>
                                  setPreviewOffset((value) => value + 20)
                                }
                              >
                                Next
                                <ChevronRight size={14} />
                              </button>
                            </div>
                          </div>
                          <p className="preview-note">
                            Preview only · 20 rows per page · cells shortened to
                            200 characters. The original CSV is unchanged.
                          </p>
                        </>
                      )}
                    </>
                  )}
                  {tab === "columns" && (
                    <div className="column-profile">
                      <label className="column-search">
                        <Search size={15} />
                        <input
                          placeholder="Search columns…"
                          aria-label="Search columns"
                          value={columnSearch}
                          onChange={(event) =>
                            setColumnSearch(event.target.value)
                          }
                        />
                      </label>
                      <div
                        className="data-preview-scroll"
                        tabIndex={0}
                        role="region"
                        aria-label="Column profiles"
                      >
                        <table>
                          <thead>
                            <tr>
                              <th>Column</th>
                              <th>Inferred type</th>
                              <th>Missing</th>
                              <th>Distinct values</th>
                              <th>Signals</th>
                            </tr>
                          </thead>
                          <tbody>
                            {columns.map((column) => (
                              <tr key={column.name}>
                                <td>
                                  <strong>{column.name}</strong>
                                  <small>{column.pandas_dtype}</small>
                                </td>
                                <td>
                                  <span
                                    className={`type-badge type-${column.inferred_type}`}
                                  >
                                    {column.inferred_type}
                                  </span>
                                </td>
                                <td>
                                  {number(column.missing_count)}{" "}
                                  <small>
                                    {(column.missing_fraction * 100).toFixed(1)}
                                    %
                                  </small>
                                </td>
                                <td>{number(column.unique_count)}</td>
                                <td>
                                  {column.likely_identifier && (
                                    <span className="signal-badge">
                                      Possible ID
                                    </span>
                                  )}
                                  {report.profile.likely_targets.includes(
                                    column.name,
                                  ) && (
                                    <span className="signal-badge target-signal">
                                      Target hint
                                    </span>
                                  )}
                                  {!column.likely_identifier &&
                                    !report.profile.likely_targets.includes(
                                      column.name,
                                    ) && <span className="muted-dash">—</span>}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {columns.length === 0 && (
                          <p className="no-columns">
                            No columns match your search.
                          </p>
                        )}
                      </div>
                      <p className="preview-note">
                        Types are inferred from the CSV. Identifier and target
                        hints are suggestions to review during planning.
                      </p>
                    </div>
                  )}
                  {tab === "quality" && <QualityView report={report} />}
                </div>
                <footer className="dataset-fingerprint">
                  <Fingerprint size={15} />
                  <span>
                    SHA-256{" "}
                    <code title={report.profile.data_hash}>
                      {report.profile.data_hash.slice(0, 16)}…
                    </code>
                  </span>
                  <span>Immutable upload</span>
                </footer>
              </section>
              <section className="dataset-next">
                <span className="role-icon blue">
                  <ArrowRight size={19} />
                </span>
                <div>
                  <span className="eyebrow">NEXT STEP · AI AGENT PLANNING</span>
                  <h3>Turn data into an experiment plan.</h3>
                  <p>
                    Target selection, metrics, preprocessing, and model choices
                    are now ready for controlled planning.
                  </p>
                </div>
                <a className="text-button" href="#planning">
                  Open planning <ArrowRight size={14} />
                </a>
              </section>
            </>
          )}
        </div>
      </div>
      {showUpload && (
        <UploadDialog
          onClose={() => setShowUpload(false)}
          onUploaded={uploaded}
          onSessionExpired={onSessionExpired}
        />
      )}
    </div>
  );
}
