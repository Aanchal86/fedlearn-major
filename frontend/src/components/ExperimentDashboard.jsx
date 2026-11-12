import { useEffect, useRef, useState } from "react";
import { datasetApi, experimentApi } from "../services/api";
import "./ExperimentDashboard.css";
import {
  downloadText, safeFilename, experimentRoundsCsv, experimentClientsCsv,
  experimentSummary, comparisonCsv, comparisonRoundsCsv,
} from "../utils/reportExports";

function MetricChart({ history, field, title, percent = false }) {
  const points = history.filter((round) => Number.isFinite(round[field]));
  if (points.length === 0) return <p className="chart-empty">Waiting for the first round...</p>;
  const values = points.map((round) => round[field]);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = maximum - minimum || 1;
  const coords = points.map((round, index) => {
    const x = 35 + (index * 510) / Math.max(1, points.length - 1);
    const y = 165 - ((round[field] - minimum) / range) * 125;
    return `${x},${y}`;
  }).join(" ");
  const format = (value) => percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  return (
    <div className="metric-chart">
      <h3>{title}</h3>
      <svg viewBox="0 0 580 205" role="img" aria-label={`${title} across completed rounds`}>
        <line x1="35" y1="165" x2="550" y2="165" stroke="#cbd5e1" />
        <line x1="35" y1="35" x2="35" y2="165" stroke="#cbd5e1" />
        <text x="2" y="40" fontSize="12" fill="#64748b">{format(maximum)}</text>
        <text x="2" y="165" fontSize="12" fill="#64748b">{format(minimum)}</text>
        <polyline points={coords} fill="none" stroke="#4f46e5" strokeWidth="3" strokeLinejoin="round" />
        {points.map((round, index) => {
          const [cx, cy] = coords.split(" ")[index].split(",").map(Number);
          return <circle key={round.round_number} cx={cx} cy={cy} r="4" fill="#4f46e5"><title>{`Round ${round.round_number}: ${format(round[field])}`}</title></circle>;
        })}
        <text x="290" y="198" textAnchor="middle" fontSize="12" fill="#64748b">Training round</text>
      </svg>
    </div>
  );
}


function ComparisonChart({ left, right, field, title, percent = false }) {
  const series = [
    { name: left.name, color: "#4f46e5", history: left.history || [] },
    { name: right.name, color: "#e11d48", history: right.history || [] },
  ];
  const all = series.flatMap((item) => item.history.filter((round) => Number.isFinite(round[field])));
  if (!all.length) return <p>No round metrics available.</p>;
  const values = all.map((round) => round[field]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const range = high - low || 1;
  const maxRound = Math.max(1, ...all.map((round) => round.round_number));
  const toX = (round) => 48 + ((round - 1) / Math.max(1, maxRound - 1)) * 480;
  const toY = (value) => 160 - ((value - low) / range) * 115;
  const format = (value) => percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  return (
    <div className="metric-chart comparison-chart">
      <h3>{title}</h3>
      <svg viewBox="0 0 560 205" role="img" aria-label={`${title} comparison by training round`}>
        <line x1="48" y1="160" x2="530" y2="160" stroke="#cbd5e1" />
        <line x1="48" y1="38" x2="48" y2="160" stroke="#cbd5e1" />
        <text x="3" y="42" fontSize="12" fill="#64748b">{format(high)}</text>
        <text x="3" y="158" fontSize="12" fill="#64748b">{format(low)}</text>
        <text x="48" y="181" fontSize="12" fill="#64748b">1</text>
        <text x="530" y="181" textAnchor="end" fontSize="12" fill="#64748b">{maxRound}</text>
        <text x="280" y="200" textAnchor="middle" fontSize="12" fill="#64748b">Training round</text>
        {series.map((item) => {
          const rounds = item.history.filter((round) => Number.isFinite(round[field]));
          const points = rounds.map((round) => `${toX(round.round_number)},${toY(round[field])}`).join(" ");
          return (
            <g key={item.name + item.color}>
              <polyline points={points} fill="none" stroke={item.color} strokeWidth="3" strokeLinejoin="round" />
              {rounds.map((round) => (
                <circle key={round.round_number} cx={toX(round.round_number)} cy={toY(round[field])} r="3.5" fill={item.color}>
                  <title>{`${item.name}, round ${round.round_number}: ${format(round[field])}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <div className="comparison-legend">
        {series.map((item) => <span key={item.color}><i style={{ background: item.color }} />{item.name}</span>)}
      </div>
    </div>
  );
}

function ExperimentComparison({ left, right }) {
  const comparisonName = `comparison_${left.experiment_id}_vs_${right.experiment_id}`;
  const exportComparison = (kind) => {
    const content = kind === "metrics" ? comparisonCsv(left, right) : comparisonRoundsCsv(left, right);
    downloadText(`${comparisonName}_${kind}.csv`, content, "text/csv;charset=utf-8");
  };
  const leftLast = left.history?.at(-1);
  const rightLast = right.history?.at(-1);
  const metrics = [
    { label: "Final accuracy", field: "global_accuracy", higher: true, percent: true },
    { label: "Final loss", field: "global_loss", higher: false, percent: false },
    { label: "Final macro F1", field: "global_macro_f1", higher: true, percent: true },
  ];
  const configRows = [
    ["Dataset", "dataset"], ["Clients", "num_clients"], ["Rounds", "num_rounds"],
    ["Local epochs", "local_epochs"], ["Batch size", "batch_size"],
    ["Learning rate", "learning_rate"], ["Aggregation", "aggregation_method"],
    ["Partition", "partition_type"], ["Dirichlet alpha", "dirichlet_alpha"],
  ];
  const fmt = (value, percent = false) => Number.isFinite(value)
    ? (percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(4)) : "â€”";
  return (
    <div className="comparison-output">
      {left.configuration?.dataset !== right.configuration?.dataset && (
        <p className="alert alert-error">These experiments use different datasets. Their performance scores are not directly comparable.</p>
      )}
      <div className="export-actions">
        <button type="button" className="export-button" onClick={() => exportComparison("metrics")}>Download Comparison Metrics CSV</button>
        <button type="button" className="export-button" onClick={() => exportComparison("rounds")}>Download Learning Curves CSV</button>
        <button type="button" className="export-button" onClick={() => downloadText(`${comparisonName}.json`, JSON.stringify({ experiment_A: left, experiment_B: right }, null, 2), "application/json;charset=utf-8")}>Download Comparison JSON</button>
      </div>
      <h3>Final metrics</h3>
      <div className="results-scroll">
        <table className="results-table">
          <thead><tr><th>Metric</th><th>{left.name}</th><th>{right.name}</th><th>Difference (B âˆ’ A)</th><th>Better</th></tr></thead>
          <tbody>
            {metrics.map((metric) => {
              const a = leftLast?.[metric.field];
              const b = rightLast?.[metric.field];
              const valid = Number.isFinite(a) && Number.isFinite(b);
              const difference = valid ? b - a : null;
              const better = !valid ? "N/A" : Math.abs(difference) < 1e-10 ? "Tie" :
                (difference > 0) === metric.higher ? "B" : "A";
              return <tr key={metric.field}><td>{metric.label}</td><td>{fmt(a, metric.percent)}</td><td>{fmt(b, metric.percent)}</td><td>{valid ? `${difference > 0 ? "+" : ""}${fmt(difference, metric.percent)}` : "â€”"}</td><td>{better}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
      <p className="form-note">A = first experiment, B = second experiment. Accuracy and F1 differences are percentage points; lower loss is better. Final metrics use each experiment's last saved round.</p>
      <h3>Learning curves</h3>
      <div className="monitor-charts">
        <ComparisonChart left={left} right={right} field="global_accuracy" title="Accuracy" percent />
        <ComparisonChart left={left} right={right} field="global_loss" title="Loss" />
        <ComparisonChart left={left} right={right} field="global_macro_f1" title="Macro F1" percent />
      </div>
      <h3>Experiment configuration</h3>
      <div className="results-scroll">
        <table className="results-table">
          <thead><tr><th>Setting</th><th>{left.name}</th><th>{right.name}</th></tr></thead>
          <tbody>{configRows.map(([label, key]) => (
            <tr key={key}><td>{label}</td><td>{String(left.configuration?.[key] ?? "â€”")}</td><td>{String(right.configuration?.[key] ?? "â€”")}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}

const initialForm = {
  name: "",
  dataset: "fashion_mnist",
  num_clients: 5,
  num_rounds: 10,
  local_epochs: 1,
  batch_size: 64,
  learning_rate: 0.001,
  aggregation_method: "fedavg",
  partition_type: "iid",
  dirichlet_alpha: 0.5,
};

export default function ExperimentDashboard() {
  const [datasets, setDatasets] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [startingId, setStartingId] = useState(null);
  const [selectedResults, setSelectedResults] = useState(null);
  const [selectedExperimentId, setSelectedExperimentId] = useState(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const selectedIdRef = useRef(null);
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");
  const [comparison, setComparison] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState("");


  useEffect(() => {
    async function loadData() {
      try {
        const [datasetData, experimentData] =
          await Promise.all([
            datasetApi.getAll(),
            experimentApi.getAll(),
          ]);

        setDatasets(datasetData);
        setExperiments(experimentData.experiments);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  function updateField(event) {
    const { name, value, type } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: type === "number" ? Number(value) : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setSuccess("");

    try {
      const { name, ...config } = form;

      const created = await experimentApi.create(
        name.trim(),
        config
      );

      setExperiments((previous) => [created, ...previous]);
      setSuccess(`Experiment #${created.id} created successfully.`);

      setForm((previous) => ({
        ...previous,
        name: "",
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function handleStartTraining(experiment) {
    setStartingId(experiment.id);
    setError("");
    setSuccess("");

    try {
      await experimentApi.start(experiment.id);

      setExperiments((previous) =>
        previous.map((item) =>
          item.id === experiment.id
            ? { ...item, status: "running" }
            : item
        )
      );

      selectedIdRef.current = experiment.id;
      setSelectedExperimentId(experiment.id);
      setSelectedResults(null);
      setSuccess(
        `Training started for experiment #${experiment.id}.`
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setStartingId(null);
    }
  }

  async function handleViewResults(experimentId) {
    selectedIdRef.current = experimentId;
    setSelectedExperimentId(experimentId);
    setError("");
    setResultsLoading(true);
    try {
      const results = await experimentApi.getResults(experimentId);
      if (selectedIdRef.current === experimentId) setSelectedResults(results);
    } catch (err) {
      setError(err.message);
    } finally {
      setResultsLoading(false);
    }
  }

  const hasRunning = experiments.some((experiment) => experiment.status === "running");

  useEffect(() => {
    if (!hasRunning) return;
    let active = true;
    let busy = false;
    const refresh = async () => {
      if (busy) return;
      busy = true;
      try {
        const data = await experimentApi.getAll();
        if (!active) return;
        setExperiments(data.experiments);
        const id = selectedIdRef.current;
        if (id !== null) {
          const results = await experimentApi.getResults(id);
          if (active && selectedIdRef.current === id) setSelectedResults(results);
        }
      } catch (err) {
        if (active) console.error("Live training refresh failed:", err);
      } finally {
        busy = false;
      }
    };
    const interval = setInterval(refresh, 5000);
    return () => { active = false; clearInterval(interval); };
  }, [hasRunning]);

  async function handleCompare(event) {
    event.preventDefault();
    setComparisonError("");
    setComparison(null);
    if (!compareA || !compareB || compareA === compareB) {
      setComparisonError("Select two different completed experiments.");
      return;
    }
    setComparisonLoading(true);
    try {
      const [left, right] = await Promise.all([
        experimentApi.getResults(Number(compareA)),
        experimentApi.getResults(Number(compareB)),
      ]);
      if (left.status !== "completed" || right.status !== "completed") {
        throw new Error("Both experiments must be completed before comparison.");
      }
      setComparison({ left, right });
    } catch (err) {
      setComparisonError(err.message);
    } finally {
      setComparisonLoading(false);
    }
  }

  const completedExperiments = experiments.filter((item) => item.status === "completed");

  const selectedExperiment = experiments.find((experiment) => experiment.id === selectedExperimentId);
  const totalRounds = selectedExperiment?.configuration?.num_rounds || 0;
  const completedRounds = selectedResults?.history?.length || 0;
  const progress = totalRounds ? Math.min(100, (completedRounds / totalRounds) * 100) : 0;
  const latestRound = selectedResults?.history?.at(-1);

  function exportSelected(kind) {
    if (!selectedResults) return;
    const base = `experiment_${selectedResults.experiment_id}_${safeFilename(selectedResults.name)}`;
    if (kind === "rounds") downloadText(`${base}_rounds.csv`, experimentRoundsCsv(selectedResults), "text/csv;charset=utf-8");
    if (kind === "clients") downloadText(`${base}_clients.csv`, experimentClientsCsv(selectedResults), "text/csv;charset=utf-8");
    if (kind === "json") downloadText(`${base}.json`, JSON.stringify(selectedResults, null, 2), "application/json;charset=utf-8");
    if (kind === "summary") downloadText(`${base}_summary.txt`, experimentSummary(selectedResults));
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <h1>FedLearn</h1>
          <p>Federated Learning Experiment Dashboard</p>
        </div>
        <span className="dashboard-tag">Research Platform</span>
      </header>

      <section className="stats-grid">
        <div className="stat-card">
          <span>Supported Datasets</span>
          <strong>{datasets.length}</strong>
        </div>

        <div className="stat-card">
          <span>Saved Experiments</span>
          <strong>{experiments.length}</strong>
        </div>

        <div className="stat-card">
          <span>Aggregation Methods</span>
          <strong>2</strong>
        </div>
      </section>

      {error && (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      )}

      {success && (
        <div className="alert alert-success" role="status">
          {success}
        </div>
      )}

      <div className="dashboard-grid">
        <section className="panel">
          <h2>Create Experiment</h2>
          <p className="panel-description">
            Configure a new federated learning experiment.
          </p>

          <form onSubmit={handleSubmit} className="experiment-form">
            <label>
              Experiment Name
              <input
                name="name"
                value={form.name}
                onChange={updateField}
                placeholder="e.g. Pneumonia FedAvg Baseline"
                minLength={3}
                maxLength={150}
                required
              />
            </label>

            <label>
              Dataset
              <select
                name="dataset"
                value={form.dataset}
                onChange={updateField}
                disabled={loading || datasets.length === 0}
              >
                {datasets.map((dataset) => (
                  <option key={dataset.name} value={dataset.name}>
                    {dataset.display_name}
                  </option>
                ))}
              </select>
            </label>

            <div className="form-row">
              <label>
                Number of Clients
                <input
                  type="number"
                  name="num_clients"
                  min="1"
                  step="1"
                  value={form.num_clients}
                  onChange={updateField}
                  required
                />
              </label>

              <label>
                Training Rounds
                <input
                  type="number"
                  name="num_rounds"
                  min="1"
                  step="1"
                  value={form.num_rounds}
                  onChange={updateField}
                  required
                />
              </label>
            </div>

            <div className="form-row">
              <label>
                Local Epochs
                <input
                  type="number"
                  name="local_epochs"
                  min="1"
                  step="1"
                  value={form.local_epochs}
                  onChange={updateField}
                  required
                />
              </label>

              <label>
                Batch Size
                <input
                  type="number"
                  name="batch_size"
                  min="1"
                  step="1"
                  value={form.batch_size}
                  onChange={updateField}
                  required
                />
              </label>
            </div>

            <label>
              Learning Rate
              <input
                type="number"
                name="learning_rate"
                min="0.000001"
                step="any"
                value={form.learning_rate}
                onChange={updateField}
                required
              />
            </label>

            <div className="form-row">
              <label>
                Aggregation Method
                <select
                  name="aggregation_method"
                  value={form.aggregation_method}
                  onChange={updateField}
                >
                  <option value="fedavg">FedAvg</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </label>

              <label>
                Data Partition
                <select
                  name="partition_type"
                  value={form.partition_type}
                  onChange={updateField}
                >
                  <option value="iid">IID</option>
                  <option value="dirichlet">Non-IID (Dirichlet)</option>
                </select>
              </label>
            </div>

            {form.partition_type === "dirichlet" && (
              <label>
                Dirichlet Alpha
                <input
                  type="number"
                  name="dirichlet_alpha"
                  min="0.001"
                  step="any"
                  value={form.dirichlet_alpha}
                  onChange={updateField}
                  required
                />
              </label>
            )}

            <button
              className="primary-button"
              type="submit"
              disabled={
                submitting ||
                loading ||
                datasets.length === 0
              }
            >
              {submitting ? "Creating..." : "Create Experiment"}
            </button>

            <p className="form-note">
              Creating an experiment saves its configuration.
              For now, dashboard training supports IID with FedAvg only.
            </p>
          </form>
        </section>

        <section className="panel">
          <h2>Saved Experiments</h2>
          <p className="panel-description">
            Previously configured experiments.
          </p>

          {loading ? (
            <p>Loading experiments...</p>
          ) : experiments.length === 0 ? (
            <p className="empty-state">
              No experiments yet. Create your first experiment.
            </p>
          ) : (
            <div className="experiment-list">
              {experiments.map((experiment) => (
                <article
                  className="experiment-card"
                  key={experiment.id}
                >
                  <div className="experiment-card-header">
                    <h3>{experiment.name}</h3>
                    <span className="status-tag">
                      {experiment.status}
                    </span>
                  </div>

                  <p>
                    Dataset:{" "}
                    <strong>
                      {experiment.configuration.dataset}
                    </strong>
                  </p>

                  <p>
                    Clients:{" "}
                    {experiment.configuration.num_clients}
                    {" Ã‚Â· "}
                    Rounds:{" "}
                    {experiment.configuration.num_rounds}
                  </p>

                  <p>
                    Aggregation:{" "}
                    {experiment.configuration.aggregation_method}
                  </p>

                  <small>
                    Experiment #{experiment.id}
                  </small>
                  <div className="experiment-actions">
                    {experiment.status === "created" && (
                      <button
                        type="button"
                        className="primary-button"
                        disabled={
                          startingId === experiment.id ||
                          experiment.configuration.partition_type !== "iid" ||
                          experiment.configuration.aggregation_method !== "fedavg"
                        }
                        onClick={() => handleStartTraining(experiment)}
                      >
                        {startingId === experiment.id
                          ? "Starting..."
                          : "Start Training"}
                      </button>
                    )}

                    {experiment.status === "running" && (
                      <button type="button" className="primary-button" onClick={() => handleViewResults(experiment.id)}>
                         {selectedExperimentId === experiment.id ? "Monitoring Training" : "Monitor Live"}
                       </button>
                    )}

                    {experiment.status === "completed" && (
                      <button
                        type="button"
                        className="primary-button"
                        onClick={() => handleViewResults(experiment.id)}
                      >
                        View Results
                      </button>
                    )}

                    {experiment.status === "failed" && (
                      <p>Training failed.</p>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>

      <section className="panel comparison-panel">
        <h2>Experiment Comparison</h2>
        <p className="panel-description">Select two completed experiments to compare final performance, learning curves, and configurations.</p>
        <form className="comparison-form" onSubmit={handleCompare}>
          <label>Experiment A
            <select value={compareA} onChange={(event) => { setCompareA(event.target.value); setComparison(null); }} required>
              <option value="">Choose experiment A</option>
              {completedExperiments.map((item) => <option key={item.id} value={item.id}>#{item.id} â€” {item.name}</option>)}
            </select>
          </label>
          <label>Experiment B
            <select value={compareB} onChange={(event) => { setCompareB(event.target.value); setComparison(null); }} required>
              <option value="">Choose experiment B</option>
              {completedExperiments.map((item) => <option key={item.id} value={item.id}>#{item.id} â€” {item.name}</option>)}
            </select>
          </label>
          <button className="primary-button" type="submit" disabled={comparisonLoading || completedExperiments.length < 2}>
            {comparisonLoading ? "Comparing..." : "Compare Experiments"}
          </button>
        </form>
        {completedExperiments.length < 2 && <p className="form-note">Complete at least two experiments to enable comparison.</p>}
        {comparisonError && <p className="alert alert-error" role="alert">{comparisonError}</p>}
        {comparison && <ExperimentComparison left={comparison.left} right={comparison.right} />}
      </section>

      {selectedExperimentId !== null && (
        <section className="panel live-monitor">
          <div className="monitor-heading">
            <div>
              <h2>Live Training Monitor</h2>
              <p className="panel-description">{selectedExperiment?.name || `Experiment #${selectedExperimentId}`}</p>
            </div>
            <span className="status-tag">{selectedExperiment?.status || selectedResults?.status || "loading"}</span>
          </div>
          <div className="monitor-progress-label">
            <strong>Round {completedRounds} of {totalRounds || "?"}</strong>
            <strong>{Math.round(progress)}%</strong>
          </div>
          <div className="monitor-progress" role="progressbar" aria-label="Training progress" aria-valuenow={completedRounds} aria-valuemin={0} aria-valuemax={totalRounds || 1}>
            <div style={{ width: `${progress}%` }} />
          </div>
          <div className="monitor-metrics">
            <div><span>Latest Accuracy</span><strong>{latestRound ? `${(latestRound.global_accuracy * 100).toFixed(2)}%` : "â€”"}</strong></div>
            <div><span>Latest Loss</span><strong>{latestRound?.global_loss?.toFixed(4) ?? "â€”"}</strong></div>
            <div><span>Latest Macro F1</span><strong>{latestRound?.global_macro_f1?.toFixed(4) ?? "â€”"}</strong></div>
          </div>
          {resultsLoading && <p>Loading results...</p>}
          {selectedExperiment?.status === "running" && <p className="form-note">Updating every 5 seconds as rounds finish.</p>}
          {selectedExperiment?.status === "failed" && <p className="alert alert-error">Training failed. Previously saved rounds remain visible below.</p>}
          {selectedResults && (
            <>
              <div className="export-section">
                <h3>Export Experiment Results</h3>
                <p className="form-note">Downloads use the latest saved metrics for this experiment.</p>
                <div className="export-actions">
                  <button type="button" className="export-button" onClick={() => exportSelected("rounds")}>Rounds CSV</button>
                  <button type="button" className="export-button" onClick={() => exportSelected("clients")}>Client Metrics CSV</button>
                  <button type="button" className="export-button" onClick={() => exportSelected("json")}>Full JSON</button>
                  <button type="button" className="export-button" onClick={() => exportSelected("summary")}>Summary TXT</button>
                </div>
              </div>
              <div className="monitor-charts">
                <MetricChart history={selectedResults.history || []} field="global_accuracy" title="Global Accuracy" percent />
                <MetricChart history={selectedResults.history || []} field="global_loss" title="Global Loss" />
                <MetricChart history={selectedResults.history || []} field="global_macro_f1" title="Macro F1" />
              </div>
              <h3>Round-by-round Results</h3>
              <div className="results-scroll">
                <table className="results-table">
                  <thead><tr><th>Round</th><th>Accuracy</th><th>Loss</th><th>Macro F1</th><th>Time (s)</th><th>Clients</th></tr></thead>
                  <tbody>
                    {(selectedResults.history || []).map((round) => (
                      <tr key={round.round_number}>
                        <td>{round.round_number}</td>
                        <td>{round.global_accuracy?.toFixed(4) ?? "â€”"}</td>
                        <td>{round.global_loss?.toFixed(4) ?? "â€”"}</td>
                        <td>{round.global_macro_f1?.toFixed(4) ?? "â€”"}</td>
                        <td>{round.round_time_seconds?.toFixed(2) ?? "â€”"}</td>
                        <td>{round.clients?.length ?? "â€”"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}
