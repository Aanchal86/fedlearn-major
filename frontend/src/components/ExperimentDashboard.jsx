import { useEffect, useRef, useState } from "react";
import { datasetApi, experimentApi } from "../services/api";
import "./ExperimentDashboard.css";

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

  useEffect(() => {
    const hasRunning = experiments.some((experiment) => experiment.status === "running");
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
  }, [experiments.some((experiment) => experiment.status === "running")]);

  const selectedExperiment = experiments.find((experiment) => experiment.id === selectedExperimentId);
  const totalRounds = selectedExperiment?.configuration?.num_rounds || 0;
  const completedRounds = selectedResults?.history?.length || 0;
  const progress = totalRounds ? Math.min(100, (completedRounds / totalRounds) * 100) : 0;
  const latestRound = selectedResults?.history?.at(-1);

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
                    {" Â· "}
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
            <div><span>Latest Accuracy</span><strong>{latestRound ? `${(latestRound.global_accuracy * 100).toFixed(2)}%` : "—"}</strong></div>
            <div><span>Latest Loss</span><strong>{latestRound?.global_loss?.toFixed(4) ?? "—"}</strong></div>
            <div><span>Latest Macro F1</span><strong>{latestRound?.global_macro_f1?.toFixed(4) ?? "—"}</strong></div>
          </div>
          {resultsLoading && <p>Loading results...</p>}
          {selectedExperiment?.status === "running" && <p className="form-note">Updating every 5 seconds as rounds finish.</p>}
          {selectedExperiment?.status === "failed" && <p className="alert alert-error">Training failed. Previously saved rounds remain visible below.</p>}
          {selectedResults && (
            <>
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
                        <td>{round.global_accuracy?.toFixed(4) ?? "—"}</td>
                        <td>{round.global_loss?.toFixed(4) ?? "—"}</td>
                        <td>{round.global_macro_f1?.toFixed(4) ?? "—"}</td>
                        <td>{round.round_time_seconds?.toFixed(2) ?? "—"}</td>
                        <td>{round.clients?.length ?? "—"}</td>
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
