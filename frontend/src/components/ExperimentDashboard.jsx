
import { useEffect, useState } from "react";
import { datasetApi, experimentApi } from "../services/api";
import "./ExperimentDashboard.css";

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
    setError("");

    try {
      const results = await experimentApi.getResults(experimentId);
      setSelectedResults(results);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    if (!experiments.some((experiment) => experiment.status === "running")) {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const data = await experimentApi.getAll();
        setExperiments(data.experiments);
      } catch (err) {
        console.error("Failed to refresh experiments:", err);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [experiments]);

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
                    {" · "}
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
                      <p>Training in progress...</p>
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

      {selectedResults && (
        <section className="panel" style={{ marginTop: "24px" }}>
          <h2>Training Results</h2>
          <p>Experiment: <strong>{selectedResults.name}</strong></p>
          <p>Status: {selectedResults.status}</p>
          <p>Completed Rounds: {selectedResults.total_rounds_completed}</p>

          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Accuracy</th>
                  <th>Loss</th>
                  <th>Macro F1</th>
                  <th>Time (s)</th>
                </tr>
              </thead>
              <tbody>
                {selectedResults.history.map((round) => (
                  <tr key={round.round_number}>
                    <td>{round.round_number}</td>
                    <td>{round.global_accuracy?.toFixed(4) ?? "—"}</td>
                    <td>{round.global_loss?.toFixed(4) ?? "—"}</td>
                    <td>{round.global_macro_f1?.toFixed(4) ?? "—"}</td>
                    <td>{round.round_time_seconds?.toFixed(2) ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
