
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
              Training will be enabled in a later phase.
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
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
