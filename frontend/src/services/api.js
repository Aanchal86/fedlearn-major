
const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000"
).replace(/\/$/, "");

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  let data;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const detail = data?.detail;

    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map((error) => error.msg || "Invalid input")
              .join("; ")
          : `Request failed with status ${response.status}`;

    throw new Error(message);
  }

  return data;
}

export const datasetApi = {
  getAll() {
    return apiRequest("/api/datasets");
  },

  getByName(datasetName) {
    return apiRequest(
      `/api/datasets/${encodeURIComponent(datasetName)}`
    );
  },
};

export const experimentApi = {
  getAll() {
    return apiRequest("/api/experiments");
  },

  getById(experimentId) {
    return apiRequest(
      `/api/experiments/${encodeURIComponent(experimentId)}`
    );
  },

  create(name, config) {
    return apiRequest("/api/experiments", {
      method: "POST",
      body: JSON.stringify({
        name,
        config,
      }),
    });
  },
};
