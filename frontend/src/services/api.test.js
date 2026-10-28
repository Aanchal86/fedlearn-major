
import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  datasetApi,
  experimentApi,
} from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Dataset API", () => {
  it("fetches supported datasets", async () => {
    const datasets = [
      { name: "fashion_mnist" },
      { name: "pneumonia_mnist" },
    ];

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => datasets,
    });

    vi.stubGlobal("fetch", mockFetch);

    const result = await datasetApi.getAll();

    expect(result).toEqual(datasets);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/datasets",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      })
    );
  });

  it("fetches dataset details", async () => {
    const dataset = {
      name: "pneumonia_mnist",
      num_classes: 2,
    };

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => dataset,
      })
    );

    const result = await datasetApi.getByName(
      "pneumonia_mnist"
    );

    expect(result.num_classes).toBe(2);
  });
});

describe("Experiment API", () => {
  it("fetches saved experiments", async () => {
    const response = {
      experiments: [],
      total: 0,
    };

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => response,
      })
    );

    const result = await experimentApi.getAll();

    expect(result.total).toBe(0);
    expect(result.experiments).toEqual([]);
  });

  it("creates an experiment", async () => {
    const created = {
      id: 1,
      name: "Fashion Baseline",
      status: "created",
    };

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => created,
    });

    vi.stubGlobal("fetch", mockFetch);

    const result = await experimentApi.create(
      "Fashion Baseline",
      {
        dataset: "fashion_mnist",
        num_clients: 5,
      }
    );

    expect(result.status).toBe("created");

    const requestOptions = mockFetch.mock.calls[0][1];

    expect(requestOptions.method).toBe("POST");

    expect(JSON.parse(requestOptions.body)).toEqual({
      name: "Fashion Baseline",
      config: {
        dataset: "fashion_mnist",
        num_clients: 5,
      },
    });
  });

  it("handles backend validation errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        json: async () => ({
          detail: [
            { msg: "Input should be greater than 0" },
          ],
        }),
      })
    );

    await expect(
      experimentApi.create("Invalid Experiment", {
        num_clients: 0,
      })
    ).rejects.toThrow(
      "Input should be greater than 0"
    );
  });

  it("handles a missing experiment", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: async () => ({
          detail: "Experiment not found",
        }),
      })
    );

    await expect(
      experimentApi.getById(999)
    ).rejects.toThrow("Experiment not found");
  });
});
