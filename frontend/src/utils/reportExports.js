// Day 24: pure report generation helpers. No backend changes required.
const csvCell = (value) => {
  const text = value == null ? "" : String(value);
  return `"${text.replace(/"/g, '""')}"`;
};
export const csvTable = (headers, rows) => [headers, ...rows]
  .map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";

export const safeFilename = (name) => String(name || "experiment")
  .replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "experiment";

export function experimentRoundsCsv(result) {
  const headers = ["experiment_id", "experiment_name", "status", "round", "global_accuracy", "global_loss", "global_macro_f1", "round_time_seconds", "client_count"];
  return csvTable(headers, (result.history || []).map((round) => [
    result.experiment_id, result.name, result.status, round.round_number,
    round.global_accuracy, round.global_loss, round.global_macro_f1,
    round.round_time_seconds, round.clients?.length ?? 0,
  ]));
}

export function experimentClientsCsv(result) {
  const headers = ["experiment_id", "round", "client_id", "num_samples", "aggregation_weight", "training_loss", "training_accuracy"];
  return csvTable(headers, (result.history || []).flatMap((round) =>
    (round.clients || []).map((client) => [result.experiment_id, round.round_number,
      client.client_id, client.num_samples, client.aggregation_weight,
      client.training_loss, client.training_accuracy])));
}

const lastRound = (result) => result.history?.at(-1) || {};
export function comparisonCsv(left, right) {
  const metrics = [
    ["Accuracy", "global_accuracy"], ["Loss", "global_loss"],
    ["Macro F1", "global_macro_f1"],
  ];
  return csvTable(["metric", "experiment_A", "experiment_B", "difference_B_minus_A"],
    metrics.map(([label, key]) => {
      const a = lastRound(left)[key];
      const b = lastRound(right)[key];
      return [label, a, b, Number.isFinite(a) && Number.isFinite(b) ? b - a : ""];
    }));
}
export function comparisonRoundsCsv(left, right) {
  const rounds = new Map();
  for (const [prefix, result] of [["A", left], ["B", right]]) {
    for (const round of result.history || []) {
      const row = rounds.get(round.round_number) || { round: round.round_number };
      row[`${prefix}_accuracy`] = round.global_accuracy;
      row[`${prefix}_loss`] = round.global_loss;
      row[`${prefix}_macro_f1`] = round.global_macro_f1;
      rounds.set(round.round_number, row);
    }
  }
  return csvTable(["round", "A_accuracy", "B_accuracy", "A_loss", "B_loss", "A_macro_f1", "B_macro_f1"],
    [...rounds.values()].sort((a,b) => a.round - b.round).map((r) =>
      [r.round, r.A_accuracy, r.B_accuracy, r.A_loss, r.B_loss, r.A_macro_f1, r.B_macro_f1]));
}
export function experimentSummary(result) {
  const rounds = result.history || [];
  const last = lastRound(result);
  const config = result.configuration || {};
  const fmt = (v) => Number.isFinite(v) ? v.toFixed(4) : "N/A";
  const totalSeconds = rounds.reduce((sum, r) => sum + (Number(r.round_time_seconds) || 0), 0);
  return [
    `FedLearn Experiment Summary`, `===========================`,
    `Experiment: ${result.name || "Unnamed"} (#${result.experiment_id})`,
    `Status: ${result.status || "unknown"}`, `Dataset: ${config.dataset || "N/A"}`,
    `Clients: ${config.num_clients ?? "N/A"}`,
    `Rounds completed: ${rounds.length} / ${config.num_rounds ?? "N/A"}`,
    `Local epochs: ${config.local_epochs ?? "N/A"}`,
    `Batch size: ${config.batch_size ?? "N/A"}`,
    `Learning rate: ${config.learning_rate ?? "N/A"}`,
    `Aggregation: ${config.aggregation_method || "N/A"}`,
    `Partition: ${config.partition_type || "N/A"}`,
    `Final accuracy: ${fmt(last.global_accuracy)}`,
    `Final loss: ${fmt(last.global_loss)}`,
    `Final macro F1: ${fmt(last.global_macro_f1)}`,
    `Sum of recorded round times: ${totalSeconds.toFixed(2)} seconds`,
    `Note: Values reflect saved experiment results; timing excludes unrecorded overhead.`,
    "",
  ].join("\n");
}

export function downloadText(filename, contents, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
