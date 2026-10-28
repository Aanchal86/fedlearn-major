
// import { useState } from "react";
// import { datasetApi, experimentApi } from "./services/api";

// function App() {
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState("");

//   async function runRequest(request) {
//     setLoading(true);
//     setError("");
//     setResult(null);

//     try {
//       const data = await request();
//       setResult(data);
//     } catch (err) {
//       setError(err.message);
//     } finally {
//       setLoading(false);
//     }
//   }

//   return (
//     <main style={{ padding: "40px", fontFamily: "Arial" }}>
//       <h1>FedLearn API Connection Test</h1>

//       <p>Test communication between React and FastAPI.</p>

//       <div style={{ display: "flex", gap: "12px" }}>
//         <button
//           onClick={() => runRequest(datasetApi.getAll)}
//           disabled={loading}
//         >
//           Load Datasets
//         </button>

//         <button
//           onClick={() => runRequest(experimentApi.getAll)}
//           disabled={loading}
//         >
//           Load Experiments
//         </button>
//       </div>

//       {loading && <p>Loading...</p>}

//       {error && (
//         <p style={{ color: "red" }}>
//           Error: {error}
//         </p>
//       )}

//       {result && (
//         <pre
//           style={{
//             marginTop: "20px",
//             padding: "16px",
//             background: "#f3f4f6",
//             color: "#111827",
//             borderRadius: "8px",
//             overflowX: "auto",
//           }}
//         >
//           {JSON.stringify(result, null, 2)}
//         </pre>
//       )}
//     </main>
//   );
// }

// export default App;


import ExperimentDashboard from "./components/ExperimentDashboard";

export default function App() {
  return <ExperimentDashboard />;
}
