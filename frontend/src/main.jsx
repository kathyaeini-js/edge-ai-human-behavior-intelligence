import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [health, setHealth] = useState("checking");
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      const [h, e] = await Promise.all([
        fetch(`${API}/health`),
        fetch(`${API}/events?limit=100`)
      ]);
      if (!h.ok || !e.ok) throw new Error("API request failed");
      const healthData = await h.json();
      const eventData = await e.json();
      setHealth(healthData.status);
      setItems(eventData.items || []);
      setError("");
    } catch (err) {
      setHealth("offline");
      setError("Start the FastAPI server and process a video to see events.");
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 4000);
    return () => clearInterval(timer);
  }, []);

  const tracks = new Set(items.map(x => x.track_id)).size;
  const labels = items.reduce((acc, item) => {
    acc[item.activity] = (acc[item.activity] || 0) + 1;
    return acc;
  }, {});

  return (
    <main className="shell">
      <header>
        <div>
          <p className="eyebrow">EDGE AI · VIDEO ANALYTICS</p>
          <h1>Human Behavior Intelligence</h1>
          <p className="sub">Pose observations and activity events from authorized video.</p>
        </div>
        <span className={`status ${health === "ok" ? "online" : ""}`}>{health}</span>
      </header>

      <section className="notice">
        <strong>Prototype notice</strong>
        <span>These are uncertain pose-derived observations, not employee identity, intent, emotion, or performance scores.</span>
      </section>

      {error && <p className="error">{error}</p>}

      <section className="stats">
        <article><span>Recent events</span><strong>{items.length}</strong></article>
        <article><span>Track IDs seen</span><strong>{tracks}</strong></article>
        <article><span>Activity labels</span><strong>{Object.keys(labels).length}</strong></article>
      </section>

      <section className="panel">
        <div className="panel-head">
          <div><h2>Recent activity events</h2><p>Refreshes every 4 seconds</p></div>
          <button onClick={refresh}>Refresh</button>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Timestamp</th><th>Track</th><th>Observation</th><th>Pose confidence</th><th>Frame</th></tr></thead>
            <tbody>
              {items.map((x, i) => (
                <tr key={`${x.timestamp}-${x.track_id}-${i}`}>
                  <td>{x.timestamp ? new Date(x.timestamp).toLocaleString() : "—"}</td>
                  <td>#{x.track_id}</td>
                  <td><span className="pill">{x.activity}</span></td>
                  <td>{typeof x.mean_keypoint_confidence === "number" ? x.mean_keypoint_confidence.toFixed(2) : "—"}</td>
                  <td>{x.frame ?? "—"}</td>
                </tr>
              ))}
              {!items.length && <tr><td colSpan="5" className="empty">No events yet. Process a video, then refresh.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
      <footer>Local prototype · Keep footage and event logs access-controlled.</footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
