import Link from "next/link";

export default function HomePage() {
  return (
    <div className="grid">
      <section className="card">
        <div className="prose">
          <h1>Cliff Walking with Positive‑Only Reinforcement</h1>
          <p>
            Can an agent learn safe, effective behavior without punishment—only encouragement? This
            project compares traditional cliff-walking Q‑learning against a positive‑only variant
            inspired by R+ dog training.
          </p>
        </div>
        <p className="small">
          This site is a static viewer for exported training runs (learning curves, heatmap
          animations, and trajectories). Train locally using the CLI, then sync a run bundle into{" "}
          <code>site/public/runs</code> for deployment.
        </p>
        <p className="small">
          Go to <Link href="/runs">Runs</Link> to browse available bundles.
        </p>
        <div className="kpi">
          <Link className="pill pill--accent" href="/compare">
            Compare runs
          </Link>
          <Link className="pill" href="/reflection">
            Read the reflection
          </Link>
        </div>
      </section>

      <section className="card">
        <div className="prose">
          <h2>What’s included</h2>
        </div>
        <ul className="small">
          <li>Positive-only and traditional reward variants</li>
          <li>Episode return and step plots</li>
          <li>Max-Q heatmap animation over training</li>
          <li>Agent trajectory playback data (static JSON)</li>
        </ul>
      </section>
    </div>
  );
}

