import Link from "next/link";

export default function ReflectionPage() {
  return (
    <div className="flow">
      <section className="card section">
        <div className="prose">
          <h1>Reflection: AI, empathy, and R+ training</h1>
          <p>
            This project started as a simple RL question (can an agent learn without punishment?) and
            turned into a cross‑disciplinary reflection on how we teach learners—machines, dogs, and
            people.
          </p>
        </div>
        <div className="kpi">
          <Link className="pill pill--accent" href="/compare">
            Compare variants
          </Link>
          <a className="pill" href="/reflection.pdf">
            Full PDF (download)
          </a>
        </div>
      </section>

      <section className="card section">
        <div className="prose">
          <h2>Wait—what does a robot have to do with dog training?</h2>
          <p>
            Reinforcement learning and positive reinforcement dog training share a core idea:
            behavior changes through consequences. In modern R+ training, we avoid punishment and
            reinforce the moments the learner gets it right—clear feedback, consistent rewards, and a
            focus on progress.
          </p>
          <p>
            Cliff Walking is a perfect sandbox for this analogy. The traditional setup uses a harsh
            cliff penalty and a step penalty; the positive‑only setup rewards moving closer to the goal
            and simply resets after “mistakes.”
          </p>
        </div>
      </section>

      <section className="card section">
        <div className="prose">
          <h2>What changed with positive‑only rewards?</h2>
          <ul>
            <li>
              The value landscape becomes a smooth gradient toward the goal instead of sharp “danger
              zones.”
            </li>
            <li>Exploration stays safer and more curious because there’s no punitive cliff signal.</li>
            <li>
              Learning can be slower—but the “emotional environment” (the reward landscape) is calmer.
            </li>
          </ul>
          <p>
            The takeaway isn’t that punishment never “works,” but that you don’t need it to teach—and
            the trade‑offs matter: stress, hesitation, trust, and exploration.
          </p>
        </div>
      </section>

      <section className="card section">
        <div className="prose">
          <h2>Read the full reflection</h2>
          <p>
            The full write‑up includes the operant conditioning framing, the reward shaping logic, and
            a deeper comparison between fear‑driven avoidance and confidence‑building learning.
          </p>
        </div>
        <div className="empty">
          If the embedded PDF doesn’t load in your browser, use the download link above.
        </div>
        <div style={{ marginTop: 14 }}>
          <object data="/reflection.pdf" type="application/pdf" width="100%" height="820">
            <a className="pill pill--accent" href="/reflection.pdf">
              Open reflection.pdf
            </a>
          </object>
        </div>
      </section>
    </div>
  );
}
