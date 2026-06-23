export type RunVariant = "positive_only" | "traditional";

export type RunIndexEntry = {
  runId: string;
  variant: RunVariant;
  summaryPath: string;
};

export type RunIndex = {
  schemaVersion: string;
  generatedAt: string;
  runs: RunIndexEntry[];
};

export type RunMediaPaths = {
  returns_png: string;
  steps_png: string;
  maxq_heatmaps_gif: string;
  qvalues_snapshots_json: string;
};

export type RunSummary = {
  schema_version: string;
  variant: RunVariant;
  run_id: string;
  xsize: number;
  ysize: number;
  cliff_positions: [number, number][];
  hyperparameters: Record<string, unknown> & {
    cliff_layout?: string;
    width?: number;
    height?: number;
    run_id?: string;
  };
  metrics: {
    episode_returns?: number[];
    steps_per_episode?: number[];
    trajectories_path?: string;
    [key: string]: unknown;
  };
  media?: RunMediaPaths;
};

export type QValuesSnapshots = {
  schema_version: string;
  xsize: number;
  ysize: number;
  actions: string[];
  frame_episodes: number[];
  min: number;
  max: number;
  qvalues: number[][][][];
};

async function loadJson<T>(path: string): Promise<T> {
  const res = await fetch(`/runs/${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function fetchRunIndex(): Promise<RunIndex> {
  return loadJson<RunIndex>("index.json");
}

export async function fetchSummary(summaryPath: string): Promise<RunSummary> {
  return loadJson<RunSummary>(summaryPath);
}
