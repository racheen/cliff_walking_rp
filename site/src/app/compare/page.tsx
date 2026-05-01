import { Suspense } from "react";

import { CompareFromSearchParams } from "./CompareFromSearchParams";

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="card">Loading…</div>}>
      <CompareFromSearchParams />
    </Suspense>
  );
}

