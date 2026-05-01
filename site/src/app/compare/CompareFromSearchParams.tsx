"use client";

import { useSearchParams } from "next/navigation";

import { CompareClient } from "./CompareClient";

export function CompareFromSearchParams() {
  const sp = useSearchParams();
  const a = sp.get("a");
  const b = sp.get("b");
  return <CompareClient initialA={a} initialB={b} />;
}

