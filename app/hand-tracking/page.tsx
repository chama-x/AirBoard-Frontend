"use client";

import dynamic from "next/dynamic";

// Dynamically import the HandTracker component with SSR disabled
// This is necessary because MediaPipe requires browser APIs that are not available during SSR
const HandTracker = dynamic(
  () => import("../../src/components/HandTracker"),
  { ssr: false }
);

export default function HandTrackingPage() {
  return (
    <div className="min-h-screen bg-gray-900">
      <HandTracker />
    </div>
  );
} 