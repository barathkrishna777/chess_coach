import { Suspense } from "react";

import TrainClient from "./TrainClient";

export default function TrainPage() {
  return (
    <Suspense
      fallback={
        <main className="min-h-screen bg-[#f6f8fb] px-5 py-6 text-[#17201d]">
          <p className="mx-auto max-w-7xl text-sm text-[#4a5a54]">
            Loading training drills.
          </p>
        </main>
      }
    >
      <TrainClient />
    </Suspense>
  );
}
