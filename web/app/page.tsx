import Board from "@/components/Board";
import HealthIndicator from "@/components/HealthIndicator";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 gap-6">
      <header className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight">chess_ml</h1>
        <p className="text-sm text-slate-400 mt-1">
          Slice 0 scaffold — interactive board only. No engine, no analysis yet.
        </p>
      </header>

      <div className="w-[480px] max-w-full">
        <Board />
      </div>

      <HealthIndicator />
    </main>
  );
}
