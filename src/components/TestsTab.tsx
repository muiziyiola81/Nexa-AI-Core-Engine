/**
 * NEXA AI Core Engine - Automated Test Suite Runner
 * Executes backend verification tests against API compliance, security, and state machines.
 */

import React, { useState } from 'react';
import { TestSuiteReport } from '../types';
import { Play, CheckCircle2, XCircle, RefreshCw, Terminal, Check } from 'lucide-react';

export const TestsTab: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [report, setReport] = useState<TestSuiteReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleRunTests = async () => {
    setIsRunning(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/v1/tests/run', { method: 'POST' });
      if (!res.ok) throw new Error(`Test execution failed with HTTP ${res.status}`);
      const data = await res.json();
      setReport(data);
    } catch (err: unknown) {
      setErrorMsg((err as Error).message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">Automated Engine Test Suite</h1>
          <div className="flex items-center gap-2 mt-1 text-xs text-neutral-400">
            <span>Regression Verification</span>
            <span aria-hidden="true">·</span>
            <span>OpenAI API Spec</span>
            <span aria-hidden="true">·</span>
            <span>State Machine Invariants</span>
          </div>
        </div>

        <button
          onClick={handleRunTests}
          disabled={isRunning}
          className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-neutral-900 bg-white hover:bg-neutral-200 rounded-md transition-colors disabled:opacity-50"
        >
          <Play className={`w-3.5 h-3.5 fill-current ${isRunning ? 'animate-spin' : ''}`} />
          <span>{isRunning ? 'Running Test Suite...' : 'Execute Automated Tests'}</span>
        </button>
      </div>

      {errorMsg && (
        <div className="p-4 bg-red-950/50 border border-red-800 rounded-lg text-xs text-red-300">
          <span className="font-semibold text-red-200">Execution Error: </span>
          {errorMsg}
        </div>
      )}

      {/* Test Report Summary */}
      {report && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
            <div className="text-xs text-neutral-400">Total Assertions</div>
            <div className="mt-1 text-xl font-medium font-mono tabular-nums text-white">{report.total}</div>
          </div>
          <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
            <div className="text-xs text-neutral-400">Passed</div>
            <div className="mt-1 text-xl font-medium font-mono tabular-nums text-emerald-400">{report.passed}</div>
          </div>
          <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
            <div className="text-xs text-neutral-400">Failed</div>
            <div className="mt-1 text-xl font-medium font-mono tabular-nums text-red-400">{report.failed}</div>
          </div>
          <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
            <div className="text-xs text-neutral-400">Duration</div>
            <div className="mt-1 text-xl font-medium font-mono tabular-nums text-white">{report.durationMs} ms</div>
          </div>
        </div>
      )}

      {/* Test Cases List */}
      <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
          <h2 className="text-sm font-semibold text-white">Test Scenarios</h2>
          <span className="text-xs font-mono text-neutral-500">
            {report ? `Executed at ${new Date(report.timestamp).toLocaleTimeString()}` : 'Awaiting manual trigger'}
          </span>
        </div>

        {!report && !isRunning ? (
          <div className="py-12 text-center text-xs text-neutral-400 space-y-3">
            <Terminal className="w-8 h-8 text-neutral-600 mx-auto" />
            <p className="font-medium text-neutral-300">Ready to execute tests</p>
            <p className="max-w-md mx-auto text-neutral-500">
              Click "Execute Automated Tests" to run all backend validation suites in real time, or run <code className="text-neutral-300 font-mono">npm test</code> in your terminal.
            </p>
          </div>
        ) : isRunning ? (
          <div className="py-12 text-center text-xs text-neutral-400 space-y-3">
            <RefreshCw className="w-8 h-8 text-neutral-400 animate-spin mx-auto" />
            <p className="font-medium text-neutral-200">Executing test matrix...</p>
            <p className="text-neutral-500">Evaluating API endpoints, schema validators, rate limits, and model state transitions.</p>
          </div>
        ) : (
          <div className="divide-y divide-neutral-800">
            {report?.results.map((r, i) => (
              <div key={i} className="py-3 flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    {r.passed ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-400 shrink-0" />
                    )}
                    <span className="text-xs font-medium text-white">{r.name}</span>
                  </div>
                  <div className="flex items-center gap-2 text-[11px] text-neutral-500 font-mono ml-6">
                    <span className="uppercase">{r.category}</span>
                    <span aria-hidden="true">·</span>
                    <span>{r.details || r.error}</span>
                  </div>
                </div>

                <div className="font-mono text-xs text-neutral-400 shrink-0 tabular-nums">
                  {r.durationMs} ms
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
