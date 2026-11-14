/**
 * NEXA AI Core Engine - Top Bar Header
 * Enforces strict 3-zone contract:
 * [Brand title, one line] — [4-5 nav links, single-line] — [1 primary action]
 */

import React from 'react';
import { RefreshCw, Activity } from 'lucide-react';

export type ActiveTab = 'overview' | 'models' | 'playground' | 'keys' | 'tests' | 'docs';

interface HeaderProps {
  activeTab: ActiveTab;
  onTabChange: (tab: ActiveTab) => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  engineOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  onRefresh,
  isRefreshing,
  engineOnline
}) => {
  return (
    <header className="flex items-center justify-between gap-8 px-6 py-3.5 border-b border-neutral-800 bg-neutral-900/90 backdrop-blur-md sticky top-0 z-50">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-3 shrink-0">
        <a
          href="#"
          onClick={(e) => { e.preventDefault(); onTabChange('overview'); }}
          className="text-base font-semibold tracking-wider text-white uppercase whitespace-nowrap shrink-0 hover:text-neutral-200 transition-colors"
        >
          NEXA AI CORE
        </a>
        <span className="text-xs text-neutral-500 font-mono">v1.0.0</span>
      </div>

      {/* Zone 2: Concise single-line navigation links */}
      <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-neutral-400">
        <button
          onClick={() => onTabChange('overview')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 ${
            activeTab === 'overview' ? 'text-white font-semibold underline underline-offset-8 decoration-neutral-400' : ''
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => onTabChange('models')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 ${
            activeTab === 'models' ? 'text-white font-semibold underline underline-offset-8 decoration-neutral-400' : ''
          }`}
        >
          Models
        </button>
        <button
          onClick={() => onTabChange('playground')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 ${
            activeTab === 'playground' ? 'text-white font-semibold underline underline-offset-8 decoration-neutral-400' : ''
          }`}
        >
          Playground
        </button>
        <button
          onClick={() => onTabChange('keys')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 ${
            activeTab === 'keys' ? 'text-white font-semibold underline underline-offset-8 decoration-neutral-400' : ''
          }`}
        >
          API Keys
        </button>
        <button
          onClick={() => onTabChange('tests')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 ${
            activeTab === 'tests' ? 'text-white font-semibold underline underline-offset-8 decoration-neutral-400' : ''
          }`}
        >
          Test Runner
        </button>
        <button
          onClick={() => onTabChange('docs')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 ${
            activeTab === 'docs' ? 'text-white font-semibold underline underline-offset-8 decoration-neutral-400' : ''
          }`}
        >
          Documentation
        </button>
      </nav>

      {/* Zone 3: 1 primary action */}
      <div className="flex items-center gap-3 shrink-0">
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-neutral-200 bg-neutral-800 border border-neutral-700 rounded-md hover:bg-neutral-700 hover:text-white transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>
    </header>
  );
};
