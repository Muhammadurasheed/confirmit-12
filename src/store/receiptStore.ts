import { create } from 'zustand';
import type { Receipt, AnalysisResult } from '@/types';

interface ReceiptState {
  // Current receipt being analyzed
  currentReceipt: Receipt | null;
  
  // Analysis progress tracking
  analysisProgress: number;
  analysisStatus: string;
  analysisMessage?: string; // Detailed message from agents
  isAnalyzing: boolean;
  
  // Results
  results: AnalysisResult | null;
  
  // Receipt history
  history: Receipt[];
  
  // Actions
  setReceipt: (receipt: Receipt) => void;
  updateProgress: (progress: number, status: string, message?: string) => void;
  setResults: (results: AnalysisResult) => void;
  startAnalysis: () => void;
  completeAnalysis: () => void;
  failAnalysis: (error: string) => void;
  addToHistory: (receipt: Receipt) => void;
  reset: () => void;
}

export const useReceiptStore = create<ReceiptState>((set) => ({
  currentReceipt: null,
  analysisProgress: 0,
  analysisStatus: 'idle',
  analysisMessage: undefined,
  isAnalyzing: false,
  results: null,
  history: [],

  setReceipt: (receipt) => 
    set({ currentReceipt: receipt }),

  updateProgress: (progress, status, message) =>
    set({ analysisProgress: progress, analysisStatus: status, analysisMessage: message || status }),

  setResults: (results) =>
    set({ results }),

  startAnalysis: () =>
    set({ isAnalyzing: true, analysisProgress: 0, analysisStatus: 'Starting analysis...', analysisMessage: undefined }),

  completeAnalysis: () =>
    set({ isAnalyzing: false, analysisProgress: 100, analysisStatus: 'Analysis complete' }),

  failAnalysis: (error) =>
    set({ isAnalyzing: false, analysisStatus: `Failed: ${error}` }),

  addToHistory: (receipt) =>
    set((state) => ({ history: [receipt, ...state.history] })),

  reset: () =>
    set({
      currentReceipt: null,
      analysisProgress: 0,
      analysisStatus: 'idle',
      analysisMessage: undefined,
      isAnalyzing: false,
      results: null,
    }),
}));
