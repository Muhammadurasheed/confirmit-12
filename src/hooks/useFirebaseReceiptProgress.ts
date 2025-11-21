/**
 * Firebase Real-time Progress Listener
 * Listens to Firestore for AI agent progress updates
 * This is the CORRECT way to get real-time updates from AI agents
 */
import { useEffect, useRef, useState } from 'react';
import { db } from '@/lib/firebase';
import { doc, onSnapshot } from 'firebase/firestore';

interface AgentProgress {
  agent: string;
  stage: string;
  message: string;
  progress: number;
  timestamp: string;
  details?: Record<string, any>;
}

interface UseFirebaseReceiptProgressOptions {
  receiptId?: string;
  onProgress?: (data: AgentProgress) => void;
  onComplete?: (data: any) => void;
  onError?: (error: any) => void;
}

export const useFirebaseReceiptProgress = ({
  receiptId,
  onProgress,
  onComplete,
  onError,
}: UseFirebaseReceiptProgressOptions) => {
  const [isListening, setIsListening] = useState(false);
  const callbacksRef = useRef({ onProgress, onComplete, onError });
  const unsubscribeRef = useRef<(() => void) | null>(null);

  // Update callbacks ref when they change
  useEffect(() => {
    callbacksRef.current = { onProgress, onComplete, onError };
  }, [onProgress, onComplete, onError]);

  useEffect(() => {
    if (!receiptId || !db) {
      console.log('⚠️ No receipt ID or Firebase not initialized');
      return;
    }

    console.log(`🔥 Starting Firebase listener for receipt: ${receiptId}`);
    setIsListening(true);

    const receiptRef = doc(db, 'receipts', receiptId);

    // Real-time listener
    const unsubscribe = onSnapshot(
      receiptRef,
      (docSnapshot) => {
        if (docSnapshot.exists()) {
          const data = docSnapshot.data();
          console.log('🔥 Firebase update received:', data);

          // Check for progress updates (now using flat structure)
          if (data.progress_agent && data.progress_stage) {
            const progress: AgentProgress = {
              agent: data.progress_agent,
              stage: data.progress_stage,
              message: data.progress_message || '',
              progress: data.progress_percentage || 0,
              timestamp: data.progress_timestamp || new Date().toISOString(),
              details: {}
            };
            
            // Extract detail fields (progress_detail_*)
            Object.keys(data).forEach(key => {
              if (key.startsWith('progress_detail_')) {
                const detailKey = key.replace('progress_detail_', '');
                progress.details![detailKey] = data[key];
              }
            });
            
            console.log(`📊 [${progress.agent}] ${progress.message} (${progress.progress}%)`);
            callbacksRef.current.onProgress?.(progress);
          }

          // Check for completion
          if (data.status === 'completed' && data.analysis) {
            console.log('✅ Analysis completed:', data.analysis);
            callbacksRef.current.onComplete?.(data.analysis);
          }

          // Check for errors
          if (data.status === 'failed' && data.error) {
            console.error('❌ Analysis failed:', data.error);
            callbacksRef.current.onError?.(data.error);
          }
        } else {
          console.log('⚠️ Receipt document does not exist yet');
        }
      },
      (error) => {
        console.error('❌ Firebase listener error:', error);
        setIsListening(false);
        callbacksRef.current.onError?.(error);
      }
    );

    unsubscribeRef.current = unsubscribe;

    // Cleanup on unmount or receipt change
    return () => {
      console.log(`🔥 Stopping Firebase listener for receipt: ${receiptId}`);
      unsubscribe();
      unsubscribeRef.current = null;
      setIsListening(false);
    };
  }, [receiptId]);

  const stopListening = () => {
    if (unsubscribeRef.current) {
      console.log('🔥 Manually stopping Firebase listener');
      unsubscribeRef.current();
      unsubscribeRef.current = null;
      setIsListening(false);
    }
  };

  return { isListening, stopListening };
};
