/**
 * On-device training data collector.
 *
 * Uses Zustand TrainingDataStore for structured, indexed storage.
 * Collects (user_msg, assistant_msg) conversation pairs, batches
 * them to the server for fine-tuning, pulls updated weights back.
 *
 * Data flow:
 *   chat exchange → collectPair() → TrainingDataStore (indexed)
 *   → batch send to server → server trains → phone pulls weights
 */

import {api} from './api-client';
import {
  useTrainingDataStore,
  hydrateTrainingData,
  type TrainingPair,
} from '../stores/training-data-store';

// Re-export types
export type {TrainingPair};

export interface TrainResult {
  success: boolean;
  checkpoint_name: string;
  loss: number;
  steps: number;
  elapsed_ms: number;
}

export interface CollectorStats {
  total: number;
  pending: number;
  synced: number;
  byQuality: Record<number, number>;
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Initialize the collector.  Call once on app start.
 */
export async function initCollector(): Promise<void> {
  await hydrateTrainingData();
}

/**
 * Record a single training pair from a chat exchange.
 *
 * @param userMsg - The user's message
 * @param assistantMsg - The assistant's response
 * @param sessionId - Current session ID
 * @param quality - Optional quality signal (1/-1/0)
 * @returns The pair ID
 */
export function collectPair(
  userMsg: string,
  assistantMsg: string,
  sessionId: string,
  quality: number = 0,
): string {
  return useTrainingDataStore.getState().addPair(
    userMsg,
    assistantMsg,
    sessionId,
    quality,
  );
}

/**
 * Update quality signal for a recent pair (e.g. thumbs up/down).
 */
export function updateQuality(pairId: string, quality: number): void {
  useTrainingDataStore.getState().updateQuality(pairId, quality);
}

/**
 * Get all pending (unsynced) training pairs.
 */
export function getPendingPairs(): TrainingPair[] {
  return useTrainingDataStore.getState().getPendingPairs();
}

/**
 * Get collector statistics.
 */
export function getStats(): CollectorStats {
  return useTrainingDataStore.getState().getStats();
}

/**
 * Clear all pending pairs (after successful training).
 */
export async function clearPending(): Promise<void> {
  useTrainingDataStore.getState().clearSynced();
}

/**
 * Send pending pairs to the server for training.
 *
 * @param checkpoint - Checkpoint name to fine-tune from
 * @param minPairs - Minimum pairs before triggering training (default: 10)
 * @returns Training result or null if not enough data
 */
export async function triggerTraining(
  checkpoint: string,
  minPairs: number = 10,
): Promise<TrainResult | null> {
  const store = useTrainingDataStore.getState();
  const pending = store.getPendingPairs();

  if (pending.length < minPairs) {
    return null;
  }

  const result = await api.mobileTrain<TrainResult>({
    pairs: pending.map(p => ({
      id: p.id,
      user_msg: p.user_msg,
      assistant_msg: p.assistant_msg,
      quality: p.quality,
      timestamp: p.timestamp,
      session_id: p.session_id,
    })),
    checkpoint,
  });

  if (result.success) {
    store.markSynced(pending.map(p => p.id));
    store.setLastTrainResult({
      loss: result.loss,
      checkpoint: result.checkpoint_name,
      timestamp: Date.now(),
    });
  }

  return result;
}

/**
 * Train from server-side inference logs instead of sending local pairs.
 * The server reads its own session/response-log files and trains directly.
 * No mobile→server data round-trip needed.
 *
 * @param opts - Optional: limit, min_length, model filter
 * @returns Training result
 */
export async function trainFromSessions(
  opts?: {limit?: number; min_length?: number; model?: string},
): Promise<TrainResult> {
  return api.trainFromSessions<TrainResult>(opts);
}

/**
 * Get auto-trainer status from the server.
 */
export async function getAutoTrainStatus(): Promise<{
  enabled: boolean;
  threshold: number;
  interval_s: number;
  pending_conversations: number;
  total_trains: number;
  last_train: string | null;
  last_loss: number;
  last_checkpoint: string;
}> {
  return api.getAutoTrainStatus();
}

/**
 * Pull latest weights from server after training.
 *
 * @param checkpoint - Trained checkpoint name
 * @returns Base64-encoded weights + config, or null if not found
 */
export async function pullWeights(
  checkpoint: string,
): Promise<{config: any; weights_b64: string} | null> {
  try {
    const data = await api.get<{config: any; weights_b64: string}>(
      `/auto-train/checkpoints/${encodeURIComponent(checkpoint)}/export-mobile`,
    );
    return data || null;
  } catch {
    return null;
  }
}
