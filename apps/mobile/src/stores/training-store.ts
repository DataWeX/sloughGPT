import {create} from 'zustand';
import {
  startTraining,
  stopTraining,
  getTrainingStatus,
  listCheckpoints,
  deleteCheckpoint,
  listDatasets,
  streamTraining,
  type TrainConfig,
  type Checkpoint,
  type Dataset,
} from '../services/training-service';

export type TrainPhase =
  | 'idle'
  | 'configuring'
  | 'TRAINING'
  | 'EVALUATING'
  | 'COMPLETE'
  | 'FAILED';

interface TrainingState {
  phase: TrainPhase;
  running: boolean;
  loss: number | null;
  lossHistory: {step: number; value: number}[];
  epoch: number;
  totalEpochs: number;
  steps: number;
  checkpoint: string | null;
  error: string | null;
  checkpoints: Checkpoint[];
  datasets: Dataset[];
  config: TrainConfig;

  setConfig: (partial: Partial<TrainConfig>) => void;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  refresh: () => Promise<void>;
  loadCheckpoint: (name: string) => Promise<void>;
  deleteCheckpoint: (name: string) => Promise<void>;
  clearError: () => void;
}

let abortController: AbortController | null = null;

const defaultConfig: TrainConfig = {
  epochs: 10,
  learning_rate: 0.001,
  batch_size: 64,
  soul_name: 'assistant',
  algo: 'bpe',
};

export const useTrainingStore = create<TrainingState>((set, get) => ({
  phase: 'idle',
  running: false,
  loss: null,
  lossHistory: [],
  epoch: 0,
  totalEpochs: 10,
  steps: 0,
  checkpoint: null,
  error: null,
  checkpoints: [],
  datasets: [],
  config: {...defaultConfig},

  setConfig: partial => set(s => ({config: {...s.config, ...partial}})),

  start: async () => {
    const state = get();
    const cfg = state.config;

    if (!cfg.source_text && !cfg.dataset_id && !cfg.checkpoint_name) {
      set({error: 'Provide training text, select a dataset, or load a checkpoint'});
      return;
    }

    set({
      phase: 'TRAINING',
      running: true,
      loss: null,
      lossHistory: [],
      epoch: 0,
      totalEpochs: cfg.epochs,
      steps: 0,
      checkpoint: null,
      error: null,
    });

    try {
      await startTraining(cfg);
    } catch (err: any) {
      set({phase: 'FAILED', error: err.message, running: false});
      return;
    }

    abortController = new AbortController();

    try {
      for await (const event of streamTraining(abortController.signal)) {
        const data = event.meta || {};

        if (event.token) {
          try {
            const parsed = JSON.parse(event.token);
            if (parsed.phase) {
              set({phase: parsed.phase as TrainPhase});
            }
            if (parsed.data) {
              const d = parsed.data;
              if (d.loss !== undefined) {
                const step = d.step || get().steps;
                set(s => ({
                  loss: d.loss,
                  steps: step,
                  epoch: d.epoch || s.epoch,
                  lossHistory: [...s.lossHistory, {step, value: d.loss}],
                }));
              }
              if (d.checkpoint) {
                set({checkpoint: d.checkpoint});
              }
            }
          } catch {
            // plain text progress — skip
          }
        }

        if (event.done) {
          if (event.error) {
            set({phase: 'FAILED', error: event.error});
          } else {
            set({phase: 'COMPLETE'});
          }
          break;
        }
      }

      get().refresh();
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        set({phase: 'FAILED', error: err.message});
      }
    } finally {
      set({running: false});
      abortController = null;
    }
  },

  stop: async () => {
    abortController?.abort();
    try {
      await stopTraining();
    } catch {}
    set({phase: 'idle', running: false});
  },

  refresh: async () => {
    try {
      const [ckpts, datasets, status] = await Promise.all([
        listCheckpoints().catch(() => []),
        listDatasets().catch(() => []),
        getTrainingStatus().catch(() => ({running: false, config: {}})),
      ]);
      set({
        checkpoints: ckpts,
        datasets,
        running: status.running,
      });
    } catch {}
  },

  loadCheckpoint: async (name: string) => {
    try {
      set({checkpoint: name, phase: 'idle'});
    } catch (err: any) {
      set({error: err.message});
    }
  },

  deleteCheckpoint: async (name: string) => {
    try {
      await deleteCheckpoint(name);
      await get().refresh();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  clearError: () => set({error: null}),
}));
