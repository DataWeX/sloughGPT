import {create} from 'zustand';
import {
  startTraining,
  stopTraining,
  getTrainingStatus,
  listCheckpoints,
  deleteCheckpoint,
  loadCheckpoint,
  listDatasets,
  streamTraining,
  type TrainConfig,
  type Checkpoint,
  type Dataset,
} from '../services/training-service';

export type TrainPhase =
  | 'idle'
  | 'configuring'
  | 'GENERATE_DATA'
  | 'DISTILL'
  | 'TRAIN'
  | 'EVALUATE'
  | 'DEPLOY'
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
        const raw = event.raw || {};
        const rawPhase = raw.phase as string | undefined;
        const rawStatus = raw.status as string | undefined;
        const rawData = (raw.data || {}) as Record<string, any>;

        if (rawPhase) {
          set({phase: rawPhase as TrainPhase});
        }

        if (rawData.loss !== undefined) {
          const step = rawData.step ?? rawData.global_step ?? get().steps;
          const ep = rawData.epoch ?? rawData.current_epoch ?? get().epoch;
          set(s => ({
            loss: Number(rawData.loss),
            steps: Number(step),
            epoch: Number(ep),
            lossHistory: [
              ...s.lossHistory,
              {step: Number(step), value: Number(rawData.loss)},
            ],
          }));
        }

        if (rawData.checkpoint) {
          set({checkpoint: String(rawData.checkpoint)});
        }

        if (rawData.progress_percent !== undefined) {
          const ep = Math.round(
            (Number(rawData.progress_percent) / 100) * get().totalEpochs,
          );
          set({epoch: ep});
        }

        if (rawData.final_loss !== undefined) {
          set({loss: Number(rawData.final_loss)});
        }

        if (rawData.cancelled) {
          set({phase: 'idle', running: false});
          break;
        }

        if (rawStatus === 'complete') {
          if (rawData.checkpoint) {
            set({checkpoint: String(rawData.checkpoint)});
          }
          set({phase: 'COMPLETE', running: false});
          break;
        }

        if (rawStatus === 'error') {
          const msg = raw.message || rawData.error || 'Training failed';
          set({phase: 'FAILED', error: String(msg), running: false});
          break;
        }
      }

      get().refresh();
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        set({phase: 'FAILED', error: err.message, running: false});
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
      await loadCheckpoint(name);
      set({checkpoint: name});
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
