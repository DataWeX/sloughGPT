import {create} from 'zustand';
import {api} from '../services/api-client';
import {
  startTraining,
  stopTraining,
  getTrainingStatus,
  listCheckpoints,
  deleteCheckpoint,
  loadCheckpoint,
  listDatasets,
  streamTraining,
  startLoraFinetune,
  listTrainingJobs,
  deleteTrainingJob,
  stopTrainingJob,
  listFineTunedModels,
  loadFineTunedModel,
  deleteFineTunedModel,
  loadAdapter,
  unloadAdapter,
  importDatasetUrl,
  importDatasetGithub,
  importDatasetHuggingface,
  importDatasetCsv,
  type TrainConfig,
  type Checkpoint,
  type Dataset,
  type FineTunedModel,
  type TrainingJob,
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

export type TrainingMethod = 'distill' | 'finetune';

interface HFTrainingOpts {
  model: string;
  dataset: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  use_lora: boolean;
  lora_rank: number;
}

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
  method: TrainingMethod;
  hfOpts: HFTrainingOpts;
  hfJobId: string | null;
  hfJobs: TrainingJob[];
  hfFinetunedPath: string | null;
  finetunedModels: FineTunedModel[];
  adapterLoaded: boolean;

  setConfig: (partial: Partial<TrainConfig>) => void;
  setHfOpts: (partial: Partial<HFTrainingOpts>) => void;
  setMethod: (m: TrainingMethod) => void;
  start: () => Promise<void>;
  startLoraFinetune: () => Promise<void>;
  stop: () => Promise<void>;
  refresh: () => Promise<void>;
  refreshFinetunedModels: () => Promise<void>;
  loadCheckpoint: (name: string) => Promise<void>;
  deleteCheckpoint: (name: string) => Promise<void>;
  deleteJob: (jobId: string) => Promise<void>;
  stopJob: (jobId: string) => Promise<void>;
  loadFinetunedModel: (name: string) => Promise<void>;
  deleteFinetunedModel: (name: string) => Promise<void>;
  loadAdapterModel: (path: string) => Promise<void>;
  unloadAdapterModel: () => Promise<void>;
  importDataset: (source: string, name: string, type: 'url' | 'github' | 'huggingface' | 'csv') => Promise<void>;
  clearError: () => void;
}

let abortController: AbortController | null = null;
let hfPollTimer: ReturnType<typeof setInterval> | null = null;

/** Cleanup SSE + poll timers — call on screen unmount. */
export function cleanupTraining() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
  if (hfPollTimer) {
    clearInterval(hfPollTimer);
    hfPollTimer = null;
  }
}

const defaultConfig: TrainConfig = {
  epochs: 10,
  learning_rate: 0.001,
  batch_size: 64,
  soul_name: 'assistant',
  algo: 'bpe',
};

const defaultHfOpts: HFTrainingOpts = {
  model: '',
  dataset: '',
  epochs: 3,
  batch_size: 4,
  learning_rate: 2e-5,
  use_lora: true,
  lora_rank: 8,
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
  method: 'distill',
  hfOpts: {...defaultHfOpts},
  hfJobId: null,
  hfJobs: [],
  hfFinetunedPath: null,
  finetunedModels: [],
  adapterLoaded: false,

  setConfig: partial => set(s => ({config: {...s.config, ...partial}})),
  setHfOpts: partial => set(s => ({hfOpts: {...s.hfOpts, ...partial}})),
  setMethod: method => set({method, error: null, hfJobId: null, hfFinetunedPath: null}),

  start: async () => {
    const state = get();
    if (state.method === 'finetune') {
      await get().startLoraFinetune();
      return;
    }

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
        const rawPhase = event.phase;
        const rawStatus = event.status;
        const rawData = (event.data || {}) as Record<string, any>;

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
          const msg = event.message || rawData.error || 'Training failed';
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

  startLoraFinetune: async () => {
    const {hfOpts, config} = get();
    const dataset = hfOpts.dataset || config.dataset_id;
    if (!dataset) {
      set({error: 'Select a dataset for fine-tuning'});
      return;
    }
    const modelPath = hfOpts.model || 'models/gpt2.slnc';

    set({
      phase: 'TRAINING',
      running: true,
      loss: null,
      epoch: 0,
      totalEpochs: hfOpts.epochs,
      steps: 0,
      error: null,
      hfFinetunedPath: null,
    });

    try {
      const result = await startLoraFinetune({
        model_path: modelPath,
        dataset,
        rank: hfOpts.lora_rank,
        epochs: hfOpts.epochs,
        batch_size: hfOpts.batch_size,
        learning_rate: hfOpts.learning_rate,
      });
      const jobId = result.job_id;
      set({hfJobId: jobId});

      hfPollTimer = setInterval(async () => {
        try {
          const jobs = await listTrainingJobs();
          const job = (jobs || []).find(
            (j: TrainingJob) => j.job_id === jobId || j.id === jobId,
          );
          if (!job) return;

          const jobStatus = job.status || '';
          const jobPhase = job.phase || 'TRAINING';
          set({
            phase:
              jobPhase === 'complete'
                ? 'COMPLETE'
                : jobPhase === 'failed'
                ? 'FAILED'
                : ('TRAINING' as TrainPhase),
            steps: job.global_step || get().steps,
            epoch: job.current_epoch || get().epoch,
            loss: job.loss ?? job.eval_loss ?? get().loss,
            hfJobs: jobs || [],
          });

          const path = (job as any).output_dir || (job as any).model_path;
          if (path) {
            set({hfFinetunedPath: path});
          }

          if (jobStatus === 'completed' || jobPhase === 'complete') {
            set({
              phase: 'COMPLETE',
              running: false,
              loss: job.eval_loss ?? job.loss ?? get().loss,
            });
            clearInterval(hfPollTimer!);
            hfPollTimer = null;
            get().refresh();
          } else if (jobStatus === 'failed' || jobPhase === 'failed') {
            set({
              phase: 'FAILED',
              error: job.error || 'Fine-tuning failed',
              running: false,
            });
            clearInterval(hfPollTimer!);
            hfPollTimer = null;
          }
        } catch {}
      }, 3000);
    } catch (err: any) {
      set({phase: 'FAILED', error: err.message, running: false});
    }
  },

  stop: async () => {
    if (get().method === 'finetune') {
      if (hfPollTimer) {
        clearInterval(hfPollTimer);
        hfPollTimer = null;
      }
      const jobId = get().hfJobId;
      if (jobId) {
        try {
          await stopTrainingJob(jobId);
        } catch {}
      }
      set({phase: 'idle', running: false, hfJobId: null});
      return;
    }
    abortController?.abort();
    try {
      await stopTraining();
    } catch {}
    set({phase: 'idle', running: false});
  },

  refresh: async () => {
    try {
      const [ckpts, datasets, status, hfJobs] = await Promise.all([
        listCheckpoints().catch(() => []),
        listDatasets().catch(() => []),
        getTrainingStatus().catch(() => ({running: false, config: {}})),
        listTrainingJobs().catch(() => []),
      ]);
      set({
        checkpoints: ckpts,
        datasets,
        running: status.running,
        hfJobs,
      });
    } catch {}
  },

  refreshFinetunedModels: async () => {
    try {
      const result = await listFineTunedModels();
      set({finetunedModels: result.models || []});
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

  deleteJob: async (jobId: string) => {
    try {
      await deleteTrainingJob(jobId);
      await get().refresh();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  stopJob: async (jobId: string) => {
    try {
      await stopTrainingJob(jobId);
      await get().refresh();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  loadFinetunedModel: async (name: string) => {
    try {
      await loadFineTunedModel(name);
      set({adapterLoaded: true});
    } catch (err: any) {
      set({error: err.message});
    }
  },

  deleteFinetunedModel: async (name: string) => {
    try {
      await deleteFineTunedModel(name);
      await get().refreshFinetunedModels();
    } catch (err: any) {
      set({error: err.message});
    }
  },

  loadAdapterModel: async (path: string) => {
    try {
      await loadAdapter(path);
      set({adapterLoaded: true});
    } catch (err: any) {
      set({error: err.message});
    }
  },

  unloadAdapterModel: async () => {
    try {
      await unloadAdapter();
      set({adapterLoaded: false});
    } catch (err: any) {
      set({error: err.message});
    }
  },

  importDataset: async (source: string, name: string, type: 'url' | 'github' | 'huggingface' | 'csv') => {
    try {
      switch (type) {
        case 'url':
          await importDatasetUrl(source, name || undefined);
          break;
        case 'github':
          await importDatasetGithub(source, name || undefined);
          break;
        case 'huggingface':
          await importDatasetHuggingface(source, name || undefined);
          break;
        case 'csv':
          await importDatasetCsv(source, name || undefined);
          break;
      }
      await get().refresh();
    } catch (err: any) {
      set({error: err.message});
      throw err;
    }
  },

  clearError: () => set({error: null}),
}));
