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
  startLoraFinetune,
  listTrainingJobs,
  type TrainConfig,
  type Checkpoint,
  type Dataset,
} from '../services/training-service';
import {triggerHaptic} from '../services/haptics';

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
  hfJobs: any[];
  hfFinetunedPath: string | null;

  setConfig: (partial: Partial<TrainConfig>) => void;
  setHfOpts: (partial: Partial<HFTrainingOpts>) => void;
  setMethod: (m: TrainingMethod) => void;
  start: () => Promise<void>;
  startLoraFinetune: () => Promise<void>;
  stop: () => Promise<void>;
  refresh: () => Promise<void>;
  loadCheckpoint: (name: string) => Promise<void>;
  deleteCheckpoint: (name: string) => Promise<void>;
  clearError: () => void;
}

let abortController: AbortController | null = null;
let hfPollTimer: ReturnType<typeof setInterval> | null = null;

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
      triggerHaptic('error');
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
          triggerHaptic('success');
          set({phase: 'COMPLETE', running: false});
          break;
        }

        if (rawStatus === 'error') {
          const msg = event.message || rawData.error || 'Training failed';
          triggerHaptic('error');
          set({phase: 'FAILED', error: String(msg), running: false});
          break;
        }
      }

      get().refresh();
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        triggerHaptic('error');
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

      // Poll for completion
      hfPollTimer = setInterval(async () => {
        try {
          const jobs = await listTrainingJobs();
          const job = (jobs || []).find((j: any) => j.job_id === jobId || j.id === jobId);
          if (!job) return;

          const jobStatus = job.status || '';
          const jobPhase = job.phase || 'TRAINING';
          set({
            phase: jobPhase === 'complete' ? 'COMPLETE' : jobPhase === 'failed' ? 'FAILED' : 'TRAINING' as TrainPhase,
            steps: job.steps || job.global_step || get().steps,
            epoch: job.epoch || job.current_epoch || get().epoch,
            loss: job.loss ?? job.final_loss ?? get().loss,
            hfJobs: jobs || [],
          });

          if (job.model_path) {
            set({hfFinetunedPath: job.model_path});
          }

          if (jobStatus === 'completed' || jobPhase === 'complete') {
            triggerHaptic('success');
            set({phase: 'COMPLETE', running: false, loss: job.final_loss ?? job.loss ?? get().loss});
            clearInterval(hfPollTimer!);
            hfPollTimer = null;
            get().refresh();
          } else if (jobStatus === 'failed' || jobPhase === 'failed') {
            triggerHaptic('error');
            set({phase: 'FAILED', error: job.error || 'Fine-tuning failed', running: false});
            clearInterval(hfPollTimer!);
            hfPollTimer = null;
          }
        } catch {}
      }, 3000);
    } catch (err: any) {
      triggerHaptic('error');
      set({phase: 'FAILED', error: err.message, running: false});
    }
  },

  stop: async () => {
    if (get().method === 'finetune') {
      if (hfPollTimer) {
        clearInterval(hfPollTimer);
        hfPollTimer = null;
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

  loadCheckpoint: async (name: string) => {
    try {
      await loadCheckpoint(name);
      triggerHaptic('success');
      set({checkpoint: name});
    } catch (err: any) {
      triggerHaptic('error');
      set({error: err.message});
    }
  },

  deleteCheckpoint: async (name: string) => {
    try {
      await deleteCheckpoint(name);
      triggerHaptic('success');
      await get().refresh();
    } catch (err: any) {
      triggerHaptic('error');
      set({error: err.message});
    }
  },

  clearError: () => set({error: null}),
}));
