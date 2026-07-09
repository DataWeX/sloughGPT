import * as trainingService from '../../services/training-service';

jest.mock('../../services/training-service');

const mockService = trainingService as jest.Mocked<typeof trainingService>;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useTrainingStore} = require('../training-store');

function makeCheckpoint(overrides: Partial<any> = {}) {
  return {
    name: 'ckpt-1',
    soul: 'assistant',
    loss: 1.23,
    steps: 100,
    traits: {},
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

function makeDataset(overrides: Partial<any> = {}) {
  return {
    id: 'ds-1',
    name: 'Test Dataset',
    file_count: 3,
    total_chars: 12345,
    ...overrides,
  };
}

const defaultState = {
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
  method: 'distill',
  hfJobId: null,
  hfJobs: [],
  hfFinetunedPath: null,
};

let hfTimerHolder: ReturnType<typeof setInterval> | null = null;

beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers({advanceTimers: true});
  useTrainingStore.setState(defaultState);
  mockService.streamTraining.mockReturnValue(
    (async function* () {})() as AsyncGenerator<any>,
  );
});

afterEach(() => {
  jest.useRealTimers();
});

// ── Initial state ──────────────────────────────────────────────────────────

describe('initial state', () => {
  it('has correct defaults', () => {
    const s = useTrainingStore.getState();
    expect(s.phase).toBe('idle');
    expect(s.running).toBe(false);
    expect(s.method).toBe('distill');
    expect(s.config.epochs).toBe(10);
    expect(s.hfOpts.use_lora).toBe(true);
  });
});

// ── setConfig ──────────────────────────────────────────────────────────────

describe('setConfig', () => {
  it('merges partial config', () => {
    useTrainingStore.getState().setConfig({epochs: 20, batch_size: 128});
    const s = useTrainingStore.getState();
    expect(s.config.epochs).toBe(20);
    expect(s.config.batch_size).toBe(128);
    expect(s.config.learning_rate).toBe(0.001);
  });
});

// ── setHfOpts ──────────────────────────────────────────────────────────────

describe('setHfOpts', () => {
  it('merges partial HF opts', () => {
    useTrainingStore.getState().setHfOpts({epochs: 5, model: 'gpt2-medium'});
    const s = useTrainingStore.getState();
    expect(s.hfOpts.epochs).toBe(5);
    expect(s.hfOpts.model).toBe('gpt2-medium');
    expect(s.hfOpts.batch_size).toBe(4);
  });
});

// ── setMethod ──────────────────────────────────────────────────────────────

describe('setMethod', () => {
  it('switches to finetune and clears related fields', () => {
    useTrainingStore.setState({hfJobId: 'j1', hfFinetunedPath: '/tmp/model', error: 'old err'});
    useTrainingStore.getState().setMethod('finetune');
    const s = useTrainingStore.getState();
    expect(s.method).toBe('finetune');
    expect(s.hfJobId).toBeNull();
    expect(s.hfFinetunedPath).toBeNull();
    expect(s.error).toBeNull();
  });
});

// ── start (distill) ────────────────────────────────────────────────────────

describe('start (distill)', () => {
  it('calls startTraining and streams events', async () => {
    useTrainingStore.setState({config: {...useTrainingStore.getState().config, source_text: 'hello'}});
    mockService.startTraining.mockResolvedValue({status: 'started', data_path: '/tmp', epochs: 10});
    mockService.streamTraining.mockReturnValue(
      (async function* () {
        yield {raw: {status: 'working', data: {loss: 0.5, step: 1}}};
        yield {raw: {status: 'complete', data: {checkpoint: 'final'}}};
      })(),
    );

    await useTrainingStore.getState().start();

    const s = useTrainingStore.getState();
    expect(s.phase).toBe('COMPLETE');
    expect(s.running).toBe(false);
    expect(s.loss).toBe(0.5);
    expect(s.checkpoint).toBe('final');
  });

  it('sets error when no source text, dataset, or checkpoint', async () => {
    useTrainingStore.setState({config: {...useTrainingStore.getState().config, source_text: undefined, dataset_id: undefined, checkpoint_name: undefined}});
    await useTrainingStore.getState().start();
    expect(useTrainingStore.getState().error).toContain('training text');
    expect(useTrainingStore.getState().running).toBe(false);
  });

  it('handles SSE stream error event', async () => {
    useTrainingStore.setState({config: {...useTrainingStore.getState().config, source_text: 'hello'}});
    mockService.startTraining.mockResolvedValue({status: 'started', data_path: '/tmp', epochs: 10});
    mockService.streamTraining.mockReturnValue(
      (async function* () {
        yield {raw: {status: 'error', message: 'OOM'}};
      })(),
    );

    await useTrainingStore.getState().start();
    expect(useTrainingStore.getState().phase).toBe('FAILED');
    expect(useTrainingStore.getState().error).toContain('OOM');
  });

  it('handles startTraining failure', async () => {
    useTrainingStore.setState({config: {...useTrainingStore.getState().config, source_text: 'hello'}});
    mockService.startTraining.mockRejectedValue(new Error('server down'));

    await useTrainingStore.getState().start();
    expect(useTrainingStore.getState().phase).toBe('FAILED');
    expect(useTrainingStore.getState().error).toContain('server down');
  });

  it('records cancelled event', async () => {
    useTrainingStore.setState({config: {...useTrainingStore.getState().config, source_text: 'hello'}});
    mockService.startTraining.mockResolvedValue({status: 'started', data_path: '/tmp', epochs: 10});
    mockService.streamTraining.mockReturnValue(
      (async function* () {
        yield {raw: {data: {cancelled: true}}};
      })(),
    );

    // Mock refresh dependencies
    mockService.listCheckpoints.mockResolvedValue([]);
    mockService.listDatasets.mockResolvedValue([]);
    mockService.getTrainingStatus.mockResolvedValue({running: false, config: {}});
    mockService.listTrainingJobs.mockResolvedValue([]);

    await useTrainingStore.getState().start();
    expect(useTrainingStore.getState().phase).toBe('idle');
    expect(useTrainingStore.getState().running).toBe(false);
  });

  it('delegates to startHFFineTune when method is finetune', async () => {
    useTrainingStore.setState({method: 'finetune', hfOpts: {...useTrainingStore.getState().hfOpts, dataset: 'ds-1'}});
    mockService.startHFFineTune.mockResolvedValue({job_id: 'hf-1'});
    mockService.listTrainingJobs.mockResolvedValue([]);

    await useTrainingStore.getState().start();
    expect(mockService.startHFFineTune).toHaveBeenCalled();
  });
});

// ── startHFFineTune ────────────────────────────────────────────────────────

describe('startHFFineTune', () => {
  it('starts HF job and polls for completion', async () => {
    useTrainingStore.setState({hfOpts: {...useTrainingStore.getState().hfOpts, dataset: 'ds-1'}});
    mockService.startHFFineTune.mockResolvedValue({job_id: 'hf-1'});

    mockService.listTrainingJobs.mockResolvedValue([
      {job_id: 'hf-1', status: 'completed', steps: 50, epoch: 3, final_loss: 0.3, model_path: '/tmp/model'},
    ]);

    await useTrainingStore.getState().startHFFineTune();

    jest.advanceTimersByTime(3100);
    await Promise.resolve();

    const s = useTrainingStore.getState();
    expect(s.phase).toBe('COMPLETE');
    expect(s.hfFinetunedPath).toBe('/tmp/model');
    expect(s.running).toBe(false);
  });

  it('sets error when no dataset', async () => {
    useTrainingStore.setState({hfOpts: {...useTrainingStore.getState().hfOpts, dataset: ''}, config: {...useTrainingStore.getState().config, dataset_id: undefined}});
    await useTrainingStore.getState().startHFFineTune();
    expect(useTrainingStore.getState().error).toContain('dataset');
    expect(useTrainingStore.getState().running).toBe(false);
  });

  it('handles HF start failure', async () => {
    useTrainingStore.setState({hfOpts: {...useTrainingStore.getState().hfOpts, dataset: 'ds-1'}});
    mockService.startHFFineTune.mockRejectedValue(new Error('quota exceeded'));

    await useTrainingStore.getState().startHFFineTune();
    expect(useTrainingStore.getState().phase).toBe('FAILED');
    expect(useTrainingStore.getState().error).toContain('quota exceeded');
  });

  it('handles HF job failed status', async () => {
    useTrainingStore.setState({hfOpts: {...useTrainingStore.getState().hfOpts, dataset: 'ds-1'}});
    mockService.startHFFineTune.mockResolvedValue({job_id: 'hf-1'});
    mockService.listTrainingJobs.mockResolvedValue([{job_id: 'hf-1', status: 'failed', error: 'OOM'}]);

    await useTrainingStore.getState().startHFFineTune();

    jest.advanceTimersByTime(3100);
    await Promise.resolve();

    expect(useTrainingStore.getState().phase).toBe('FAILED');
  });
});

// ── stop ───────────────────────────────────────────────────────────────────

describe('stop', () => {
  it('aborts distill training', async () => {
    useTrainingStore.setState({running: true, phase: 'TRAINING'});
    mockService.stopTraining.mockResolvedValue({status: 'stopped'});

    await useTrainingStore.getState().stop();
    expect(useTrainingStore.getState().phase).toBe('idle');
    expect(useTrainingStore.getState().running).toBe(false);
    expect(mockService.stopTraining).toHaveBeenCalled();
  });

  it('clears HF polling for finetune', async () => {
    useTrainingStore.setState({method: 'finetune', hfJobId: 'hf-1', running: true});

    await useTrainingStore.getState().stop();
    expect(useTrainingStore.getState().hfJobId).toBeNull();
    expect(useTrainingStore.getState().running).toBe(false);
  });
});

// ── refresh ────────────────────────────────────────────────────────────────

describe('refresh', () => {
  it('fetches checkpoints, datasets, status, and HF jobs', async () => {
    const checkpoints = [makeCheckpoint()];
    const datasets = [makeDataset()];
    const status = {running: false, config: {}};
    const hfJobs = [{job_id: 'hf-1', status: 'completed'}];

    mockService.listCheckpoints.mockResolvedValue(checkpoints);
    mockService.listDatasets.mockResolvedValue(datasets);
    mockService.getTrainingStatus.mockResolvedValue(status);
    mockService.listTrainingJobs.mockResolvedValue(hfJobs);

    await useTrainingStore.getState().refresh();

    const s = useTrainingStore.getState();
    expect(s.checkpoints).toEqual(checkpoints);
    expect(s.datasets).toEqual(datasets);
    expect(s.running).toBe(false);
    expect(s.hfJobs).toEqual(hfJobs);
  });

  it('handles partial failures gracefully', async () => {
    mockService.listCheckpoints.mockRejectedValue(new Error('err'));
    mockService.listDatasets.mockResolvedValue([makeDataset()]);
    mockService.getTrainingStatus.mockRejectedValue(new Error('err'));
    mockService.listTrainingJobs.mockRejectedValue(new Error('err'));

    await useTrainingStore.getState().refresh();

    const s = useTrainingStore.getState();
    expect(s.checkpoints).toEqual([]);
    expect(s.datasets).toHaveLength(1);
    expect(s.running).toBe(false);
  });
});

// ── loadCheckpoint ─────────────────────────────────────────────────────────

describe('loadCheckpoint', () => {
  it('loads and sets checkpoint name', async () => {
    mockService.loadCheckpoint.mockResolvedValue(undefined);
    await useTrainingStore.getState().loadCheckpoint('best-model');
    expect(useTrainingStore.getState().checkpoint).toBe('best-model');
  });

  it('sets error on failure', async () => {
    mockService.loadCheckpoint.mockRejectedValue(new Error('not found'));
    await useTrainingStore.getState().loadCheckpoint('missing');
    expect(useTrainingStore.getState().error).toBe('not found');
  });
});

// ── deleteCheckpoint ───────────────────────────────────────────────────────

describe('deleteCheckpoint', () => {
  it('deletes and refreshes', async () => {
    mockService.deleteCheckpoint.mockResolvedValue(undefined);
    mockService.listCheckpoints.mockResolvedValue([]);
    mockService.listDatasets.mockResolvedValue([]);
    mockService.getTrainingStatus.mockResolvedValue({running: false, config: {}});
    mockService.listTrainingJobs.mockResolvedValue([]);

    await useTrainingStore.getState().deleteCheckpoint('old-ckpt');
    expect(mockService.deleteCheckpoint).toHaveBeenCalledWith('old-ckpt');
  });

  it('sets error on failure', async () => {
    mockService.deleteCheckpoint.mockRejectedValue(new Error('permission denied'));
    await useTrainingStore.getState().deleteCheckpoint('protected');
    expect(useTrainingStore.getState().error).toBe('permission denied');
  });
});

// ── clearError ─────────────────────────────────────────────────────────────

describe('clearError', () => {
  it('clears error', () => {
    useTrainingStore.setState({error: 'some error'});
    useTrainingStore.getState().clearError();
    expect(useTrainingStore.getState().error).toBeNull();
  });
});
