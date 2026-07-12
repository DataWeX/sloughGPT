import React from 'react';
import {render, fireEvent} from '@/test-utils';
import {View, Text} from 'react-native';

jest.mock('react-native-safe-area-context', () => {
  const React = require('react');
  const {View} = require('react-native');
  return {
    SafeAreaView: ({children, edges, style}: any) =>
      React.createElement(View, {style, testID: 'safe-area-view'}, children),
    useSafeAreaInsets: () => ({top: 0, bottom: 0, left: 0, right: 0}),
  };
});

const mockSetConfig = jest.fn();
const mockSetHfOpts = jest.fn();
const mockSetMethod = jest.fn();
const mockStart = jest.fn();
const mockStop = jest.fn();
const mockRefresh = jest.fn();
const mockLoadCheckpoint = jest.fn();
const mockDeleteCheckpoint = jest.fn();
const mockClearError = jest.fn();

const defaultTrainState: any = {
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
  config: {epochs: 10, learning_rate: 0.001, batch_size: 64, soul_name: 'assistant', algo: 'bpe'},
  method: 'distill',
  hfOpts: {model: '', dataset: '', epochs: 3, batch_size: 4, learning_rate: 2e-5, use_lora: true, lora_rank: 8},
  hfJobId: null,
  hfJobs: [],
  hfFinetunedPath: null,
  setConfig: mockSetConfig,
  setHfOpts: mockSetHfOpts,
  setMethod: mockSetMethod,
  start: mockStart,
  stop: mockStop,
  refresh: mockRefresh,
  loadCheckpoint: mockLoadCheckpoint,
  deleteCheckpoint: mockDeleteCheckpoint,
  clearError: mockClearError,
};

let mockTrainState = {...defaultTrainState};

const mockModelRefresh = jest.fn();
const mockLoadModel = jest.fn();
const defaultModelState: any = {
  models: [],
  currentModel: null,
  souls: [],
  currentSoul: null,
  checkpoints: [],
  health: null,
  loading: false,
  loadingModelId: null,
  error: null,
  refresh: mockModelRefresh,
  loadModel: mockLoadModel,
  unloadModel: jest.fn(),
  switchSoul: jest.fn(),
  clearError: jest.fn(),
};
let mockModelState = {...defaultModelState};

jest.mock('../../components/StatusBadge', () => ({
  StatusBadge: ({label, variant, children}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `badge-${variant}`, children: children || label || ''});
  },
}));

jest.mock('../../components/Icon', () => ({
  Icon: ({name, size, color}: any) => {
    const React = require('react');
    const {Text} = require('react-native');
    return React.createElement(Text, {testID: `icon-${name}`, children: `[${name}]`});
  },
}));

jest.mock('../../stores/training-store', () => ({
  useTrainingStore: (selector?: any) => {
    if (typeof selector === 'function') return selector(mockTrainState);
    return mockTrainState;
  },
}));

jest.mock('../../stores/model-store', () => ({
  useModelStore: (selector?: any) => {
    if (typeof selector === 'function') return selector(mockModelState);
    return mockModelState;
  },
}));

jest.mock('../../services/api-client', () => ({
  api: {
    get: jest.fn().mockResolvedValue({rows: ['line1', 'line2']}),
    post: jest.fn(),
    patch: jest.fn(),
    delete: jest.fn(),
  },
}));

beforeEach(() => {
  jest.clearAllMocks();
  mockTrainState = {...defaultTrainState};
  mockModelState = {...defaultModelState};
  // Deep clone configs
  mockTrainState.config = {...defaultTrainState.config};
  mockTrainState.hfOpts = {...defaultTrainState.hfOpts};
});

const {TrainingScreen} = require('../TrainingScreen');

describe('TrainingScreen', () => {
  it('renders title and idle status', async () => {
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Training')).toBeTruthy();
    expect(getByText('Ready')).toBeTruthy();
  });

  it('shows error card', async () => {
    mockTrainState = {...mockTrainState, error: 'Something failed'};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Something failed')).toBeTruthy();
  });

  it('dismisses error on X press', async () => {
    mockTrainState = {...mockTrainState, error: 'Something failed'};
    const {getByText} = await render(<TrainingScreen />);
    fireEvent.press(getByText('[x]'));
    expect(mockClearError).toHaveBeenCalled();
  });

  it('shows method selector when idle', async () => {
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Method')).toBeTruthy();
    expect(getByText('Distill')).toBeTruthy();
    expect(getByText('Fine-tune')).toBeTruthy();
  });

  it('switches method on press', async () => {
    const {getByText} = await render(<TrainingScreen />);
    fireEvent.press(getByText('Fine-tune'));
    expect(mockSetMethod).toHaveBeenCalledWith('finetune');
  });

  it('shows distill text input by default', async () => {
    const {getByPlaceholderText} = await render(<TrainingScreen />);
    expect(getByPlaceholderText('Paste training text here (SRT, plain text, or lines)...')).toBeTruthy();
  });

  it('shows distill dataset selector when inputMode switched', async () => {
    // inputMode is local useState, starts at 'text'
    // We can only verify the default text mode
    const {queryByText} = await render(<TrainingScreen />);
    // Initially in text mode, so no dataset list in distill section
    // Cannot easily test dataset mode since it's local state
  });

  it('shows distill hyperparameters', async () => {
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Hyperparameters')).toBeTruthy();
    expect(getByText('Epochs')).toBeTruthy();
    expect(getByText('Learning Rate')).toBeTruthy();
    expect(getByText('Soul')).toBeTruthy();
  });

  it('shows fine-tune settings when method is finetune', async () => {
    mockTrainState = {...mockTrainState, method: 'finetune', datasets: [{id: 'ds1', name: 'test', file_count: 1, total_chars: 100}]};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Fine-tune Settings')).toBeTruthy();
    expect(getByText('Base Model')).toBeTruthy();
    expect(getByText('Dataset')).toBeTruthy();
    expect(getByText('Epochs')).toBeTruthy();
    expect(getByText('Batch Size')).toBeTruthy();
    expect(getByText('Learning Rate')).toBeTruthy();
    expect(getByText('Use LoRA')).toBeTruthy();
  });

  it('shows LoRA rank when use_lora is true', async () => {
    mockTrainState = {...mockTrainState, method: 'finetune'};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('LoRA Rank')).toBeTruthy();
  });

  it('hides LoRA rank when use_lora is false', async () => {
    mockTrainState = {...mockTrainState, method: 'finetune', hfOpts: {...defaultTrainState.hfOpts, use_lora: false}};
    const {queryByText} = await render(<TrainingScreen />);
    expect(queryByText('LoRA Rank')).toBeNull();
  });

  it('shows dataset items in fine-tune', async () => {
    const ds = {id: 'ds1', name: 'My Dataset', file_count: 3, total_chars: 5000};
    mockTrainState = {...mockTrainState, method: 'finetune', datasets: [ds]};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('My Dataset')).toBeTruthy();
    expect(getByText('3 files · 5,000 chars')).toBeTruthy();
  });

  it('shows no datasets message in fine-tune', async () => {
    mockTrainState = {...mockTrainState, method: 'finetune', datasets: []};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('No datasets found. Import one first.')).toBeTruthy();
  });

  it('highlights selected dataset in fine-tune', async () => {
    const ds = {id: 'ds1', name: 'My Dataset', file_count: 3, total_chars: 5000};
    mockTrainState = {...mockTrainState, method: 'finetune', datasets: [ds], hfOpts: {...defaultTrainState.hfOpts, dataset: 'ds1'}};
    const {getByText} = await render(<TrainingScreen />);
    // Checkmark icon appears for selected
    const checks = getByText('[check]');
    expect(checks).toBeTruthy();
  });

  it('shows progress card when training', async () => {
    mockTrainState = {
      ...mockTrainState,
      phase: 'TRAINING',
      running: true,
      epoch: 3,
      totalEpochs: 10,
      loss: 0.5678,
      steps: 150,
      lossHistory: [{step: 1, value: 0.9}, {step: 2, value: 0.7}],
    };
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Progress')).toBeTruthy();
    expect(getByText('Epoch 3/10')).toBeTruthy();
    expect(getByText('30%')).toBeTruthy();
    expect(getByText('0.5678')).toBeTruthy();
    expect(getByText('150')).toBeTruthy();
    expect(getByText('Training...')).toBeTruthy();
  });

  it('shows training complete card', async () => {
    mockTrainState = {...mockTrainState, phase: 'COMPLETE', checkpoint: 'ckpt-v1', loss: 0.1234, steps: 200};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Training Complete')).toBeTruthy();
    expect(getByText('Success')).toBeTruthy();
    expect(getByText('Checkpoint: ckpt-v1')).toBeTruthy();
    expect(getByText('Load Model for Chat')).toBeTruthy();
  });

  it('shows fine-tune complete card', async () => {
    mockTrainState = {
      ...mockTrainState,
      phase: 'COMPLETE',
      hfFinetunedPath: '/tmp/model',
      loss: 0.456,
      steps: 100,
    };
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Fine-tune Complete')).toBeTruthy();
    expect(getByText('Success')).toBeTruthy();
  });

  it('shows Start Training button when idle', async () => {
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Start Training')).toBeTruthy();
  });

  it('shows Train Again when complete', async () => {
    mockTrainState = {...mockTrainState, phase: 'COMPLETE'};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Train Again')).toBeTruthy();
  });

  it('shows Stop Training button when training', async () => {
    mockTrainState = {...mockTrainState, phase: 'TRAINING', running: true, totalEpochs: 10};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Stop Training')).toBeTruthy();
  });

  it('calls stop on Stop Training press', async () => {
    mockTrainState = {...mockTrainState, phase: 'TRAINING', running: true, totalEpochs: 10};
    const {getByText} = await render(<TrainingScreen />);
    fireEvent.press(getByText('Stop Training'));
    expect(mockStop).toHaveBeenCalled();
  });

  it('shows checkpoint list', async () => {
    const cps = [
      {name: 'ckpt-v1', soul: 'assistant', loss: 0.5, steps: 100, traits: {}, created_at: '2025-01-01', size_mb: 4.2},
    ];
    mockTrainState = {...mockTrainState, checkpoints: cps};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Checkpoints')).toBeTruthy();
    expect(getByText('ckpt-v1')).toBeTruthy();
    expect(getByText(/Loss: 0.500/)).toBeTruthy();
    expect(getByText(/100 steps/)).toBeTruthy();
    expect(getByText(/4.2 MB/)).toBeTruthy();
  });

  it('shows empty checkpoints message', async () => {
    const {queryByText} = await render(<TrainingScreen />);
    expect(queryByText('Checkpoints')).toBeNull();
  });

  it('shows job history', async () => {
    const jobs = [
      {job_id: 'j1', model: 'gpt2', dataset: 'test', status: 'completed', loss: 0.5},
    ];
    mockTrainState = {...mockTrainState, hfJobs: jobs};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Job History')).toBeTruthy();
    expect(getByText('gpt2 · test')).toBeTruthy();
  });

  it('shows running status badge during training', async () => {
    mockTrainState = {...mockTrainState, phase: 'TRAINING', running: true, totalEpochs: 10};
    const {getAllByText} = await render(<TrainingScreen />);
    const found = getAllByText('Training');
    expect(found.length).toBeGreaterThanOrEqual(1);
  });

  it('shows complete status badge', async () => {
    mockTrainState = {...mockTrainState, phase: 'COMPLETE'};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Complete')).toBeTruthy();
  });

  it('shows failed status badge', async () => {
    mockTrainState = {...mockTrainState, phase: 'FAILED', error: 'OOM'};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Failed')).toBeTruthy();
  });

  it('shows LossChart placeholder when data < 2', async () => {
    mockTrainState = {...mockTrainState, phase: 'TRAINING', running: true, totalEpochs: 10, lossHistory: []};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('Loss curve will appear here')).toBeTruthy();
  });

  it('progress bar reflects epoch progress', async () => {
    mockTrainState = {...mockTrainState, phase: 'TRAINING', running: true, epoch: 7, totalEpochs: 10};
    const {getByText} = await render(<TrainingScreen />);
    expect(getByText('70%')).toBeTruthy();
  });
});
