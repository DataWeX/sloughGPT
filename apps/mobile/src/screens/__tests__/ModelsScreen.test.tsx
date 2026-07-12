/**
 * Tests for the ModelsScreen local inference card.
 *
 * NOTE: render() is async in RNTL v14 — always await the result.
 */

import React from 'react';
import {render, fireEvent} from '@/test-utils';

jest.mock('../../stores/model-store', () => ({
  useModelStore: () => ({
    models: [],
    currentModel: null,
    souls: [],
    currentSoul: null,
    checkpoints: [],
    health: {model_loaded: false},
    loading: false,
    loadingModelId: null,
    error: null,
    refresh: jest.fn(),
    loadModel: jest.fn(),
    unloadModel: jest.fn(),
    switchSoul: jest.fn(),
    clearError: jest.fn(),
  }),
}));

jest.mock('../../stores/hybrid-inference-store', () => ({
  useHybridStore: (overrides?: any) => ({
    slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: ''},
    qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
    activeEngine: 'remote',
    downloadProgress: 0,
    lastError: null,
    setActiveEngine: jest.fn(),
    loadSloNet: jest.fn(),
    loadQwen: jest.fn(),
    unloadSloNet: jest.fn(),
    unloadQwen: jest.fn(),
    unloadAll: jest.fn(),
    decideRoute: jest.fn(),
    executeLocal: jest.fn(),
    ...(overrides || {}),
  }),
}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {ModelsScreen} = require('../ModelsScreen');

describe('ModelsScreen — Local Inference card', () => {
  it('renders without crashing', async () => {
    await expect(render(<ModelsScreen />)).resolves.not.toThrow();
  });

  it('renders the card title', async () => {
    const view = await render(<ModelsScreen />);
    expect(view.getByText('On-Device Inference')).toBeTruthy();
  });

  it('renders engine selector chips', async () => {
    const view = await render(<ModelsScreen />);
    expect(view.getByText('SloNet')).toBeTruthy();
    expect(view.getByText('Qwen')).toBeTruthy();
    expect(view.getByText('Server')).toBeTruthy();
  });

  it('calls setActiveEngine when Qwen chip pressed', async () => {
    const setActiveEngine = jest.fn();
    // Override useHybridStore mock for this test
    const overrideMock = jest.spyOn(
      require('../../stores/hybrid-inference-store'),
      'useHybridStore',
    );
    overrideMock.mockReturnValue({
      slonet: {kind: 'slonet', loaded: false, modelName: '', downloadProgress: null, description: ''},
      qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
      activeEngine: 'remote',
      downloadProgress: 0,
      lastError: null,
      setActiveEngine,
      loadSloNet: jest.fn(),
      loadQwen: jest.fn(),
      unloadSloNet: jest.fn(),
      unloadQwen: jest.fn(),
      unloadAll: jest.fn(),
      decideRoute: jest.fn(),
      executeLocal: jest.fn(),
    });

    const view = await render(<ModelsScreen />);
    fireEvent.press(view.getByText('Qwen'));
    expect(setActiveEngine).toHaveBeenCalledWith('qwen');
    overrideMock.mockRestore();
  });

  it('shows Load button for SloNet', async () => {
    const view = await render(<ModelsScreen />);
    expect(view.getByText('Load')).toBeTruthy();
  });

  it('shows Unload button when SloNet is loaded', async () => {
    // Override mock to set SloNet loaded
    const overrideMock = jest.spyOn(
      require('../../stores/hybrid-inference-store'),
      'useHybridStore',
    );
    overrideMock.mockReturnValue({
      slonet: {kind: 'slonet', loaded: true, modelName: 'test', downloadProgress: null, description: ''},
      qwen: {kind: 'qwen', loaded: false, modelName: '', downloadProgress: null, description: ''},
      activeEngine: 'slonet',
      downloadProgress: 0,
      lastError: null,
      setActiveEngine: jest.fn(),
      loadSloNet: jest.fn(),
      loadQwen: jest.fn(),
      unloadSloNet: jest.fn(),
      unloadQwen: jest.fn(),
      unloadAll: jest.fn(),
      decideRoute: jest.fn(),
      executeLocal: jest.fn(),
    });

    const view = await render(<ModelsScreen />);
    expect(view.getByText('Unload')).toBeTruthy();
    overrideMock.mockRestore();
  });
});
