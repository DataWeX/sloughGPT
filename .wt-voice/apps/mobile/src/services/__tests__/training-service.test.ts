jest.mock('../api-client', () => ({
  api: {
    post: jest.fn().mockResolvedValue({status: 'ok'}),
    get: jest.fn().mockResolvedValue([]),
    delete: jest.fn().mockResolvedValue({status: 'deleted'}),
  },
}));

jest.mock('../sse-client', () => ({
  streamSSE: jest.fn(),
}));

import * as training from '../training-service';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const mockApi = require('../api-client').api;

beforeEach(() => {
  mockApi.post.mockReset();
  mockApi.post.mockResolvedValue({status: 'ok'});
  mockApi.get.mockReset();
  mockApi.get.mockResolvedValue([]);
  mockApi.delete.mockReset();
  mockApi.delete.mockResolvedValue({status: 'deleted'});
});

describe('training-service', () => {
  it('startTraining calls POST /auto-train/start', async () => {
    const result = await training.startTraining({
      epochs: 5,
      learning_rate: 0.001,
      batch_size: 8,
      soul_name: 'test',
      algo: 'lstm',
    });
    expect(mockApi.post).toHaveBeenCalledWith('/auto-train/start', expect.objectContaining({
      epochs: 5,
      algo: 'lstm',
    }));
    expect(result).toEqual({status: 'ok'});
  });

  it('stopTraining calls POST /auto-train/stop', async () => {
    await training.stopTraining();
    expect(mockApi.post).toHaveBeenCalledWith('/auto-train/stop');
  });

  it('getTrainingStatus calls GET /auto-train/status', async () => {
    mockApi.get.mockResolvedValueOnce({running: false, config: {}});
    const result = await training.getTrainingStatus();
    expect(mockApi.get).toHaveBeenCalledWith('/auto-train/status');
    expect(result.running).toBe(false);
  });

  it('listCheckpoints calls GET /auto-train/checkpoints', async () => {
    mockApi.get.mockResolvedValueOnce([{name: 'cp1', loss: 0.5}]);
    const result = await training.listCheckpoints();
    expect(mockApi.get).toHaveBeenCalledWith('/auto-train/checkpoints');
    expect(result.length).toBe(1);
  });

  it('deleteCheckpoint calls DELETE', async () => {
    await training.deleteCheckpoint('cp1');
    expect(mockApi.delete).toHaveBeenCalledWith('/auto-train/checkpoints/cp1');
  });

  it('loadCheckpoint calls POST', async () => {
    await training.loadCheckpoint('cp1');
    expect(mockApi.post).toHaveBeenCalledWith('/auto-train/checkpoints/cp1/load');
  });

  it('listDatasets calls GET /datasets', async () => {
    mockApi.get.mockResolvedValueOnce([]);
    await training.listDatasets();
    expect(mockApi.get).toHaveBeenCalledWith('/datasets');
  });

  it('listTrainingJobs calls GET /training/jobs', async () => {
    mockApi.get.mockResolvedValueOnce([]);
    await training.listTrainingJobs();
    expect(mockApi.get).toHaveBeenCalledWith('/training/jobs');
  });

  it('startLoraFinetune calls POST /training/lora-finetune', async () => {
    await training.startLoraFinetune({model_path: '/m', dataset: 'd'});
    expect(mockApi.post).toHaveBeenCalledWith('/training/lora-finetune', expect.objectContaining({
      model_path: '/m',
    }));
  });
});
