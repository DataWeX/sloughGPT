const mockPost = jest.fn().mockResolvedValue({status: 'ok'});
const mockGet = jest.fn().mockResolvedValue([]);
const mockDelete = jest.fn().mockResolvedValue({status: 'deleted'});

jest.mock('../api-client', () => ({
  api: {
    post: (...args: any[]) => mockPost(...args),
    get: (...args: any[]) => mockGet(...args),
    delete: (...args: any[]) => mockDelete(...args),
  },
}));

jest.mock('../sse-client', () => ({
  streamSSE: Object.assign(
    jest.fn(async function* () { yield* []; }),
    {mockClear: jest.fn()},
  ),
}));

import * as training from '../training-service';

beforeEach(() => jest.clearAllMocks());

describe('training-service', () => {
  it('startTraining calls POST /auto-train/start', async () => {
    const result = await training.startTraining({
      epochs: 5,
      learning_rate: 0.001,
      batch_size: 8,
      soul_name: 'test',
      algo: 'lstm',
    });
    expect(mockPost).toHaveBeenCalledWith('/auto-train/start', expect.objectContaining({
      epochs: 5,
      algo: 'lstm',
    }));
    expect(result).toEqual({status: 'ok'});
  });

  it('stopTraining calls POST /auto-train/stop', async () => {
    await training.stopTraining();
    expect(mockPost).toHaveBeenCalledWith('/auto-train/stop');
  });

  it('getTrainingStatus calls GET /auto-train/status', async () => {
    mockGet.mockResolvedValueOnce({running: false, config: {}});
    const result = await training.getTrainingStatus();
    expect(mockGet).toHaveBeenCalledWith('/auto-train/status');
    expect(result.running).toBe(false);
  });

  it('listCheckpoints calls GET /auto-train/checkpoints', async () => {
    mockGet.mockResolvedValueOnce([{name: 'cp1', loss: 0.5}]);
    const result = await training.listCheckpoints();
    expect(mockGet).toHaveBeenCalledWith('/auto-train/checkpoints');
    expect(result.length).toBe(1);
  });

  it('deleteCheckpoint calls DELETE', async () => {
    await training.deleteCheckpoint('cp1');
    expect(mockDelete).toHaveBeenCalledWith('/auto-train/checkpoints/cp1');
  });

  it('loadCheckpoint calls POST', async () => {
    await training.loadCheckpoint('cp1');
    expect(mockPost).toHaveBeenCalledWith('/auto-train/checkpoints/cp1/load');
  });

  it('listDatasets calls GET /datasets', async () => {
    mockGet.mockResolvedValueOnce([]);
    await training.listDatasets();
    expect(mockGet).toHaveBeenCalledWith('/datasets');
  });

  it('listTrainingJobs calls GET /training/jobs', async () => {
    mockGet.mockResolvedValueOnce([]);
    await training.listTrainingJobs();
    expect(mockGet).toHaveBeenCalledWith('/training/jobs');
  });

  it('startLoraFinetune calls POST /training/lora-finetune', async () => {
    await training.startLoraFinetune({model_path: '/m', dataset: 'd'});
    expect(mockPost).toHaveBeenCalledWith('/training/lora-finetune', expect.objectContaining({
      model_path: '/m',
    }));
  });
});
