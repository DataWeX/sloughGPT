import AsyncStorage from '@react-native-async-storage/async-storage';
import {api} from '../../services/api-client';

jest.mock('../../services/api-client');
jest.mock('../../services/haptics');
jest.mock('../../services/sounds');
jest.mock('../../services/toast');

const mockApi = api as jest.Mocked<typeof api>;

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {useModelStore} = require('../model-store');

const INITIAL = {
  models: [],
  currentModel: null,
  souls: [],
  currentSoul: null,
  checkpoints: [],
  health: null,
  loading: false,
  loadingModelId: null,
  error: null,
};

beforeEach(async () => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  useModelStore.setState(INITIAL);
  await new Promise(r => setTimeout(r, 5));
});

describe('model-store', () => {
  describe('initial state', () => {
    it('starts with empty model list', () => {
      const s = useModelStore.getState();
      expect(s.models).toEqual([]);
      expect(s.loading).toBe(false);
      expect(s.error).toBeNull();
    });
  });

  describe('refresh', () => {
    it('fetches all data in parallel', async () => {
      const models = [{id: 'gpt2', name: 'GPT-2'}];
      const soulsData = {souls: [{name: 'friendly', id: 'friendly'}], current_soul: 'friendly'};
      const currentSoul = {name: 'friendly', id: 'friendly'};
      const checkpoints = [{name: 'cp1'}];
      const health = {model_type: 'gpt2', status: 'healthy'};

      mockApi.get
        .mockResolvedValueOnce(models)    // /models
        .mockResolvedValueOnce(soulsData)     // /souls
        .mockResolvedValueOnce(currentSoul) // /souls/current
        .mockResolvedValueOnce(checkpoints) // /auto-train/checkpoints
        .mockResolvedValueOnce(health);   // /health

      await useModelStore.getState().refresh();
      const s = useModelStore.getState();
      expect(s.models).toEqual(models);
      expect(s.souls).toEqual(soulsData.souls);
      expect(s.currentSoul).toEqual(currentSoul);
      expect(s.checkpoints).toEqual(checkpoints);
      expect(s.currentModel).toBe('gpt2');
      expect(s.loading).toBe(false);
    });

    it('gracefully handles individual API failures', async () => {
      mockApi.get
        .mockRejectedValueOnce(new Error('failed'))
        .mockRejectedValueOnce(new Error('failed'))
        .mockRejectedValueOnce(new Error('failed'))
        .mockRejectedValueOnce(new Error('failed'))
        .mockRejectedValueOnce(new Error('failed'));

      await useModelStore.getState().refresh();
      const s = useModelStore.getState();
      expect(s.models).toEqual([]);
      expect(s.souls).toEqual([]);
      expect(s.loading).toBe(false);
    });

    it('sets loading during fetch', async () => {
      mockApi.get
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce(null)
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce(null);

      const promise = useModelStore.getState().refresh();
      expect(useModelStore.getState().loading).toBe(true);
      await promise;
      expect(useModelStore.getState().loading).toBe(false);
    });
  });

  describe('loadModel', () => {
    it('POSTs model_id and refreshes', async () => {
      mockApi.post.mockResolvedValue(undefined);
      mockApi.get
        .mockResolvedValue([])
        .mockResolvedValue([])
        .mockResolvedValue(null)
        .mockResolvedValue([])
        .mockResolvedValue(null);

      await useModelStore.getState().loadModel('gpt2');
      expect(mockApi.post).toHaveBeenCalledWith('/models/load', {model_id: 'gpt2'});
    });

    it('sets loadingModelId during load', async () => {
      mockApi.post.mockImplementation(() => new Promise(r => setTimeout(r, 50)));
      mockApi.get
        .mockResolvedValue([])
        .mockResolvedValue([])
        .mockResolvedValue(null)
        .mockResolvedValue([])
        .mockResolvedValue(null);

      const promise = useModelStore.getState().loadModel('gpt2');
      expect(useModelStore.getState().loadingModelId).toBe('gpt2');
      await promise;
    });

    it('sets error on failure', async () => {
      mockApi.post.mockRejectedValue(new Error('Model not found'));
      await useModelStore.getState().loadModel('bad-model');
      expect(useModelStore.getState().error).toBe('Model not found');
      expect(useModelStore.getState().loadingModelId).toBeNull();
    });
  });

  describe('unloadModel', () => {
    it('POSTs to unload and refreshes', async () => {
      mockApi.post.mockResolvedValue(undefined);
      mockApi.get
        .mockResolvedValue([])
        .mockResolvedValue([])
        .mockResolvedValue(null)
        .mockResolvedValue([])
        .mockResolvedValue(null);

      await useModelStore.getState().unloadModel();
      expect(mockApi.post).toHaveBeenCalledWith('/models/unload');
    });
  });

  describe('switchSoul', () => {
    it('POSTs soul name and checkpoint', async () => {
      mockApi.post.mockResolvedValue(undefined);
      mockApi.get
        .mockResolvedValue([])
        .mockResolvedValue([])
        .mockResolvedValue(null)
        .mockResolvedValue([])
        .mockResolvedValue(null);

      await useModelStore.getState().switchSoul('friendly', 'cp1');
      expect(mockApi.post).toHaveBeenCalledWith('/souls/switch', {
        soul: 'friendly',
        checkpoint_name: 'cp1',
      });
    });

    it('works without checkpoint', async () => {
      mockApi.post.mockResolvedValue(undefined);
      mockApi.get
        .mockResolvedValue([])
        .mockResolvedValue([])
        .mockResolvedValue(null)
        .mockResolvedValue([])
        .mockResolvedValue(null);

      await useModelStore.getState().switchSoul('friendly');
      expect(mockApi.post).toHaveBeenCalledWith('/souls/switch', {
        soul: 'friendly',
        checkpoint_name: null,
      });
    });

    it('sets error on failure', async () => {
      mockApi.post.mockRejectedValue(new Error('Soul not found'));
      await useModelStore.getState().switchSoul('missing');
      expect(useModelStore.getState().error).toBe('Soul not found');
    });
  });

  describe('clearError', () => {
    it('clears the error field', () => {
      useModelStore.setState({error: 'something'});
      useModelStore.getState().clearError();
      expect(useModelStore.getState().error).toBeNull();
    });
  });
});
