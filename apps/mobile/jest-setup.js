// Mock NativeModules for llama.rn and onnxruntime-react-native
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.NativeModules.LlamaContext = {
    initContext: jest.fn().mockResolvedValue(true),
    loadModel: jest.fn().mockResolvedValue('loaded'),
    unloadModel: jest.fn().mockResolvedValue(true),
    completion: jest.fn().mockResolvedValue('test response'),
    completionStream: jest.fn().mockResolvedValue(true),
    releaseContext: jest.fn().mockResolvedValue(true),
    tokenize: jest.fn().mockResolvedValue([1, 2, 3]),
  };
  RN.NativeModules.Onnxruntime = {
    loadModel: jest.fn().mockResolvedValue('loaded'),
    run: jest.fn().mockResolvedValue({output: [1, 2, 3]}),
    dispose: jest.fn().mockResolvedValue(true),
  };
  return RN;
});
