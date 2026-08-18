const mockRecording = jest.fn().mockImplementation(() => ({
  prepareToRecordAsync: jest.fn().mockResolvedValue(undefined),
  startAsync: jest.fn().mockResolvedValue(undefined),
  stopAndUnloadAsync: jest.fn().mockResolvedValue(undefined),
  getURI: jest.fn().mockReturnValue('file:///rec.m4a'),
}));

module.exports = {
  Recording: mockRecording,
  RecordingOptionsPresets: {HIGH_QUALITY: {}},
};
