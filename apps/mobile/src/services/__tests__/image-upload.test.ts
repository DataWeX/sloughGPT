jest.mock('react-native', () => ({
  Alert: {alert: jest.fn()},
  Linking: {openSettings: jest.fn().mockResolvedValue(undefined)},
  Platform: {OS: 'ios'},
}));

jest.mock('../api-client', () => ({
  api: {post: jest.fn()},
  getApiUrl: jest.fn(() => 'http://localhost:8000'),
}));

const mockLaunchImageLibrary = jest.fn();
const mockRequestMediaLibraryPermissions = jest.fn();
const mockRequestCameraPermissions = jest.fn();
const mockLaunchCamera = jest.fn();

jest.mock('expo-image-picker', () => ({
  launchImageLibraryAsync: (...args: any[]) => mockLaunchImageLibrary(...args),
  requestMediaLibraryPermissionsAsync: (...args: any[]) => mockRequestMediaLibraryPermissions(...args),
  requestCameraPermissionsAsync: (...args: any[]) => mockRequestCameraPermissions(...args),
  launchCameraAsync: (...args: any[]) => mockLaunchCamera(...args),
}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {pickImage, takePhoto, analyzeImage} = require('../image-upload');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const {Alert} = require('react-native');

beforeEach(() => {
  jest.clearAllMocks();
  mockRequestMediaLibraryPermissions.mockResolvedValue({granted: true});
  mockRequestCameraPermissions.mockResolvedValue({granted: true});
  mockLaunchImageLibrary.mockResolvedValue({
    canceled: false,
    assets: [{uri: 'file:///test.jpg', base64: 'base64data', width: 100, height: 100}],
  });
  mockLaunchCamera.mockResolvedValue({
    canceled: false,
    assets: [{uri: 'file:///photo.jpg', base64: 'base64photo', width: 200, height: 200}],
  });
});

describe('image-upload', () => {
  describe('pickImage', () => {
    it('returns image result on success', async () => {
      const result = await pickImage();
      expect(result).not.toBeNull();
      expect(result!.uri).toBe('file:///test.jpg');
      expect(result!.base64).toBe('base64data');
      expect(result!.width).toBe(100);
      expect(result!.height).toBe(100);
    });

    it('returns null when permission denied', async () => {
      mockRequestMediaLibraryPermissions.mockResolvedValue({granted: false});
      const result = await pickImage();
      expect(result).toBeNull();
      expect(Alert.alert).toHaveBeenCalledWith(
        'Photo library permission required',
        expect.any(String),
        expect.any(Array),
      );
    });

    it('returns null when canceled', async () => {
      mockLaunchImageLibrary.mockResolvedValue({canceled: true, assets: []});
      const result = await pickImage();
      expect(result).toBeNull();
    });
  });

  describe('takePhoto', () => {
    it('returns photo result on success', async () => {
      const result = await takePhoto();
      expect(result).not.toBeNull();
      expect(result!.uri).toBe('file:///photo.jpg');
      expect(result!.base64).toBe('base64photo');
      expect(result!.width).toBe(200);
      expect(result!.height).toBe(200);
    });

    it('returns null when camera permission denied', async () => {
      mockRequestCameraPermissions.mockResolvedValue({granted: false});
      const result = await takePhoto();
      expect(result).toBeNull();
      expect(Alert.alert).toHaveBeenCalledWith(
        'Camera permission required',
        expect.any(String),
        expect.any(Array),
      );
    });

    it('returns null when canceled', async () => {
      mockLaunchCamera.mockResolvedValue({canceled: true, assets: []});
      const result = await takePhoto();
      expect(result).toBeNull();
    });
  });

  describe('analyzeImage', () => {
    it('returns analysis result', async () => {
      global.fetch = jest.fn().mockResolvedValue({
        ok: true,
        json: jest.fn().mockResolvedValue({description: 'A test image', tags: ['test'], caption: 'Test caption'}),
      });
      const result = await analyzeImage({uri: 'file:///test.jpg', base64: 'base64data', width: 100, height: 100});
      expect(result).not.toBeNull();
      expect(result!.description).toBe('A test image');
      expect(result!.tags).toEqual(['test']);
      expect(result!.caption).toBe('Test caption');
    });

    it('returns empty on fetch failure', async () => {
      global.fetch = jest.fn().mockRejectedValue(new Error('network'));
      const result = await analyzeImage({uri: 'file:///test.jpg', base64: 'base64data', width: 100, height: 100});
      expect(result!.description).toBe('');
    });
  });
});
