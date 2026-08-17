jest.mock('react-native', () => ({
  Alert: {alert: jest.fn()},
  Linking: {openSettings: jest.fn().mockResolvedValue(undefined)},
}));

// eslint-disable-next-line @typescript-eslint/no-var-requires
const {isTextFile, isImageFile, formatFileSize} = require('../file-upload');

describe('file-upload', () => {
  describe('isTextFile', () => {
    it('returns true for text/plain', () => {
      expect(isTextFile({name: 'file.txt', mimeType: 'text/plain', uri: '', size: 0})).toBe(true);
    });

    it('returns true for application/json', () => {
      expect(isTextFile({name: 'data.json', mimeType: 'application/json', uri: '', size: 0})).toBe(true);
    });

    it('returns true for .py extension', () => {
      expect(isTextFile({name: 'script.py', mimeType: 'application/octet-stream', uri: '', size: 0})).toBe(true);
    });

    it('returns true for .js extension', () => {
      expect(isTextFile({name: 'app.js', mimeType: 'application/octet-stream', uri: '', size: 0})).toBe(true);
    });

    it('returns true for .md extension', () => {
      expect(isTextFile({name: 'readme.md', mimeType: 'application/octet-stream', uri: '', size: 0})).toBe(true);
    });

    it('returns false for image/jpeg', () => {
      expect(isTextFile({name: 'photo.jpg', mimeType: 'image/jpeg', uri: '', size: 0})).toBe(false);
    });

    it('returns false for video/mp4', () => {
      expect(isTextFile({name: 'video.mp4', mimeType: 'video/mp4', uri: '', size: 0})).toBe(false);
    });

    it('returns false for unknown extension', () => {
      expect(isTextFile({name: 'file.xyz', mimeType: 'application/octet-stream', uri: '', size: 0})).toBe(false);
    });
  });

  describe('isImageFile', () => {
    it('returns true for image/jpeg', () => {
      expect(isImageFile({name: 'photo.jpg', mimeType: 'image/jpeg', uri: '', size: 0})).toBe(true);
    });

    it('returns true for image/png', () => {
      expect(isImageFile({name: 'logo.png', mimeType: 'image/png', uri: '', size: 0})).toBe(true);
    });

    it('returns true for image/gif', () => {
      expect(isImageFile({name: 'anim.gif', mimeType: 'image/gif', uri: '', size: 0})).toBe(true);
    });

    it('returns false for text/plain', () => {
      expect(isImageFile({name: 'file.txt', mimeType: 'text/plain', uri: '', size: 0})).toBe(false);
    });

    it('returns false for application/pdf', () => {
      expect(isImageFile({name: 'doc.pdf', mimeType: 'application/pdf', uri: '', size: 0})).toBe(false);
    });
  });

  describe('formatFileSize', () => {
    it('formats bytes', () => {
      expect(formatFileSize(500)).toBe('500 B');
    });

    it('formats kilobytes', () => {
      expect(formatFileSize(1536)).toBe('1.5 KB');
    });

    it('formats megabytes', () => {
      expect(formatFileSize(2621440)).toBe('2.5 MB');
    });

    it('formats exact 1KB', () => {
      expect(formatFileSize(1024)).toBe('1.0 KB');
    });

    it('formats exact 1MB', () => {
      expect(formatFileSize(1048576)).toBe('1.0 MB');
    });

    it('formats zero', () => {
      expect(formatFileSize(0)).toBe('0 B');
    });
  });
});
