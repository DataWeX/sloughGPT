const RNFS = {
  DocumentDirectoryPath: '/mock/documents',
  CachesDirectoryPath: '/mock/cache',
  downloadFile: jest.fn().mockResolvedValue({statusCode: 200}),
  readDir: jest.fn().mockResolvedValue([]),
  readFile: jest.fn().mockResolvedValue(''),
  writeFile: jest.fn().mockResolvedValue(undefined),
  unlink: jest.fn().mockResolvedValue(undefined),
  exists: jest.fn().mockResolvedValue(true),
  mkdir: jest.fn().mockResolvedValue(undefined),
  moveFile: jest.fn().mockResolvedValue(undefined),
  copyFile: jest.fn().mockResolvedValue(undefined),
  read: jest.fn().mockResolvedValue(''),
  getAllExternalFilesDirs: jest.fn().mockResolvedValue([]),
  appendFile: jest.fn().mockResolvedValue(undefined),
  getFSInfo: jest.fn().mockResolvedValue({
    totalSpace: 64000000000,
    freeSpace: 32000000000,
  }),
};
export default RNFS;
