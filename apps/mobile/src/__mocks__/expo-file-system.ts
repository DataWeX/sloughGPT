const readAsStringAsync = jest.fn().mockResolvedValue('');
const writeAsStringAsync = jest.fn().mockResolvedValue(undefined);
const deleteAsync = jest.fn().mockResolvedValue(undefined);
const getInfoAsync = jest.fn().mockResolvedValue({ exists: false });
const readDirectoryAsync = jest.fn().mockResolvedValue([]);
const makeDirectoryAsync = jest.fn().mockResolvedValue(undefined);
const documentDirectory = 'file:///mock-documents/';
const cacheDirectory = 'file:///mock-cache/';

export default {
  readAsStringAsync,
  writeAsStringAsync,
  deleteAsync,
  getInfoAsync,
  readDirectoryAsync,
  makeDirectoryAsync,
  documentDirectory,
  cacheDirectory,
};

export {
  readAsStringAsync,
  writeAsStringAsync,
  deleteAsync,
  getInfoAsync,
  readDirectoryAsync,
  makeDirectoryAsync,
  documentDirectory,
  cacheDirectory,
};
