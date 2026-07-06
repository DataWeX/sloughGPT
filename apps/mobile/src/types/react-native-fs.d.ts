declare module 'react-native-fs' {
  const RNFS: {
    DocumentDirectoryPath: string;
    CachesDirectoryPath: string;
    exists(path: string): Promise<boolean>;
    downloadFile(options: {
      fromUrl: string;
      toFile: string;
      headers?: Record<string, string>;
      progress?: (res: {contentLength: number; bytesWritten: number}) => void;
      progressDivider?: number;
      begin?: (res: {statusCode: number; contentLength: number}) => void;
    }): {
      promise: Promise<{statusCode: number; bytesWritten: number}>;
    };
    readDir(path: string): Promise<Array<{name: string; path: string; size: number; isFile: () => boolean; isDirectory: () => boolean}>>;
    unlink(path: string): Promise<void>;
    writeFile(path: string, content: string): Promise<void>;
    readFile(path: string): Promise<string>;
    moveFile(from: string, to: string): Promise<void>;
    copyFile(from: string, to: string): Promise<void>;
    mkdir(path: string): Promise<void>;
  };
  export default RNFS;
}
