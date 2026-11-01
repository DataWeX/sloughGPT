declare module 'expo-av' {
  export interface AVPlaybackStatus {
    isLoaded: boolean;
    didJustFinish?: boolean;
    isPlaying?: boolean;
    positionMillis?: number;
    durationMillis?: number;
  }

  export class Sound {
    static createAsync(
      source: {uri: string},
      initialStatus?: Record<string, any>,
    ): Promise<{sound: Sound; status: AVPlaybackStatus}>;
    playAsync(): Promise<AVPlaybackStatus>;
    pauseAsync(): Promise<AVPlaybackStatus>;
    stopAsync(): Promise<AVPlaybackStatus>;
    unloadAsync(): Promise<AVPlaybackStatus>;
    setOnPlaybackStatusUpdate(callback: (status: AVPlaybackStatus) => void): void;
  }

  export interface RecordingOptions {
    isMeteringEnabled?: boolean;
  }

  export class Recording {
    static requestPermissionsAsync(): Promise<{granted: boolean}>;
    prepareToRecordAsync(options?: RecordingOptions): Promise<void>;
    startAsync(): Promise<void>;
    stopAndUnloadAsync(): Promise<void>;
    getURI(): string | null;
  }

  export const RecordingOptionsPresets: {
    HIGH_QUALITY: RecordingOptions;
  };

  export const Audio: {
    Sound: typeof Sound;
    Recording: typeof Recording;
  };
}
