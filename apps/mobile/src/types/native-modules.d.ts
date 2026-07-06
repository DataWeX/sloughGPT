/** llama.rn native module type declarations. */
declare module 'llama.rn' {
  export interface LlamaContextConfig {
    modelPath: string;
    nCtx?: number;
    nGpuLayers?: number;
  }

  export interface LlamaCompletionOptions {
    prompt: string;
    nPredict?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repeatPenalty?: number;
    stop?: string[];
  }

  export interface LlamaCompletionResult {
    text: string;
    tokens?: number;
  }

  export class LlamaContext {
    static create(config: LlamaContextConfig): Promise<LlamaContext>;
    completion(opts: LlamaCompletionOptions): Promise<LlamaCompletionResult>;
    completion(
      opts: LlamaCompletionOptions & {stream: true},
    ): AsyncGenerator<string, LlamaCompletionResult, void>;
    release(): Promise<void>;
  }
}

declare module 'onnxruntime-react-native' {
  export interface InferenceSessionOptions {
    executionProviders?: string[];
  }

  export interface InferenceSessionResult {
    [name: string]: {
      data: Float32Array;
      dims: number[];
    };
  }

  export class InferenceSession {
    static create(
      modelPath: string,
      options?: InferenceSessionOptions,
    ): Promise<InferenceSession>;
    run(
      feeds: Record<string, Float32Array>,
    ): Promise<InferenceSessionResult>;
    release(): Promise<void>;
  }
}
