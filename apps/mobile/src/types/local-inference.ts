/** Configuration for a SloTransformer exported for on-device inference. */
export interface MobileExportConfig {
  vocab_size: number;
  n_embed: number;
  n_layer: number;
  n_head: number;
  block_size: number;
  num_weights: number;
}

/** Response from GET /auto-train/checkpoints/{name}/export-mobile */
export interface MobileExportResponse {
  config: MobileExportConfig;
  weights_b64: string;
}

/** State of a local inference engine (SloNet or llama.rn). */
export interface LocalModelState {
  kind: 'slonet' | 'qwen' | 'idle';
  loaded: boolean;
  modelName: string;
  /** Download progress 0-1 or null if not downloading */
  downloadProgress: number | null;
  /** Human-readable description */
  description: string;
}

/** Result from an on-device generate call. */
export interface LocalGenerateResult {
  text: string;
  tokens_generated: number;
  elapsed_ms: number;
}

/** Callback for streaming tokens during local inference. */
export type OnTokenCallback = (token: string) => void;

/** Which inference engine the user wants to use. */
export type ActiveEngine = 'slonet' | 'qwen' | 'remote';

/** Direction given to the hybrid router. */
export type RoutingDecision =
  | { target: 'local'; engine: 'slonet' | 'qwen' }
  | { target: 'remote'; reason: string };

export interface HybridState {
  slonet: LocalModelState;
  qwen: LocalModelState;
  activeEngine: ActiveEngine;
}
