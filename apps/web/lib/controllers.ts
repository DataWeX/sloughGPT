/**
 * Frontend Controllers Index
 *
 * Single source of truth for all MVC controllers.
 * Only re-exports controllers that are actually imported from this barrel.
 */

export { modelController } from './model-controller'
export { trainingJobsController } from './training-controller'
export { trainingController } from './training-controller'
export { sessionController } from './session-controller'
export { datasetController } from './dataset-controller'
export { agentsController } from './agents-controller'
export { soulsController } from './souls-controller'
export { multimodalController } from './multimodal-controller'
export { errorsController } from './errors-controller'
export { experimentsController } from './experiments-controller'
export { workflowController } from './workflow-controller'
export { filesController } from './files-controller'
export { voiceController } from './voice-controller'
export { registryController } from './registry-controller'
export { selfTrainController } from './self-train-controller'
export { loraEvalController } from './lora-eval-controller'

export type { ModelInfo, ModelStatus, ModelLoadResponse, HealthStatus } from './model-controller'
export type { TrainingJob, TrainingStatus, RecoverableJob, Webhook, WebhookStats } from './training-controller'
export type { Session, Conversation } from './session-controller'
export type { Dataset, DatasetPreview, ImportResponse, ImportSource, GitHubRepo, BookResult } from './dataset-controller'
export type { Soul, Checkpoint, SoulsResponse, CheckpointsResponse } from './souls-controller'
export type { MultimodalCapabilities, TrainingReport } from './multimodal-controller'
export type { GroupedError, TrendPoint, RecentError } from './errors-controller'
export type { Experiment } from './experiments-controller'
export type { WorkflowStatus, WorkflowConfig, WorkflowStats } from './workflow-controller'
export type { FileEntry, SearchResult as FilesSearchResult } from './files-controller'
export type { VoiceStatus, TTSResult } from './voice-controller'
export type { RegisteredModel, RegistryStats } from './registry-controller'
export type { SelfTrainStatus } from './self-train-controller'
export type { LoraEvalResult, LoraEvalHistory } from './lora-eval-controller'
