/**
 * Unified training facade — one entry point for all training features.
 *
 * Groups the existing controllers (no logic moved, no behavior changed):
 * - jobs / turbo / checkpoints → trainingJobsController
 * - automation → auto-train settings + self-train controllers
 * - evaluation → lora-eval controller
 *
 * Bindings are lazy getters so the facade never breaks at import time.
 * New code should import { trainingFacade } instead of reaching for the
 * individual controllers directly.
 */
import { loraEvalController } from './lora-eval-controller'
import { selfTrainController } from './self-train-controller'
import { settingsController } from './settings-controller'
import { trainingJobsController } from './training-controller'

const jobsCtl = trainingJobsController
const autoCtl = settingsController
const selfCtl = selfTrainController
const evalCtl = loraEvalController

export const trainingFacade = {
  jobs: {
    get list() {
      return jobsCtl.list.bind(jobsCtl)
    },
    get get() {
      return jobsCtl.get.bind(jobsCtl)
    },
    get create() {
      return jobsCtl.create.bind(jobsCtl)
    },
    get startAutoTrain() {
      return jobsCtl.startAutoTrain.bind(jobsCtl)
    },
    get stopAutoTrain() {
      return jobsCtl.stopAutoTrain.bind(jobsCtl)
    },
    get stop() {
      return jobsCtl.stop.bind(jobsCtl)
    },
    get delete() {
      return jobsCtl.delete.bind(jobsCtl)
    },
    get purge() {
      return jobsCtl.purgeJobs.bind(jobsCtl)
    },
    get recoverable() {
      return jobsCtl.recoverable.bind(jobsCtl)
    },
    get recover() {
      return jobsCtl.recover.bind(jobsCtl)
    },
    get summary() {
      return jobsCtl.getSummary.bind(jobsCtl)
    },
    get status() {
      return jobsCtl.getStatus.bind(jobsCtl)
    },
    get download() {
      return jobsCtl.downloadTrainingJob.bind(jobsCtl)
    },
    get finetune() {
      return jobsCtl.startLoraFinetune.bind(jobsCtl)
    },
    get visualTrain() {
      return jobsCtl.startVisualTrain.bind(jobsCtl)
    },
  },
  turbo: {
    get start() {
      return jobsCtl.startTurboTrain.bind(jobsCtl)
    },
    get status() {
      return jobsCtl.getTurboStatus.bind(jobsCtl)
    },
    get pause() {
      return jobsCtl.pauseTraining.bind(jobsCtl)
    },
    get resume() {
      return jobsCtl.resumeTraining.bind(jobsCtl)
    },
    get stop() {
      return jobsCtl.stopAutoTrain.bind(jobsCtl)
    },
    get loadCheckpoint() {
      return jobsCtl.loadCheckpoint.bind(jobsCtl)
    },
  },
  checkpoints: {
    get list() {
      return jobsCtl.listCheckpoints.bind(jobsCtl)
    },
    get info() {
      return jobsCtl.getCheckpointInfo.bind(jobsCtl)
    },
    get load() {
      return jobsCtl.loadCheckpoint.bind(jobsCtl)
    },
    get remove() {
      return jobsCtl.deleteCheckpoint.bind(jobsCtl)
    },
    get removeBatch() {
      return jobsCtl.deleteCheckpointsBatch.bind(jobsCtl)
    },
    get download() {
      return jobsCtl.downloadCheckpoint.bind(jobsCtl)
    },
    get builds() {
      return jobsCtl.listBuilds.bind(jobsCtl)
    },
  },
  automation: {
    get autoTrainStatus() {
      return autoCtl.getAutoTrainSettingsStatus.bind(autoCtl)
    },
    get updateAutoTrainConfig() {
      return autoCtl.updateAutoTrainSettingsConfig.bind(autoCtl)
    },
    get selfTrainStatus() {
      return selfCtl.getStatus.bind(selfCtl)
    },
    get startSelfTrain() {
      return selfCtl.start.bind(selfCtl)
    },
    get stopSelfTrain() {
      return selfCtl.stop.bind(selfCtl)
    },
  },
  evaluation: {
    get run() {
      return evalCtl.runEval.bind(evalCtl)
    },
    get history() {
      return evalCtl.getHistory.bind(evalCtl)
    },
    get aggregate() {
      return evalCtl.aggregate.bind(evalCtl)
    },
  },
}

export type TrainingFacade = typeof trainingFacade
