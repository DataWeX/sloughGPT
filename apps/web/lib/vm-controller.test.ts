import { describe, it, expect } from 'vitest'
import { setupApiMocks, mockGet, mockPost, apiClient } from './__test-helper'
import { vmController } from './vm-controller'

setupApiMocks()

describe('vmController', () => {
  describe('run', () => {
    it('calls POST /vm/run with source and default opts', async () => {
      const fakeResult = {
        success: true,
        exit_code: 0,
        steps_executed: 10,
        elapsed_ms: 1.5,
        output: '',
        registers: [{ name: 'EAX', value: 42, hex: '0x0000002A' }],
        eip: 0,
        eip_hex: '0x00000000',
        status: 'halted',
      }
      mockPost('/vm/run', fakeResult)

      const result = await vmController.run('MOV EAX, 42\nHLT')

      expect(apiClient.apiPost).toHaveBeenCalledWith('/vm/run', {
        source: 'MOV EAX, 42\nHLT',
        max_steps: 5000,
        role: 'user',
        debug: false,
        keyboard_input: undefined,
      })
      expect(result).toEqual(fakeResult)
      expect(result.success).toBe(true)
      expect(result.registers).toHaveLength(1)
    })

    it('passes custom opts', async () => {
      mockPost('/vm/run', { success: true })

      await vmController.run('HLT', {
        maxSteps: 100,
        role: 'admin',
        debug: true,
        keyboardInput: 'abc',
      })

      expect(apiClient.apiPost).toHaveBeenCalledWith('/vm/run', {
        source: 'HLT',
        max_steps: 100,
        role: 'admin',
        debug: true,
        keyboard_input: 'abc',
      })
    })

    it('returns error result on failure', async () => {
      const failResult = {
        success: false,
        exit_code: -1,
        steps_executed: 0,
        elapsed_ms: 0,
        output: '',
        registers: [],
        eip: 0,
        eip_hex: '0x0',
        status: 'error',
        error: 'assembly error',
      }
      mockPost('/vm/run', failResult)

      const result = await vmController.run('BAD INSTRUCTION')
      expect(result.success).toBe(false)
      expect(result.error).toBe('assembly error')
    })

    it('passes training_result through from the response', async () => {
      const fakeResult = {
        success: true,
        exit_code: 0,
        steps_executed: 10,
        elapsed_ms: 1.5,
        output: '',
        registers: [],
        eip: 0,
        eip_hex: '0x0',
        status: 'halted',
        training_job_id: 1,
        training_result: '{"success": true, "final_loss": 1.5}',
      }
      mockPost('/vm/run', fakeResult)

      const result = await vmController.run('MOV EAX, 30\nINT 0x80\nHLT')
      expect(result.training_job_id).toBe(1)
      expect(result.training_result).toContain('final_loss')
    })
  })

  describe('builtins', () => {
    it('calls GET /vm/builtins', async () => {
      const fakeBuiltins = {
        programs: [
          { name: 'hello', description: 'Hello world' },
          { name: 'count', description: 'Count to 10' },
        ],
      }
      mockGet('/vm/builtins', fakeBuiltins)

      const result = await vmController.builtins()
      expect(apiClient.apiGet).toHaveBeenCalledWith('/vm/builtins')
      expect(result.programs).toHaveLength(2)
      expect(result.programs[0].name).toBe('hello')
    })
  })

  describe('info', () => {
    it('calls GET /vm/info', async () => {
      const fakeInfo = {
        isa: 'x86-32',
        max_steps: 1000000,
        default_memory: 0x100000,
        max_memory: 0x1000000,
        registers: ['EAX', 'ECX'],
        features: ['protected mode'],
      }
      mockGet('/vm/info', fakeInfo)

      const result = await vmController.info()
      expect(apiClient.apiGet).toHaveBeenCalledWith('/vm/info')
      expect(result.isa).toBe('x86-32')
      expect(result.registers).toContain('EAX')
      expect(result.features).toContain('protected mode')
    })
  })

  describe('trainingJob', () => {
    it('calls GET /vm/training/jobs/{id}', async () => {
      const fakeJob = {
        job_id: 7,
        api_job_id: 'abc-123',
        status: 'running',
        progress: 0.42,
        error: null,
      }
      mockGet('/vm/training/jobs/7', fakeJob)

      const result = await vmController.trainingJob(7)
      expect(apiClient.apiGet).toHaveBeenCalledWith('/vm/training/jobs/7')
      expect(result).toEqual(fakeJob)
      expect(result.status).toBe('running')
      expect(result.job_id).toBe(7)
    })

    it('passes completed job with error field through', async () => {
      const fakeJob = {
        job_id: 3,
        api_job_id: 'done-9',
        status: 'failed',
        progress: 0.5,
        error: 'dataset not found',
      }
      mockGet('/vm/training/jobs/3', fakeJob)

      const result = await vmController.trainingJob(3)
      expect(result.status).toBe('failed')
      expect(result.error).toBe('dataset not found')
    })

    it('passes the result JSON through on completed jobs', async () => {
      const fakeJob = {
        job_id: 3,
        api_job_id: 'done-9',
        status: 'completed',
        progress: 1,
        result: '{"success": true, "final_loss": 1.2}',
      }
      mockGet('/vm/training/jobs/3', fakeJob)

      const result = await vmController.trainingJob(3)
      expect(result.status).toBe('completed')
      expect(result.result).toContain('final_loss')
    })

    it('posts to the stop endpoint', async () => {
      mockPost('/vm/training/jobs/7/stop', { status: 'stopping', job_id: 7 })

      const result = await vmController.stopTrainingJob(7)
      expect(apiClient.apiPost).toHaveBeenCalledWith('/vm/training/jobs/7/stop', {})
      expect(result.status).toBe('stopping')
      expect(result.job_id).toBe(7)
    })
  })
})
