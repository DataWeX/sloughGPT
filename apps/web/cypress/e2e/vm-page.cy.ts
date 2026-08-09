describe('VM console', () => {
  beforeEach(() => {
    cy.mockAll()
    cy.mockVm()
    cy.visit('/vm')
  })

  it('renders the console header and default role', () => {
    cy.contains('h1', 'VM Console').should('be.visible')
    cy.get('select[aria-label="VM role"]').should('have.value', 'user')
  })

  it('offers the training sample programs in the sample bar', () => {
    cy.contains('button', 'hello').should('be.visible')
    cy.contains('button', 'train').should('be.visible')
    cy.contains('button', 'train-status').should('be.visible')
  })

  it('loads the selected sample into the editor', () => {
    cy.contains('button', 'hello').click()
    cy.get('textarea').should('contain.value', 'Hello, VM!')
  })

  it('runs the source and shows the result and registers', () => {
    cy.contains('button', 'Run').click()
    cy.contains('Result').scrollIntoView().should('be.visible')
    cy.contains('halted').should('be.visible')
    cy.contains('Registers').scrollIntoView().should('be.visible')
    cy.contains('EAX').should('be.visible')
  })

  it('shows the permission denied hint when a syscall returns EAX -2', () => {
    cy.intercept('POST', 'http://localhost:8000/vm/run', {
      statusCode: 200,
      body: {
        success: true,
        exit_code: 1,
        steps_executed: 4,
        elapsed_ms: 1.0,
        output: '',
        registers: [
          { name: 'EAX', value: -2, hex: '0xfffffffe' },
          { name: 'ECX', value: 0, hex: '0x00000000' },
        ],
        eip: 0,
        eip_hex: '0x00000000',
        status: 'halted',
      },
    }).as('vmRunDenied')
    cy.contains('button', 'Run').click()
    cy.contains(/A syscall was denied for the current role/).scrollIntoView().should('be.visible')
    cy.contains(/require the(?: admin)? role/).should('be.visible')
  })

  it('shows the training result JSON card when the run returns one', () => {
    cy.intercept('POST', 'http://localhost:8000/vm/run', {
      statusCode: 200,
      body: {
        success: true,
        exit_code: 0,
        steps_executed: 12,
        elapsed_ms: 3.5,
        output: '',
        registers: [{ name: 'EAX', value: 1, hex: '0x00000001' }],
        eip: 0,
        eip_hex: '0x00000000',
        status: 'halted',
        training_job_id: 1,
        training_result: '{"success": true, "final_loss": 1.5}',
      },
    }).as('vmRunResult')
    cy.contains('button', 'Run').click()
    cy.contains('Training result').scrollIntoView().should('be.visible')
    cy.contains(/final_loss.*1\.5/).should('be.visible')
  })

  it('stops a running training job from the Training card', () => {
    cy.intercept('POST', 'http://localhost:8000/vm/run', {
      statusCode: 200,
      body: {
        success: true,
        exit_code: 0,
        steps_executed: 12,
        elapsed_ms: 3.5,
        output: '',
        registers: [{ name: 'EAX', value: 1, hex: '0x00000001' }],
        eip: 0,
        eip_hex: '0x00000000',
        status: 'halted',
        training_job_id: 1,
      },
    }).as('vmRunTrain')
    cy.intercept('GET', 'http://localhost:8000/vm/training/jobs/1', {
      statusCode: 200,
      body: { job_id: 1, api_job_id: 'abc-123', status: 'running', progress: 0.5, result: null },
    }).as('vmTrainingRunning')
    cy.intercept('POST', 'http://localhost:8000/vm/training/jobs/1/stop', {
      statusCode: 200,
      body: { status: 'stopping', job_id: 1 },
    }).as('vmTrainingStop')

    cy.contains('button', 'Run').click()
    cy.contains('button', 'Stop').scrollIntoView().should('be.visible')
    cy.contains('button', 'Stop').click()
    cy.wait('@vmTrainingStop')
  })

  it('persists the selected role and steps across reloads', () => {
    cy.get('select[aria-label="VM role"]').select('admin')
    cy.get('#vm-steps').type('{selectall}750')
    cy.reload()
    cy.get('select[aria-label="VM role"]').should('have.value', 'admin')
    cy.get('#vm-steps').should('have.value', '750')
  })

  it('resets the training launch config to defaults', () => {
    cy.get('input[aria-label="Training epochs"]').clear().type('42')
    cy.get('input[aria-label="Training learning rate"]').clear().type('0.5')
    cy.contains('button', 'Reset config').click()
    cy.get('input[aria-label="Training epochs"]').should('have.value', '1')
    cy.get('input[aria-label="Training learning rate"]').should('have.value', '0.001')
  })

  it('warns and switches to admin on the training launch card', () => {
    cy.contains(/Training is denied for the user role/).scrollIntoView().should('be.visible')
    cy.contains('button', 'Switch to admin').click()
    cy.get('select[aria-label="VM role"]').should('have.value', 'admin')
    cy.contains(/Training is denied for the user role/).should('not.exist')
  })

  it('shows fallback hints when config fields are cleared', () => {
    cy.get('input[aria-label="Training epochs"]').clear()
    cy.contains(/Epochs: using default 1/).should('be.visible')
    cy.get('input[aria-label="Training epochs"]').type('3')
    cy.contains(/Epochs: using default 1/).should('not.exist')
  })

  it('toggles the x86 reference panel', () => {
    cy.contains('button', 'Ref').click()
    cy.contains('x86 Reference').scrollIntoView().should('be.visible')
    cy.contains('button', 'Ref').click()
    cy.contains('x86 Reference').should('not.exist')
  })

  it('launches a training job from the Training launch card with a dataset dropdown', () => {
    cy.intercept('GET', 'http://localhost:8000/datasets', {
      statusCode: 200,
      body: {
        datasets: [
          { name: 'shakespeare', source: 'local', size: 1 },
          { name: 'tinyshakespeare', source: 'local', size: 1 },
        ],
      },
    }).as('datasetsList')
    cy.reload()
    cy.wait('@datasetsList')
    cy.get('select[aria-label="Training dataset"]').should('have.value', 'shakespeare')

    cy.intercept('POST', 'http://localhost:8000/vm/run', (req) => {
      expect(req.body.source).to.contain('SYS_TRAIN_START')
      expect(req.body.source).to.contain('"dataset":"tinyshakespeare","epochs":2')
      req.reply({
        statusCode: 200,
        body: {
          success: true,
          exit_code: 0,
          steps_executed: 12,
          elapsed_ms: 3.5,
          output: '',
          registers: [{ name: 'EAX', value: 1, hex: '0x00000001' }],
          eip: 0,
          eip_hex: '0x00000000',
          status: 'halted',
          training_job_id: 1,
        },
      })
    }).as('vmRunLaunch')
    cy.intercept('GET', 'http://localhost:8000/vm/training/jobs/1', {
      statusCode: 200,
      body: {
        job_id: 1,
        api_job_id: 'abc-123',
        status: 'completed',
        progress: 1,
        result: '{"success": true, "final_loss": 0.9}',
      },
    }).as('vmLaunchJob')

    cy.get('select[aria-label="Training dataset"]').select('tinyshakespeare')
    cy.get('input[aria-label="Training epochs"]').clear().type('2')
    cy.contains('button', 'Launch training').click()
    cy.wait('@vmRunLaunch')
    cy.contains('button', 'Load sample').click()
    cy.get('textarea').should('contain.value', '"dataset":"tinyshakespeare","epochs":2')
  })

  it('confirms a launched training job and dismisses the note', () => {
    cy.get('select[aria-label="VM role"]').select('admin')
    cy.intercept('POST', 'http://localhost:8000/vm/run', (req) => {
      expect(req.body.source).to.contain('SYS_TRAIN_START')
      req.reply({
        statusCode: 200,
        body: {
          success: true,
          exit_code: 0,
          steps_executed: 8,
          elapsed_ms: 2.1,
          output: '',
          registers: [{ name: 'EAX', value: 5, hex: '0x00000005' }],
          eip: 0,
          eip_hex: '0x00000000',
          status: 'halted',
        },
      })
    }).as('vmRunLaunchConfirm')

    cy.contains('button', 'Launch training').click()
    cy.wait('@vmRunLaunchConfirm')
    cy.contains(/Launched training job #5/).scrollIntoView().should('be.visible')
    cy.contains('button', 'Dismiss').click()
    cy.contains(/Launched training job #5/).should('not.exist')
  })

  it('warns when a custom dataset is unknown', () => {
    cy.intercept('GET', 'http://localhost:8000/datasets', {
      statusCode: 200,
      body: {
        datasets: [
          { id: 'ds1', name: 'shakespeare', source: 'local', size: 1 },
          { id: 'ds2', name: 'tinyshakespeare', source: 'local', size: 1 },
        ],
      },
    }).as('datasetsFull')
    cy.reload()
    cy.wait('@datasetsFull')

    cy.get('select[aria-label="Training dataset"]').select('__custom__')
    cy.get('input[aria-label="Training dataset"]').clear().type('nope')
    cy.contains(/Unknown dataset "nope"/).scrollIntoView().should('be.visible')
    cy.contains(/Available: shakespeare/).should('be.visible')
  })
})
