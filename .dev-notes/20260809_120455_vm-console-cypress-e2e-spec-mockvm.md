---
id: 20260809_120455_vm-console-cypress-e2e-spec-mockvm
title: VM console Cypress e2e spec + mockVm
status: done
tags: vm,e2e,frontend
created: 2026-08-09T12:04:55.473145+00:00
---

VM console Cypress e2e spec + mockVm

Added cy.mockVm() to cypress/support/api-mocks.ts (POST /vm/run, GET /vm/builtins, /vm/info, /vm/training/jobs/*) with runOverrides param, plus cypress.d.ts declaration. New cypress/e2e/vm-page.cy.ts with 6 tests: header + default user role, training samples in bar (hello/train/train-status), sample loads into editor, run shows Result/halted/Registers/EAX, permission-denied hint on EAX=-2 (via override intercept), Ref panel toggle. Verified tsc exit 0, vm page 30/30.