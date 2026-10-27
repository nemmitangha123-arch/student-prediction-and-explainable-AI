import assert from 'node:assert/strict'
import test from 'node:test'

import {
  accountStatusLabel,
  requestStudentAccountStatusChange,
} from '../src/studentAccountManagement.js'

const student = { id: 37, name: 'Fictional Student', account_active: true }

test('account status display distinguishes active, inactive and missing accounts', () => {
  assert.equal(accountStatusLabel(true), 'Active')
  assert.equal(accountStatusLabel(false), 'Inactive')
  assert.equal(accountStatusLabel(null), 'No account')
})

test('deactivation requires confirmation and makes no request when declined', async () => {
  const requests = []
  const result = await requestStudentAccountStatusChange(student, false, {
    confirmAction: (message) => {
      assert.match(message, /Deactivate Fictional Student/)
      assert.match(message, /records and history will be preserved/)
      return false
    },
    request: (...args) => requests.push(args),
  })

  assert.equal(result, null)
  assert.deepEqual(requests, [])
})

test('confirmed deactivation calls the faculty account-status endpoint', async () => {
  const requests = []
  const expected = { status: 'success', account_active: false }
  const result = await requestStudentAccountStatusChange(student, false, {
    confirmAction: () => true,
    request: async (...args) => {
      requests.push(args)
      return expected
    },
  })

  assert.equal(result, expected)
  assert.deepEqual(requests, [[
    '/students/37/account-status',
    { method: 'PUT', body: { is_active: false } },
  ]])
})

test('reactivation calls the same endpoint without a destructive confirmation', async () => {
  const requests = []
  await requestStudentAccountStatusChange(
    { ...student, account_active: false },
    true,
    {
      confirmAction: () => assert.fail('Reactivation must not require deactivation confirmation'),
      request: async (...args) => requests.push(args),
    },
  )

  assert.deepEqual(requests, [[
    '/students/37/account-status',
    { method: 'PUT', body: { is_active: true } },
  ]])
})
