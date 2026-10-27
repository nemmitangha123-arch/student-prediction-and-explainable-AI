import test from 'node:test'
import assert from 'node:assert/strict'
import {
  studentRegistrationCsvTemplate,
  studentRegistrationErrorsCsv,
  validateStudentRegistrationCsv,
} from '../src/studentRegistrationCsv.js'

test('registration template has the required identity and login columns only', () => {
  assert.equal(
    studentRegistrationCsvTemplate(),
    '"Student Name","Student Roll Number","Username","Password"\r\n',
  )
})

test('client validation accepts quoted fields and hides passwords from previews', () => {
  const result = validateStudentRegistrationCsv(
    '"Student Name","Student Roll Number","Username","Password"\r\n' +
    '"Student, Example","ROLL-1","student.one","A-private-password"\r\n',
  )
  assert.equal(result.fatalError, '')
  assert.equal(result.totalRows, 1)
  assert.equal(result.rows[0].status, 'valid')
  assert.equal(result.rows[0].student_name, 'Student, Example')
  assert.equal(JSON.stringify(result.rows), JSON.stringify([{
    row_number: 2,
    student_name: 'Student, Example',
    student_id: 'ROLL-1',
    username: 'student.one',
    status: 'valid',
    errors: [],
  }]))
  assert.equal(JSON.stringify(result).includes('A-private-password'), false)
})

test('client validation finds missing fields and duplicate roll numbers and usernames', () => {
  const result = validateStudentRegistrationCsv(
    'Student Name,Student Roll Number,Username,Password\n' +
    'First Student,ROLL-1,student.one,Password-One\n' +
    'Second Student,roll-1,STUDENT.ONE,Password-Two\n' +
    ',ROLL-3,student.three,Password-Three\n',
  )
  assert.equal(result.totalRows, 3)
  assert.equal(result.rows.filter((row) => row.status === 'invalid').length, 3)
  assert.ok(result.errors.some((error) => error.field === 'student_id' && error.reason.includes('duplicated')))
  assert.ok(result.errors.some((error) => error.field === 'username' && error.reason.includes('duplicated')))
  assert.ok(result.errors.some((error) => error.field === 'student_name' && error.reason.includes('required')))
  assert.equal(JSON.stringify(result).includes('Password-One'), false)
})

test('client validation requires a password without exposing its value', () => {
  const result = validateStudentRegistrationCsv(
    'Student Name,Student Roll Number,Username,Password\nA Student,ROLL-1,a.student,\n',
  )
  assert.equal(result.rows[0].status, 'invalid')
  assert.ok(result.errors.some((error) => error.field === 'password' && error.reason === 'Password is required.'))
})

test('client validation rejects malformed CSV and missing headers', () => {
  const malformed = validateStudentRegistrationCsv(
    'Student Name,Student Roll Number,Username,Password\n"unfinished',
  )
  assert.match(malformed.fatalError, /Malformed CSV/)

  const missingHeader = validateStudentRegistrationCsv(
    'Student Name,Student Roll Number,Username\nA Name,ROLL-1,user.one\n',
  )
  assert.match(missingHeader.fatalError, /required column/)
})

test('registration error report does not include password values', () => {
  const report = studentRegistrationErrorsCsv([{
    row_number: 2,
    student_id: 'ROLL-1',
    student_name: 'A Student',
    username: 'a.student',
    field: 'student_id',
    reason: 'Student roll number already exists.',
  }])
  assert.match(report, /Student Roll Number/)
  assert.equal(report.includes('password'), false)
})
