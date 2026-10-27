import assert from 'node:assert/strict'
import test from 'node:test'

import {
  sampleStudentCsvTemplate,
  successfulResultsCsv,
  validationErrorsCsv,
} from '../src/bulkUpload.js'

test('template contains the required import headers', () => {
  assert.equal(
    sampleStudentCsvTemplate(),
    '"Student ID","Student Name","Attendance","Internal Test 1","Internal Test 2","Assignment Score","Daily Study Hours"\r\n',
  )
})

test('successful results export includes risk and recommendations safely', () => {
  const csv = successfulResultsCsv([{
    student_id: 'ROLL-001',
    student_name: '=HYPERLINK("https://example.invalid")',
    predicted_marks: 42.5,
    risk_level: 'Borderline',
    recommendations: [{
      feature: 'Attendance',
      priority: 'HIGH',
      recommendation: 'Improve, steadily',
    }],
  }])

  assert.match(csv, /ROLL-001/)
  assert.match(csv, /Borderline/)
  assert.match(csv, /Attendance \(HIGH\): Improve, steadily/)
  assert.match(csv, /'=HYPERLINK\(""https:\/\/example\.invalid""\)/)
})

test('validation error export preserves row, field and escaped reason', () => {
  const csv = validationErrorsCsv([{
    row_number: 4,
    student_id: 'ROLL-004',
    student_name: 'Student Four',
    field: 'attendance',
    reason: 'Attendance must be between 52 and 100.',
  }])

  assert.match(csv, /"4","ROLL-004","Student Four","attendance","Attendance must be between 52 and 100\."/)
})
