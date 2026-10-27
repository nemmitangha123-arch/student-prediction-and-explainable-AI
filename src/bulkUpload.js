const TEMPLATE_HEADERS = [
  'Student ID',
  'Student Name',
  'Attendance',
  'Internal Test 1',
  'Internal Test 2',
  'Assignment Score',
  'Daily Study Hours',
]

function csvCell(value) {
  const text = String(value ?? '')
  const safeText = /^[\t\r\n ]*[=+\-@]/.test(text) ? `'${text}` : text
  return `"${safeText.replaceAll('"', '""')}"`
}

function makeCsv(headers, rows) {
  return [headers, ...rows].map((row) => row.map(csvCell).join(',')).join('\r\n')
}

export function sampleStudentCsvTemplate() {
  return `${makeCsv(TEMPLATE_HEADERS, [])}\r\n`
}

export function successfulResultsCsv(results) {
  return makeCsv(
    ['Student ID', 'Student Name', 'Predicted Marks', 'Risk Category', 'Recommendations'],
    results.map((result) => [
      result.student_id,
      result.student_name,
      result.predicted_marks,
      result.risk_level,
      (result.recommendations ?? [])
        .map((item) => `${item.feature} (${item.priority}): ${item.recommendation}`)
        .join(' | '),
    ]),
  )
}

export function validationErrorsCsv(errors) {
  return makeCsv(
    ['CSV Row', 'Student ID', 'Student Name', 'Field', 'Reason'],
    errors.map((error) => [
      error.row_number,
      error.student_id,
      error.student_name,
      error.field,
      error.reason,
    ]),
  )
}

export function downloadCsv(filename, content) {
  const blob = new Blob(['\uFEFF', content], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
