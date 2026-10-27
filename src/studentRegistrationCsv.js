const HEADERS = ['Student Name', 'Student Roll Number', 'Username', 'Password']
const REQUIRED_FIELDS = [
  { key: 'student_name', aliases: ['studentname', 'fullname', 'name'], label: 'Student Name' },
  { key: 'student_id', aliases: ['studentid', 'studentrollno', 'studentrollnumber', 'rollno', 'rollnumber', 'id'], label: 'Student Roll Number' },
  { key: 'username', aliases: ['username', 'loginusername'], label: 'Username' },
  { key: 'password', aliases: ['password', 'loginpassword'], label: 'Password' },
]

function normalizeHeader(value) {
  return value.trim().toLocaleLowerCase().replace(/[^a-z0-9]/g, '')
}

function parseCsv(text) {
  const rows = []
  let row = []
  let cell = ''
  let quoted = false
  let line = 1
  let rowLine = 1

  for (let index = text.charCodeAt(0) === 0xfeff ? 1 : 0; index < text.length; index += 1) {
    const char = text[index]
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          cell += '"'
          index += 1
        } else {
          quoted = false
        }
      } else {
        cell += char
        if (char === '\n') line += 1
        else if (char === '\r' && text[index + 1] !== '\n') line += 1
      }
      continue
    }
    if (char === '"' && cell.length === 0) {
      quoted = true
    } else if (char === ',') {
      row.push(cell)
      cell = ''
    } else if (char === '\n' || char === '\r') {
      row.push(cell)
      if (row.some((value) => value.trim())) rows.push({ values: row, line: rowLine })
      row = []
      cell = ''
      if (char === '\r' && text[index + 1] === '\n') index += 1
      line += 1
      rowLine = line
    } else {
      cell += char
    }
  }

  if (quoted) throw new Error(`Malformed CSV record near row ${rowLine}.`)
  if (cell.length || row.length) {
    row.push(cell)
    if (row.some((value) => value.trim())) rows.push({ values: row, line: rowLine })
  }
  return rows
}

export function validateStudentRegistrationCsv(text) {
  let parsed
  try {
    parsed = parseCsv(text)
  } catch (error) {
    return { fatalError: error.message, rows: [], errors: [], totalRows: 0 }
  }
  if (!parsed.length) {
    return { fatalError: 'The CSV file is empty or has no header row.', rows: [], errors: [], totalRows: 0 }
  }

  const [headerRow, ...dataRows] = parsed
  const positions = {}
  const headerErrors = []
  headerRow.values.forEach((header, index) => {
    const normalized = normalizeHeader(header)
    const field = REQUIRED_FIELDS.find((item) => item.aliases.includes(normalized))
    if (!field) return
    if (positions[field.key] !== undefined) {
      headerErrors.push({ row_number: 1, field: field.key, reason: 'The header appears more than once.' })
    } else {
      positions[field.key] = index
    }
  })
  for (const field of REQUIRED_FIELDS) {
    if (positions[field.key] === undefined) {
      headerErrors.push({ row_number: 1, field: field.key, reason: 'A required column is missing.' })
    }
  }
  if (headerErrors.length) {
    return { fatalError: 'The CSV must contain each required column exactly once.', rows: [], errors: headerErrors, totalRows: 0 }
  }
  if (!dataRows.length) {
    return { fatalError: 'The CSV contains no student records.', rows: [], errors: [], totalRows: 0 }
  }
  if (dataRows.length > 500) {
    return { fatalError: 'The CSV may contain at most 500 records.', rows: [], errors: [], totalRows: dataRows.length }
  }

  const rows = dataRows.map(({ values, line }) => {
    const record = {
      row_number: line,
      student_name: values[positions.student_name]?.trim() ?? '',
      student_id: values[positions.student_id]?.trim() ?? '',
      username: values[positions.username]?.trim() ?? '',
      status: 'valid',
      errors: [],
    }
    if (values.length !== headerRow.values.length) {
      record.errors.push({ field: 'row', reason: `Expected ${headerRow.values.length} columns, found ${values.length}.` })
    }
    for (const field of REQUIRED_FIELDS) {
      if (field.key === 'password') continue
      if (!record[field.key]) {
        record.errors.push({ field: field.key, reason: `${field.label} is required.` })
      }
    }
    if (record.student_name.length > 120) {
      record.errors.push({ field: 'student_name', reason: 'Student name must be 120 characters or fewer.' })
    }
    if (record.student_id.length > 80) {
      record.errors.push({ field: 'student_id', reason: 'Student roll number must be 80 characters or fewer.' })
    }
    if (record.username && !/^[A-Za-z0-9._-]{3,80}$/.test(record.username)) {
      record.errors.push({ field: 'username', reason: 'Username must be 3 to 80 characters using letters, numbers, dots, underscores or hyphens.' })
    }
    const password = values[positions.password] ?? ''
    if (!password.trim()) {
      record.errors.push({ field: 'password', reason: 'Password is required.' })
    } else if (password.length < 8) {
      record.errors.push({ field: 'password', reason: 'Password must be at least 8 characters.' })
    } else if (password.length > 256) {
      record.errors.push({ field: 'password', reason: 'Password must be 256 characters or fewer.' })
    }
    if (record.errors.length) record.status = 'invalid'
    return record
  })

  for (const [field, label] of [['student_id', 'Student roll number'], ['username', 'Username']]) {
    const firstByValue = new Map()
    for (const record of rows) {
      const value = record[field].toLocaleLowerCase()
      if (!value) continue
      const first = firstByValue.get(value)
      if (first === undefined) {
        firstByValue.set(value, record)
      } else {
        const reason = `${label} is duplicated in row ${first.row_number}.`
        record.errors.push({ field, reason })
        if (!first.errors.some((item) => item.field === field)) {
          first.errors.push({ field, reason: `${label} is duplicated in row ${record.row_number}.` })
        }
        record.status = 'invalid'
        first.status = 'invalid'
      }
    }
  }
  return {
    fatalError: '',
    rows,
    errors: rows.flatMap((record) => record.errors.map((error) => ({
      row_number: record.row_number,
      student_id: record.student_id,
      student_name: record.student_name,
      username: record.username,
      ...error,
    }))),
    totalRows: rows.length,
  }
}

function csvCell(value) {
  const text = String(value ?? '')
  const safeText = /^[\t\r\n ]*[=+\-@]/.test(text) ? `'${text}` : text
  return `"${safeText.replaceAll('"', '""')}"`
}

function makeCsv(headers, rows) {
  return [headers, ...rows].map((row) => row.map(csvCell).join(',')).join('\r\n')
}

export function studentRegistrationCsvTemplate() {
  return `${makeCsv(HEADERS, [])}\r\n`
}

export function studentRegistrationErrorsCsv(errors) {
  return makeCsv(
    ['CSV Row', 'Student Roll Number', 'Student Name', 'Username', 'Field', 'Reason'],
    errors.map((error) => [
      error.row_number,
      error.student_id,
      error.student_name,
      error.username,
      error.field,
      error.reason,
    ]),
  )
}

export function downloadRegistrationCsv(filename, content) {
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
