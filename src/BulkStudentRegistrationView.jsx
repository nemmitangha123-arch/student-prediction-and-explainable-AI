import { useRef, useState } from 'react'
import {
  ArrowLeft,
  CheckCircle2,
  CircleAlert,
  Download,
  FileSpreadsheet,
  LoaderCircle,
  RotateCcw,
  Upload,
} from 'lucide-react'
import { apiRequest } from './api'
import {
  downloadRegistrationCsv,
  studentRegistrationCsvTemplate,
  studentRegistrationErrorsCsv,
  validateStudentRegistrationCsv,
} from './studentRegistrationCsv'

const MAX_FILE_SIZE = 2 * 1024 * 1024
const MAX_ROWS = 500
const CARD = 'ui-card p-5 sm:p-6'
const PRIMARY_BUTTON = 'ui-btn ui-btn-primary'
const SECONDARY_BUTTON = 'ui-btn ui-btn-secondary'

const STEPS = ['Download Template', 'Upload CSV', 'Validate & Preview', 'Confirm & Results']

function ErrorDetails({ errors, label = 'Registration validation errors' }) {
  if (!errors?.length) return null
  return (
    <div className="mt-4 max-h-72 overflow-auto rounded-xl border border-rose-200 bg-white" role="region" aria-label={label} tabIndex={0}>
      <table className="ui-table w-full min-w-[720px] text-left text-sm">
        <caption className="sr-only">{label}</caption>
        <thead className="sticky top-0 z-10">
          <tr>
            <th scope="col" className="px-3 py-2.5">CSV row</th>
            <th scope="col" className="px-3 py-2.5">Roll number</th>
            <th scope="col" className="px-3 py-2.5">Student name</th>
            <th scope="col" className="px-3 py-2.5">Username</th>
            <th scope="col" className="px-3 py-2.5">Field</th>
            <th scope="col" className="px-3 py-2.5">Reason</th>
          </tr>
        </thead>
        <tbody>
          {errors.map((error, index) => (
            <tr key={`${error.row_number}-${error.field}-${index}`}>
              <td className="px-3 py-2.5">{error.row_number > 0 ? error.row_number : '—'}</td>
              <td className="px-3 py-2.5">{error.student_id || '—'}</td>
              <td className="px-3 py-2.5">{error.student_name || '—'}</td>
              <td className="px-3 py-2.5">{error.username || '—'}</td>
              <td className="px-3 py-2.5">{error.field || '—'}</td>
              <td className="px-3 py-2.5 text-rose-800">{error.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function workflowStep(preview, result, file) {
  if (result) return 3
  if (preview) return 2
  if (file) return 1
  return 0
}

function BulkStudentRegistrationView({ onBack, onRegistered }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [fileError, setFileError] = useState('')
  const [requestError, setRequestError] = useState('')
  const [isValidating, setIsValidating] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [confirmed, setConfirmed] = useState(false)

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setFileError('')
    setRequestError('')
    setConfirmed(false)
    if (inputRef.current) inputRef.current.value = ''
  }

  const chooseFile = (selectedFile) => {
    if (isValidating || isImporting) return
    setPreview(null)
    setResult(null)
    setRequestError('')
    setFileError('')
    setConfirmed(false)
    if (!selectedFile) return
    if (!selectedFile.name.toLocaleLowerCase().endsWith('.csv')) {
      setFile(null)
      setFileError('Only .csv files are supported.')
      return
    }
    if (selectedFile.size > MAX_FILE_SIZE) {
      setFile(null)
      setFileError('The selected file exceeds the 2 MB size limit.')
      return
    }
    setFile(selectedFile)
  }

  const validateFile = async () => {
    if (!file || isValidating) return
    setIsValidating(true)
    setRequestError('')
    setPreview(null)
    setResult(null)
    setConfirmed(false)
    try {
      const localCheck = validateStudentRegistrationCsv(await file.text())
      if (localCheck.fatalError) {
        setPreview({
          status: 'error',
          message: localCheck.fatalError,
          total_rows: localCheck.totalRows,
          valid_rows: 0,
          rejected_rows: localCheck.totalRows,
          errors: localCheck.errors,
          rows: localCheck.rows,
        })
        return
      }

      const formData = new FormData()
      formData.append('file', file)
      setPreview(await apiRequest('/students/registration/validate', {
        method: 'POST',
        body: formData,
      }))
    } catch (error) {
      setRequestError(error.message)
      setPreview(error.response?.rows ? error.response : null)
    } finally {
      setIsValidating(false)
    }
  }

  const importFile = async () => {
    if (!file || !preview?.valid_rows || !confirmed || isImporting) return
    setIsImporting(true)
    setRequestError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await apiRequest('/students/registration/import', {
        method: 'POST',
        body: formData,
      })
      setResult(response)
      setPreview(null)
      setConfirmed(false)
      await onRegistered?.()
    } catch (error) {
      setRequestError(error.message)
      if (error.response?.rows) {
        setResult(error.response)
        setPreview(null)
        if (error.response?.successful_registrations) await onRegistered?.()
      }
    } finally {
      setIsImporting(false)
    }
  }

  const activeStep = workflowStep(preview, result, file)
  const errors = result?.errors ?? preview?.errors ?? []

  return (
    <div className="space-y-6" data-testid="bulk-student-registration">
      <nav aria-label="Student registration workflow" className={CARD}>
        <ol className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {STEPS.map((step, index) => (
            <li key={step} aria-current={activeStep === index ? 'step' : undefined} className={`rounded-xl border px-3 py-3 text-sm ${activeStep === index ? 'border-blue-200 bg-blue-50 text-blue-900' : activeStep > index ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : 'border-slate-200 bg-slate-50 text-slate-600'}`}>
              <span className="font-semibold">{index + 1}. {step}</span>
              {index === 2 && preview && <span className="mt-1 block text-xs">{preview.valid_rows ?? 0} valid · {preview.rejected_rows ?? 0} invalid</span>}
            </li>
          ))}
        </ol>
      </nav>

      {!result && (
        <section className={CARD}>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Bulk Student Registration</h2>
              <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
                Create student identities and login accounts only. Passwords in the uploaded file are used to create hashed credentials,
                are never returned in the preview or results, and are not retained by this workflow.
              </p>
            </div>
            <button
              type="button"
              className={`${SECONDARY_BUTTON} w-full sm:w-auto`}
              onClick={() => downloadRegistrationCsv('student-registration-template.csv', studentRegistrationCsvTemplate())}
            >
              <Download size={16} aria-hidden="true" /> Download CSV Template
            </button>
          </div>

          <div className="mt-5 rounded-[18px] border border-blue-100 bg-blue-50/60 p-4 text-sm leading-6 text-slate-700">
            <p className="font-semibold text-slate-900">Required columns</p>
            <p className="font-medium">Student Name, Student Roll Number, Username, Password</p>
            <p className="mt-1">UTF-8 CSV only, up to 2 MB and {MAX_ROWS} student rows. Usernames are 3–80 letters, numbers, dots, underscores, or hyphens. Passwords must be at least 8 characters.</p>
            <p className="mt-1">Use a unique roll number and username for every account. Save any supplied passwords securely before uploading; passwords are not shown again after submission.</p>
          </div>

          <div className="mt-5 rounded-[22px] border-2 border-dashed border-slate-300 bg-slate-50 p-6 text-center">
            <FileSpreadsheet size={32} className="mx-auto text-blue-700" aria-hidden="true" />
            <p className="mt-3 font-semibold text-slate-900">Choose a student registration CSV file</p>
            <p className="mt-1 text-sm text-slate-600">CSV format · Maximum 2 MB · Up to {MAX_ROWS} rows</p>
            <input
              ref={inputRef}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              disabled={isValidating || isImporting}
              onChange={(event) => {
                chooseFile(event.target.files?.[0])
                event.target.value = ''
              }}
              aria-label="Choose student registration CSV file"
            />
            <button type="button" disabled={isValidating || isImporting} className={`${PRIMARY_BUTTON} mt-4`} onClick={() => inputRef.current?.click()}>
              <Upload size={16} aria-hidden="true" /> Browse CSV Files
            </button>
          </div>
          {file && (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-[18px] border border-blue-200 bg-white px-4 py-3">
              <div>
                <p className="text-sm font-semibold text-slate-900">{file.name}</p>
                <p className="text-xs text-slate-600">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button type="button" onClick={validateFile} disabled={isValidating || isImporting} aria-busy={isValidating} className={PRIMARY_BUTTON}>
                {isValidating
                  ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Validating…</>
                  : <><CheckCircle2 size={16} aria-hidden="true" /> Validate and Preview</>}
              </button>
            </div>
          )}
          {(fileError || requestError) && (
            <p role="alert" className="mt-4 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
              <CircleAlert size={17} className="mt-0.5 shrink-0" aria-hidden="true" />{fileError || requestError}
            </p>
          )}
        </section>
      )}

      {preview && (
        <>
          <section className={CARD}>
            <h2 className="text-lg font-semibold text-slate-900">Validation preview</h2>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              {[
                ['Total records', preview.total_rows ?? 0],
                ['Valid', preview.valid_rows ?? 0],
                ['Invalid', preview.rejected_rows ?? 0],
              ].map(([label, value]) => (
                <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <p className="text-sm text-slate-600">{label}</p>
                  <p className="text-2xl font-semibold text-slate-900">{value}</p>
                </div>
              ))}
            </div>
            {preview.message && <p role="alert" className="mt-4 text-sm text-rose-800">{preview.message}</p>}
            {preview.rows?.length > 0 && (
              <div className="mt-4 max-h-96 overflow-auto rounded-xl border border-slate-200">
                <table className="ui-table w-full min-w-[640px] text-left text-sm">
                  <thead className="sticky top-0 z-10">
                    <tr>
                      <th scope="col" className="px-3 py-2.5">CSV row</th>
                      <th scope="col" className="px-3 py-2.5">Roll number</th>
                      <th scope="col" className="px-3 py-2.5">Student name</th>
                      <th scope="col" className="px-3 py-2.5">Username</th>
                      <th scope="col" className="px-3 py-2.5">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row) => (
                      <tr key={row.row_number}>
                        <td className="px-3 py-2.5">{row.row_number}</td>
                        <td className="px-3 py-2.5">{row.student_id || '—'}</td>
                        <td className="px-3 py-2.5">{row.student_name || '—'}</td>
                        <td className="px-3 py-2.5">{row.username || '—'}</td>
                        <td className={`px-3 py-2.5 font-semibold ${row.status === 'valid' ? 'text-emerald-700' : 'text-rose-700'}`}>{row.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <ErrorDetails errors={errors} />
            {errors.length > 0 && (
              <button type="button" className={`${SECONDARY_BUTTON} mt-4`} onClick={() => downloadRegistrationCsv('student-registration-errors.csv', studentRegistrationErrorsCsv(errors))}>
                <Download size={16} aria-hidden="true" /> Download Error Report
              </button>
            )}
          </section>
          <section className={CARD}>
            <h2 className="text-lg font-semibold text-slate-900">Confirm account creation</h2>
            <p className="mt-1 text-sm leading-6 text-slate-600">
              {preview.valid_rows
                ? `Confirm creation of ${preview.valid_rows} valid account${preview.valid_rows === 1 ? '' : 's'}. ${preview.rejected_rows ?? 0} invalid row(s) will be rejected and remain unchanged.`
                : 'There are no valid accounts to create. Correct the CSV and validate it again.'}
            </p>
            {preview.valid_rows > 0 && (
              <label className="mt-4 flex items-start gap-2 text-sm text-slate-700">
                <input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} className="mt-1" />
                I reviewed the preview and confirm creation of the valid student accounts.
              </label>
            )}
            {requestError && <p role="alert" className="mt-3 text-sm text-rose-800">{requestError}</p>}
            <div className="mt-4 flex flex-wrap gap-3">
              <button type="button" className={PRIMARY_BUTTON} disabled={!preview.valid_rows || !confirmed || isImporting} onClick={importFile}>
                {isImporting
                  ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Creating accounts…</>
                  : <><CheckCircle2 size={16} aria-hidden="true" /> Confirm Registration</>}
              </button>
              <button type="button" className={SECONDARY_BUTTON} disabled={isImporting} onClick={reset}>
                <RotateCcw size={16} aria-hidden="true" /> Choose another file
              </button>
            </div>
          </section>
        </>
      )}

      {result && (
        <section className={CARD} data-testid="registration-results">
          <h2 className="text-lg font-semibold text-slate-900">Registration results</h2>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            {[
              ['Total records', result.total_rows ?? 0],
              ['Accounts created', result.successful_registrations ?? 0],
              ['Rejected rows', result.rejected_rows ?? 0],
            ].map(([label, value]) => (
              <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <p className="text-sm text-slate-600">{label}</p>
                <p className="text-2xl font-semibold text-slate-900">{value}</p>
              </div>
            ))}
          </div>
          {result.message && <p role="alert" className="mt-4 text-sm text-rose-800">{result.message}</p>}
          {result.results?.length > 0 && (
            <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
              Accounts were created. Passwords are not displayed or included in the results; use the credentials provided in the original CSV.
              <ul className="mt-2 list-inside list-disc">
                {result.results.map((student) => <li key={student.id}>{student.name} — {student.roll_no} — {student.username}</li>)}
              </ul>
            </div>
          )}
          <ErrorDetails errors={errors} label="Rejected registration rows" />
          {errors.length > 0 && (
            <button type="button" className={`${SECONDARY_BUTTON} mt-4`} onClick={() => downloadRegistrationCsv('student-registration-errors.csv', studentRegistrationErrorsCsv(errors))}>
              <Download size={16} aria-hidden="true" /> Download Error Report
            </button>
          )}
          <div className="mt-5 flex flex-wrap gap-3">
            <button type="button" className={PRIMARY_BUTTON} onClick={reset}><RotateCcw size={16} aria-hidden="true" /> Register Another CSV</button>
            <button type="button" className={SECONDARY_BUTTON} onClick={onBack}><ArrowLeft size={16} aria-hidden="true" /> Return to Faculty Dashboard</button>
          </div>
        </section>
      )}
    </div>
  )
}

export default BulkStudentRegistrationView
