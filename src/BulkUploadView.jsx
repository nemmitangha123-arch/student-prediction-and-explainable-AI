import { useRef, useState } from 'react'
import {
  CheckCircle2,
  CircleAlert,
  Download,
  FileSpreadsheet,
  LoaderCircle,
  RotateCcw,
  Search,
  Upload,
} from 'lucide-react'
import { apiRequest } from './api'
import {
  downloadCsv,
  sampleStudentCsvTemplate,
  successfulResultsCsv,
  validationErrorsCsv,
} from './bulkUpload'

const MAX_FILE_SIZE = 2 * 1024 * 1024
const MAX_ROWS = 500
const PRIMARY_BUTTON = 'ui-btn ui-btn-primary'
const SECONDARY_BUTTON = 'ui-btn ui-btn-secondary'
const CARD = 'ui-card p-5 sm:p-6'

function ErrorDetailsTable({ errors, label }) {
  if (!errors?.length) return null
  return (
    <div
      className="mt-3 max-h-72 overflow-auto rounded-xl border border-rose-200 bg-white"
      tabIndex={0}
      role="region"
      aria-label={`${label}; scroll to review all issues`}
    >
      <table className="ui-table w-full min-w-[760px] text-left text-sm">
        <caption className="sr-only">{label}</caption>
        <thead className="sticky top-0 z-10">
          <tr>
            <th scope="col" className="px-3 py-2.5">CSV row</th>
            <th scope="col" className="px-3 py-2.5">Student ID</th>
            <th scope="col" className="px-3 py-2.5">Student name</th>
            <th scope="col" className="px-3 py-2.5">Field</th>
            <th scope="col" className="px-3 py-2.5">Reason</th>
          </tr>
        </thead>
        <tbody>
          {errors.map((error, index) => (
            <tr key={`${error.row_number}-${error.field}-${index}`}>
              <td className="whitespace-nowrap px-3 py-2.5 font-medium text-slate-900">
                {error.row_number > 0 ? error.row_number : '—'}
              </td>
              <td className="px-3 py-2.5">{error.student_id || '—'}</td>
              <td className="px-3 py-2.5">{error.student_name || '—'}</td>
              <td className="px-3 py-2.5">{error.field || '—'}</td>
              <td className="whitespace-normal px-3 py-2.5 text-rose-800">{error.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const WORKFLOW_STEPS = ['Upload', 'Validation & Preview', 'Confirm Import', 'Results']

function WorkflowSteps({ currentStep }) {
  return (
    <nav aria-label="Bulk upload workflow" className="ui-card p-4 sm:p-5">
      <ol className="grid grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-2">
        {WORKFLOW_STEPS.map((label, index) => {
          const isCurrent = currentStep === index
          const isComplete = currentStep > index
          return (
            <li
              key={label}
              aria-current={isCurrent ? 'step' : undefined}
              className={`flex min-w-0 items-start gap-2 rounded-xl border px-3 py-3 ${
                isCurrent
                  ? 'border-blue-200 bg-blue-50 text-blue-900'
                  : isComplete
                    ? 'border-emerald-200 bg-emerald-50/70 text-emerald-900'
                    : 'border-slate-200 bg-slate-50 text-slate-600'
              }`}
            >
              <span
                aria-hidden="true"
                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                  isCurrent
                    ? 'bg-blue-700 text-white'
                    : isComplete
                      ? 'bg-emerald-700 text-white'
                      : 'bg-white text-slate-500 ring-1 ring-slate-300'
                }`}
              >
                {isComplete ? <CheckCircle2 size={14} /> : index + 1}
              </span>
              <span className="min-w-0">
                <span className="block text-xs font-semibold leading-5 sm:text-sm">{label}</span>
                <span className="block text-[11px] leading-4 text-slate-600">
                  {isCurrent ? 'Current step' : isComplete ? 'Complete' : 'Next'}
                </span>
              </span>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

function riskBadgeClass(riskLevel) {
  if (riskLevel === 'High Concern') return 'border-rose-200 bg-rose-50 text-rose-800'
  if (riskLevel === 'Borderline') return 'border-amber-200 bg-amber-50 text-amber-900'
  if (riskLevel === 'Lower Concern') return 'border-emerald-200 bg-emerald-50 text-emerald-800'
  return 'border-slate-200 bg-slate-50 text-slate-700'
}

function BulkUploadView({ onImported }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [fileError, setFileError] = useState('')
  const [requestError, setRequestError] = useState('')
  const [requestErrors, setRequestErrors] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [search, setSearch] = useState('')

  const results = result?.results ?? []
  const query = search.trim().toLocaleLowerCase()
  const filteredResults = results.filter((row) => (
      row.student_id.toLocaleLowerCase().includes(query)
      || row.student_name.toLocaleLowerCase().includes(query)
      || row.risk_level.toLocaleLowerCase().includes(query)
  ))
  const currentStep = result ? 3 : preview ? 2 : isValidating ? 1 : 0

  const chooseFile = (selectedFile) => {
    if (isValidating || isImporting) return
    setFileError('')
    setRequestError('')
    setRequestErrors([])
    setPreview(null)
    setResult(null)
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
    setRequestErrors([])
    setPreview(null)
    setResult(null)
    setConfirmed(false)
    try {
      const formData = new FormData()
      formData.append('file', file)
      setPreview(await apiRequest('/students/bulk-upload/validate', { method: 'POST', body: formData }))
    } catch (error) {
      setRequestError(error.message)
      setRequestErrors(error.response?.errors ?? [])
    } finally {
      setIsValidating(false)
    }
  }

  const importFile = async () => {
    if (!file || !preview?.valid_rows || !confirmed || isImporting) return
    setIsImporting(true)
    setRequestError('')
    setRequestErrors([])
    try {
      const formData = new FormData()
      formData.append('file', file)
      const imported = await apiRequest('/students/bulk-upload/import', {
        method: 'POST',
        body: formData,
      })
      setResult(imported)
      setPreview(null)
      setConfirmed(false)
      await onImported?.()
    } catch (error) {
      setRequestError(error.message)
      setRequestErrors(error.response?.errors ?? [])
      if (error.response?.results) {
        setResult(error.response)
        setPreview(null)
      }
    } finally {
      setIsImporting(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setFileError('')
    setRequestError('')
    setRequestErrors([])
    setConfirmed(false)
    setSearch('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="space-y-6" data-testid="bulk-upload">
      <WorkflowSteps currentStep={currentStep} />
      <section className={CARD}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Upload student records</h2>
            <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
              Import records only for existing student accounts with blank academic records and no prediction or intervention history.
              Existing identities and records are never replaced.
            </p>
          </div>
          <button
            type="button"
            className={`${SECONDARY_BUTTON} w-full sm:w-auto`}
            onClick={() => downloadCsv('student-upload-template.csv', sampleStudentCsvTemplate())}
          >
            <Download size={16} aria-hidden="true" /> Download Sample CSV Template
          </button>
        </div>

        <div className="mt-5 rounded-[18px] border border-blue-100 bg-blue-50/60 p-4 text-sm leading-6 text-slate-700">
          <p className="font-semibold text-slate-900">CSV requirements</p>
          <p>UTF-8 CSV only, up to 2 MB and {MAX_ROWS} student rows. Required columns:</p>
          <p className="mt-1 break-words font-medium">Student ID, Student Name, Attendance, Internal Test 1, Internal Test 2, Assignment Score, Daily Study Hours</p>
          <p className="mt-1">Student ID must match an existing roll number and the name must match that student's account. Inputs must be within the model ranges: attendance 52–100, tests 18–40 and 16–40, assignment 4–10, study hours 1–5.</p>
        </div>

        {!result && (
          <>
            <div className="mt-5 flex items-start gap-3 rounded-xl border border-blue-100 bg-blue-50/50 p-3 text-sm text-slate-700" role="status" aria-live="polite">
              <span className="ui-badge shrink-0 border-blue-200 bg-white text-blue-800">
                {isValidating ? 'Validating' : preview ? 'Preview ready' : file ? 'File selected' : 'Next'}
              </span>
              <p className="leading-5">
                {isValidating
                  ? 'Checking the file and every row with the server. Keep this page open.'
                  : preview
                    ? 'Review the validation details and preview below. Import is available only after you confirm.'
                    : file
                      ? 'Validate this file to check its rows and see the preview before import.'
                      : 'Choose a CSV file to begin. No student records are saved during file selection or validation.'}
              </p>
            </div>
            <div
              onDragEnter={(event) => { event.preventDefault(); setIsDragging(true) }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false) }}
              onDrop={handleDrop}
              aria-describedby="bulk-upload-file-help"
              className={`mt-4 rounded-[22px] border-2 border-dashed p-6 text-center transition-colors sm:p-9 ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-slate-50 hover:border-blue-300 hover:bg-blue-50/40'}`}
            >
              <FileSpreadsheet size={32} className="mx-auto text-blue-700" aria-hidden="true" />
              <p className="mt-3 font-semibold text-slate-900">Drag and drop a CSV file here</p>
              <p id="bulk-upload-file-help" className="mt-1 text-sm text-slate-600">CSV format · Maximum 2 MB · Up to {MAX_ROWS} student rows</p>
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
                aria-label="Choose student CSV file"
              />
              <button type="button" disabled={isValidating || isImporting} className={`${PRIMARY_BUTTON} mt-4`} onClick={() => inputRef.current?.click()}>
                <Upload size={16} aria-hidden="true" /> Browse CSV Files
              </button>
            </div>
            {file && (
              <div className="mt-4 flex flex-col items-stretch justify-between gap-3 rounded-[18px] border border-blue-200 bg-white px-4 py-3 sm:flex-row sm:items-center">
                <div className="min-w-0">
                  <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    <FileSpreadsheet size={14} aria-hidden="true" /> Selected file
                  </p>
                  <p className="mt-1 truncate text-sm font-semibold text-slate-900" title={file.name}>{file.name}</p>
                  <p className="text-xs text-slate-600">
                    {(file.size / 1024).toFixed(1)} KB
                    {preview ? ` · ${preview.total_rows} rows detected` : ' · Row count will be checked during validation'}
                  </p>
                </div>
                <button type="button" onClick={validateFile} disabled={isValidating || isImporting} aria-busy={isValidating} className={`${PRIMARY_BUTTON} w-full shrink-0 sm:w-auto`}>
                  {isValidating ? <><LoaderCircle size={16} className="animate-spin motion-reduce:animate-none" aria-hidden="true" /> Validating…</> : <><CheckCircle2 size={16} aria-hidden="true" /> Validate and Preview</>}
                </button>
              </div>
            )}
            {(fileError || requestError) && (
              <div role="alert" className="mt-4 rounded-[18px] border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
                {fileError || requestError}
                <ErrorDetailsTable errors={requestErrors} label="File validation error details" />
              </div>
            )}
          </>
        )}
      </section>

      {preview && (
        <>
          <section className={CARD}>
            <h2 className="text-lg font-semibold text-slate-900">Validation summary</h2>
            <p className="mt-1 text-sm text-slate-600" role="status" aria-live="polite">
              Validation complete. The server checked {preview.total_rows} row(s): {preview.valid_rows} valid and {preview.rejected_rows} rejected. Only valid rows can be imported.
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              {[
                ['Rows detected', preview.total_rows, 'border-slate-200 bg-slate-50 text-slate-900'],
                ['Valid to import', preview.valid_rows, 'border-emerald-200 bg-emerald-50 text-emerald-800'],
                ['Rejected', preview.rejected_rows, 'border-rose-200 bg-rose-50 text-rose-800'],
              ].map(([label, value, tone]) => (
                <div key={label} className={`flex items-center justify-between gap-3 rounded-[18px] border p-4 ${tone}`}>
                  <p className="text-sm font-medium">{label}</p>
                  <p className="text-2xl font-semibold tabular-nums" aria-label={`${value} ${label.toLocaleLowerCase()}`}>{value}</p>
                </div>
              ))}
            </div>
            {preview.errors.length > 0 && (
              <div className="mt-4 rounded-[18px] border border-rose-200 bg-rose-50 p-4">
                <h3 className="font-semibold text-rose-900">Rows needing correction</h3>
                <p className="mt-1 text-sm text-rose-800">
                  {preview.errors.length} validation issue(s) across rejected rows. Review the row, field, and reason below. All issues remain included in the downloadable report.
                </p>
                <ErrorDetailsTable errors={preview.errors} label="Validation issues by CSV row and field" />
                <button
                  type="button"
                  className={`${SECONDARY_BUTTON} mt-4`}
                  onClick={() => downloadCsv('student-upload-errors.csv', validationErrorsCsv(preview.errors))}
                >
                  <Download size={15} aria-hidden="true" /> Download Error Report
                </button>
              </div>
            )}
          </section>

          <section className={CARD}>
            <div className="flex flex-wrap items-end justify-between gap-2">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">Preview before import</h2>
                <p className="mt-1 text-sm text-slate-600">All {preview.rows.length} row(s) are available below. Scroll within the table to review the full file.</p>
              </div>
              <span className="ui-badge border-slate-200 bg-slate-50 text-slate-700">{preview.rows.length} rows</span>
            </div>
            <div className="mt-4 max-h-[28rem] overflow-auto rounded-[18px] border border-slate-200" tabIndex={0} role="region" aria-label="Scrollable student validation preview table">
              <table className="ui-table w-full min-w-[850px] text-left text-sm">
                <caption className="sr-only">All uploaded student rows with academic inputs and their validation status</caption>
                <thead className="sticky top-0 z-10 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th scope="col" className="px-4 py-3">CSV row</th><th scope="col" className="px-4 py-3">Student ID</th><th scope="col" className="px-4 py-3">Name</th>
                    <th scope="col" className="px-4 py-3 text-right">Attendance</th><th scope="col" className="px-4 py-3 text-right">Test 1</th><th scope="col" className="px-4 py-3 text-right">Test 2</th>
                    <th scope="col" className="px-4 py-3 text-right">Assignment</th><th scope="col" className="px-4 py-3 text-right">Study hours</th><th scope="col" className="px-4 py-3">Validation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {preview.rows.map((row) => (
                    <tr key={row.row_number} className={row.status === 'invalid' ? 'bg-rose-50/40' : 'bg-white'}>
                      <td className="whitespace-nowrap px-4 py-3 tabular-nums">{row.row_number}</td><td className="px-4 py-3 font-medium">{row.student_id || '—'}</td><td className="px-4 py-3">{row.student_name || '—'}</td>
                      {['attendance', 'test1', 'test2', 'assignment', 'study_hours'].map((field) => <td key={field} className="px-4 py-3 text-right tabular-nums">{row.inputs[field] ?? '—'}</td>)}
                      <td className="px-4 py-3">
                        <span className={`ui-badge ${row.status === 'valid' ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : 'border-rose-200 bg-rose-50 text-rose-800'}`}>
                          {row.status === 'valid' ? <CheckCircle2 size={13} aria-hidden="true" /> : <CircleAlert size={13} aria-hidden="true" />}
                          {row.status === 'valid' ? 'Valid' : 'Rejected'}
                        </span>
                        {row.errors.length > 0 && <ul className="mt-2 space-y-1 text-xs text-rose-800">{row.errors.map((error, index) => <li key={`${error.field}-${index}`}><strong>{error.field}:</strong> {error.reason}</li>)}</ul>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {preview.valid_rows > 0 && (
              <div className="mt-5 rounded-[18px] border border-amber-200 bg-amber-50 p-4 sm:p-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-semibold text-amber-950">Confirmation required</h3>
                  <span className="ui-badge border-amber-300 bg-white text-amber-950">{preview.valid_rows} record(s) to import</span>
                </div>
                <p className="mt-1 text-sm leading-6 text-amber-900">
                  {preview.valid_rows} valid row(s) will be saved with a prediction and SHAP-guided recommendations.
                  {preview.rejected_rows > 0 ? ` ${preview.rejected_rows} invalid row(s) will be excluded.` : ''}
                  {' '}The valid rows are committed together; if database saving fails, none are saved.
                </p>
                <label className="mt-3 flex items-start gap-2 text-sm text-amber-950">
                  <input id="bulk-import-confirmation" type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} disabled={isImporting} className="mt-1 h-4 w-4 shrink-0 accent-blue-600" />
                  I reviewed the preview and confirm importing the valid student records.
                </label>
                <button type="button" onClick={importFile} disabled={!confirmed || isImporting} aria-busy={isImporting} className={`${PRIMARY_BUTTON} mt-4 w-full sm:w-auto`}>
                  {isImporting ? <><LoaderCircle size={16} className="animate-spin motion-reduce:animate-none" aria-hidden="true" /> Importing…</> : <><CheckCircle2 size={16} aria-hidden="true" /> Confirm Import of {preview.valid_rows} Records</>}
                </button>
              </div>
            )}
            {preview.valid_rows === 0 && (
              <div className="mt-5 rounded-[18px] border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
                No rows are eligible to import. Correct the rejected rows in the CSV, then choose the corrected file to validate it again.
              </div>
            )}
            {requestError && <div role="alert" className="mt-4 rounded-[18px] border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{requestError}<ErrorDetailsTable errors={requestErrors} label="Import error details" /></div>}
          </section>
        </>
      )}

      {result && (
        <section className={CARD}>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="ui-eyebrow">Processing complete</p>
              <h2 className="mt-1 text-lg font-semibold text-slate-900">Import results</h2>
              <p className="mt-1 text-sm text-slate-600" role="status">
                Backend confirmed {result.successful_imports} successful import(s); {result.rejected_rows} row(s) were rejected.
              </p>
            </div>
            <button type="button" onClick={reset} className={`${SECONDARY_BUTTON} w-full sm:w-auto`}><RotateCcw size={15} aria-hidden="true" /> Upload Another File</button>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            {[
              ['Total rows', result.total_rows, 'border-slate-200 bg-slate-50 text-slate-900'],
              ['Successful imports', result.successful_imports, 'border-emerald-200 bg-emerald-50 text-emerald-900'],
              ['Rejected rows', result.rejected_rows, 'border-rose-200 bg-rose-50 text-rose-900'],
            ].map(([label, value, tone]) => <div key={label} className={`flex items-center justify-between gap-3 rounded-[18px] border p-4 ${tone}`}><p className="text-sm font-medium">{label}</p><p className="text-2xl font-semibold tabular-nums">{value}</p></div>)}
          </div>
          {requestError && <div role="alert" className="mt-4 rounded-[18px] border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">{requestError}<ErrorDetailsTable errors={requestErrors} label="Import error details" /></div>}
          {result.errors.length > 0 && (
            <div className="mt-4 rounded-[18px] border border-rose-200 bg-rose-50 p-4">
              <h3 className="font-semibold text-rose-900">Rejected row details</h3>
              <p className="mt-1 text-sm text-rose-800">{result.errors.length} issue(s) are listed below. Scroll the table to review every rejected row.</p>
              <ErrorDetailsTable errors={result.errors} label="Rejected import rows and error details" />
              <button type="button" onClick={() => downloadCsv('student-upload-errors.csv', validationErrorsCsv(result.errors))} className={`${SECONDARY_BUTTON} mt-4`}>
                <Download size={15} aria-hidden="true" /> Download Error Report
              </button>
            </div>
          )}
          {results.length > 0 && (
            <div className="mt-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="font-semibold text-slate-900">Saved predictions and recommendations</h3>
                  <p className="mt-1 text-sm text-slate-600" aria-live="polite">
                    Showing {filteredResults.length} of {results.length} successful result(s).
                  </p>
                </div>
                <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row">
                  <label className="flex min-h-10 w-full items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 shadow-sm focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 sm:w-auto">
                    <Search size={15} aria-hidden="true" />
                    <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search results" aria-label="Search imported results by student ID, name, or risk category" className="w-full bg-transparent outline-none sm:w-40" />
                  </label>
                  <button type="button" onClick={() => downloadCsv('student-prediction-results.csv', successfulResultsCsv(results))} className={`${SECONDARY_BUTTON} w-full sm:w-auto`}>
                    <Download size={15} aria-hidden="true" /> Download Results CSV
                  </button>
                </div>
              </div>
              <div className="mt-3 max-h-[28rem] overflow-auto rounded-[18px] border border-slate-200" tabIndex={0} role="region" aria-label="Scrollable imported student results table">
                <table className="ui-table w-full min-w-[760px] text-left text-sm">
                  <caption className="sr-only">Successful imported student predictions and recommendations</caption>
                  <thead className="sticky top-0 z-10 bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                    <tr><th scope="col" className="px-4 py-3">Student ID</th><th scope="col" className="px-4 py-3">Name</th><th scope="col" className="px-4 py-3 text-right">Predicted marks</th><th scope="col" className="px-4 py-3">Risk category</th><th scope="col" className="px-4 py-3">Recommendations</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredResults.map((row) => (
                      <tr key={row.prediction_id}>
                        <td className="px-4 py-3 font-medium text-slate-900">{row.student_id}</td>
                        <td className="px-4 py-3">{row.student_name}</td>
                        <td className="whitespace-nowrap px-4 py-3 text-right font-semibold tabular-nums">{row.predicted_marks} <span className="font-normal text-slate-500">/ 100</span></td>
                        <td className="px-4 py-3"><span className={`ui-badge ${riskBadgeClass(row.risk_level)}`}>{row.risk_level}</span></td>
                        <td className="px-4 py-3">
                          {row.recommendations.length === 0 ? 'None' : (
                            <details>
                              <summary className="cursor-pointer font-medium text-blue-700">{row.recommendations.length} available</summary>
                              <ul className="mt-2 max-w-md space-y-2 text-xs text-slate-600">
                                {row.recommendations.map((item) => <li key={`${item.feature}-${item.priority}`}><strong>{item.priority} · {item.feature}:</strong> {item.recommendation} <span className="block">{item.action}</span></li>)}
                              </ul>
                            </details>
                          )}
                        </td>
                      </tr>
                    ))}
                    {filteredResults.length === 0 && <tr><td colSpan="5" className="px-4 py-6 text-center text-slate-500">No successful results match this search. Clear the search to show all successful imports.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {results.length === 0 && (
            <div className="mt-6 rounded-[18px] border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700" role="status">
              No successful prediction rows were returned. Review the rejected-row details above, if available, or upload a corrected file.
            </div>
          )}
        </section>
      )}
    </div>
  )
}

export default BulkUploadView
