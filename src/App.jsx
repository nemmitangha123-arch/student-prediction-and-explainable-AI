import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import jsPDF from 'jspdf'
import { BrowserRouter, Routes, Route, Link, Navigate, NavLink, Outlet, useLocation, useNavigate, useOutletContext, useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import BulkUploadView from './BulkUploadView'
import CreateStudentView from './CreateStudentView'
import BulkStudentRegistrationView from './BulkStudentRegistrationView'
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowRightLeft,
  ArrowUp,
  House,
  ListChecks,
  LoaderCircle,
  Minus,
  RotateCcw,
  BadgeCheck,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleUserRound,
  FileText,
  FlaskConical,
  GraduationCap,
  LayoutDashboard,
  Search,
  ShieldCheck,
  Sparkles,
  UserRound,
  Wand2,
  BarChart3,
  ClipboardList,
  Eye,
  Users,
  BrainCog,
  CircleAlert,
  EyeOff,
  Upload,
  UserPlus,
} from 'lucide-react'
import { API_BASE_URL, apiRequest, clearStoredToken, getStoredToken, setUnauthorizedHandler, storeToken } from './api'
import {
  accountStatusLabel,
  requestStudentAccountStatusChange,
} from './studentAccountManagement'

const BTN_PRIMARY = 'ui-btn ui-btn-primary'
const BTN_SECONDARY = 'ui-btn ui-btn-secondary'
const FIELD_INPUT = 'ui-field mt-2'
const CARD = 'ui-card p-5 sm:p-6'

const PREDICTION_CONNECTION_ERROR = 'Unable to connect to the prediction server. Please make sure the Flask backend is running.'

// Early-warning display styles, keyed by the backend's risk_code.
// The boundaries themselves live only in the backend (classify_early_warning).
const EARLY_WARNING_STYLES = {
  high_concern: {
    badge: 'border-rose-200 bg-rose-50 text-rose-700',
    icon: CircleAlert,
    meaning: 'The predicted final-exam marks fall in the project\'s highest early-warning band. Timely academic support and intervention are suggested.',
  },
  borderline: {
    badge: 'border-amber-200 bg-amber-50 text-amber-700',
    icon: Eye,
    meaning: 'The predicted final-exam marks are close to the lower early-warning boundary. Closer monitoring and targeted support are suggested.',
  },
  lower_concern: {
    badge: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    icon: CheckCircle2,
    meaning: 'The predicted final-exam marks are above the early-warning bands. Continue regular progress monitoring.',
  },
}

function EarlyWarningBadge({ risk }) {
  const style = risk && EARLY_WARNING_STYLES[risk.risk_code]
  if (!style) {
    return <span className="ui-badge border-slate-200 bg-slate-50 text-slate-600">Not available</span>
  }
  const Icon = style.icon
  return (
    <span className={`ui-badge text-sm ${style.badge}`} aria-label={`Early-warning category: ${risk.risk_level}`}>
      <Icon size={16} aria-hidden="true" />
      {risk.risk_level}
    </span>
  )
}

// The five model inputs with their training-supported ranges (matches backend validation)
const MODEL_INPUT_FIELDS = [
  { key: 'attendance', label: 'Attendance (%)', min: 52, max: 100, step: 1 },
  { key: 'test1', label: 'Internal Test 1 (out of 40)', min: 18, max: 40, step: 1 },
  { key: 'test2', label: 'Internal Test 2 (out of 40)', min: 16, max: 40, step: 1 },
  { key: 'assignment', label: 'Assignment Score (out of 10)', min: 4, max: 10, step: 1 },
  { key: 'study_hours', label: 'Daily Study Hours', min: 1, max: 5, step: 0.5 },
]

// The one live prediction call used by the Prediction page and Teacher Reports:
// /predict, /explain and /recommend for the same five inputs. Throws on failure;
// an error with isServerResponse carries the server's own message.
async function fetchLivePrediction(payload) {
  const [predictRes, explainRes, recommendRes] = await Promise.all([
    fetch(`${API_BASE_URL}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
    fetch(`${API_BASE_URL}/explain`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
    fetch(`${API_BASE_URL}/recommend`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  ])

  const failedRes = [predictRes, explainRes, recommendRes].find((res) => !res.ok)
  if (failedRes) {
    // The server responded, so surface its message instead of a connection error
    const errorJson = await failedRes.json().catch(() => null)
    const serverError = new Error(errorJson?.message || `The prediction server returned an error (HTTP ${failedRes.status}).`)
    serverError.isServerResponse = true
    throw serverError
  }

  return { predict: await predictRes.json(), explain: await explainRes.json(), recommend: await recommendRes.json() }
}

// Calls /predict for a scenario (debounced) and returns the API result.
// Each response is stored with the scenario it was requested for, so a result
// from a previous scenario is never returned for the currently selected one.
function useScenarioPrediction(scenario) {
  const [response, setResponse] = useState({ key: null, result: null, error: '' })
  // Incremented by retry() to request the same scenario again
  const [attempt, setAttempt] = useState(0)
  const scenarioKey = JSON.stringify(scenario)

  useEffect(() => {
    const controller = new AbortController()
    const requestKey = JSON.stringify(scenario)
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: requestKey,
          signal: controller.signal,
        })
        const json = await res.json().catch(() => null)
        if (!res.ok) {
          setResponse({ key: requestKey, result: null, error: json?.message || `The prediction server returned an error (HTTP ${res.status}).` })
        } else {
          setResponse({ key: requestKey, result: json, error: '' })
        }
      } catch (err) {
        if (err.name === 'AbortError') return
        setResponse({ key: requestKey, result: null, error: PREDICTION_CONNECTION_ERROR })
      }
    }, 300)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [scenario, attempt])

  const retry = useCallback(() => {
    setResponse({ key: null, result: null, error: '' })
    setAttempt((count) => count + 1)
  }, [])

  // Loading from the moment the scenario changes (including the debounce wait)
  // until the response for that exact scenario arrives
  const isCurrent = response.key === scenarioKey
  return {
    result: isCurrent ? response.result : null,
    error: isCurrent ? response.error : '',
    isLoading: !isCurrent,
    retry,
  }
}

// Five predefined demonstration profiles: starting inputs only. They are not real
// students; every result shown for them comes from the live /predict, /explain and /recommend calls.
const DEMO_STUDENTS = [
  { id: 1, name: 'Demo Student 1', inputs: { attendance: 92, test1: 36, test2: 37, assignment: 9, study_hours: 4 } },
  { id: 2, name: 'Demo Student 2', inputs: { attendance: 80, test1: 30, test2: 28, assignment: 8, study_hours: 3 } },
  { id: 3, name: 'Demo Student 3', inputs: { attendance: 70, test1: 24, test2: 25, assignment: 7, study_hours: 2 } },
  { id: 4, name: 'Demo Student 4', inputs: { attendance: 65, test1: 24, test2: 22, assignment: 6, study_hours: 2 } },
  { id: 5, name: 'Demo Student 5', inputs: { attendance: 58, test1: 20, test2: 19, assignment: 5, study_hours: 1 } },
]

const DemoStudentsContext = createContext(null)

// Practice pages (Prediction, What-If): the sample profile in use and its latest practice result
// (the submitted inputs plus the raw /predict, /explain and /recommend responses).
// React state only: nothing is persisted, so reloading the page starts afresh.
function DemoStudentsProvider({ children }) {
  const [selectedId, setSelectedId] = useState(DEMO_STUDENTS[0].id)
  const [results, setResults] = useState({})
  const latestRequest = useRef({})

  // Only the most recent request made for a Demo Student may store its result
  const beginRequest = useCallback((studentId) => {
    const token = (latestRequest.current[studentId] ?? 0) + 1
    latestRequest.current[studentId] = token
    return token
  }, [])

  const recordResult = useCallback((studentId, token, entry) => {
    if (latestRequest.current[studentId] !== token) return
    setResults((prev) => ({ ...prev, [studentId]: entry }))
  }, [])

  const value = useMemo(
    () => ({ selectedId, setSelectedId, results, beginRequest, recordResult }),
    [selectedId, results, beginRequest, recordResult],
  )
  return <DemoStudentsContext.Provider value={value}>{children}</DemoStudentsContext.Provider>
}

function useDemoStudents() {
  return useContext(DemoStudentsContext)
}

const AuthContext = createContext(null)

// The signed-in user. On startup a stored token is checked with GET /auth/me:
// status is 'checking' until then, 'signedIn' or 'signedOut' afterwards, and
// 'unreachable' if the server could not be asked (the token is kept to retry).
function AuthProvider({ children }) {
  const navigate = useNavigate()
  const [auth, setAuth] = useState(() => ({ status: getStoredToken() ? 'checking' : 'signedOut', user: null }))
  const [restoreAttempt, setRestoreAttempt] = useState(0)

  // A rejected token (expired, or signed out elsewhere) ends the session and returns to /login
  useEffect(() => {
    setUnauthorizedHandler(() => {
      // expired is kept in state as well: a portal's own redirect to /login may replace this navigation's state
      setAuth({ status: 'signedOut', user: null, expired: true })
      navigate('/login', { replace: true, state: { sessionExpired: true } })
    })
    return () => setUnauthorizedHandler(null)
  }, [navigate])

  useEffect(() => {
    if (!getStoredToken()) return undefined
    let cancelled = false
    apiRequest('/auth/me')
      .then((data) => { if (!cancelled) setAuth({ status: 'signedIn', user: data.user }) })
      .catch((err) => {
        if (cancelled) return
        if (err.status === 0) setAuth({ status: 'unreachable', user: null })
        // Any other failure means the token is unusable; apiRequest already handled a 401
        else if (err.status !== 401) {
          clearStoredToken()
          setAuth({ status: 'signedOut', user: null })
        }
      })
    return () => { cancelled = true }
  }, [restoreAttempt])

  const retryRestore = useCallback(() => {
    setAuth({ status: 'checking', user: null })
    setRestoreAttempt((count) => count + 1)
  }, [])

  const login = useCallback(async (username, password) => {
    const data = await apiRequest('/auth/login', { method: 'POST', body: { username, password }, auth: false })
    storeToken(data.token)
    setAuth({ status: 'signedIn', user: data.user })
    return data.user
  }, [])

  const logout = useCallback(async () => {
    // The local session ends even if the server cannot be reached
    await apiRequest('/auth/logout', { method: 'POST' }).catch(() => null)
    clearStoredToken()
    setAuth({ status: 'signedOut', user: null })
  }, [])

  const value = useMemo(() => ({ ...auth, login, logout, retryRestore }), [auth, login, logout, retryRestore])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

function useAuth() {
  return useContext(AuthContext)
}

// Where each role starts after signing in
const ROLE_HOME = { faculty: '/faculty', student: '/student' }
const ROLE_PORTAL_NAME = { faculty: 'Teacher Portal', student: 'Student Portal' }

// Shows a portal only to a signed-in user with the given role. This keeps the
// frontend behaving correctly; the backend still checks every request.
function RequireRole({ role, children }) {
  const { status, user, logout, retryRestore } = useAuth()
  const shell = (content) => <div className="flex min-h-screen items-center justify-center p-4"><div className="w-full max-w-lg">{content}</div></div>

  if (status === 'checking') {
    return shell(<p role="status" className="flex items-center justify-center gap-2 text-sm text-slate-600"><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Checking your session…</p>)
  }
  if (status === 'unreachable') {
    return shell(<ServiceErrorNotice title="Server unavailable" message="Your session could not be checked because the server could not be reached. Start the Flask backend and try again." onRetry={retryRestore} />)
  }
  if (status !== 'signedIn') return <Navigate to="/login" replace />
  if (user.role !== role) {
    return shell(
      <div className={CARD}>
        <h1 className="text-xl font-semibold text-slate-900">This portal is not available for your account</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">You are signed in as <span className="font-semibold text-slate-900">{user.username}</span> ({user.role}). The {ROLE_PORTAL_NAME[role]} is only for {role} accounts.</p>
        <div className="mt-5 flex flex-wrap gap-3">
          <NavLink to={ROLE_HOME[user.role] || '/'} className={BTN_PRIMARY}>Go to the {ROLE_PORTAL_NAME[user.role]} <ArrowRight size={16} aria-hidden="true" /></NavLink>
          <button type="button" onClick={logout} className={BTN_SECONDARY}>Sign out</button>
        </div>
      </div>,
    )
  }
  return children
}

// GET request for a protected backend resource (null path = nothing to load).
// Each response is stored with the path it was requested for, so data for a
// previously selected student is never shown for the current one.
function useApiResource(path) {
  const [response, setResponse] = useState({ path: null, data: null, error: null })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!path) return undefined
    let cancelled = false
    apiRequest(path)
      .then((data) => { if (!cancelled) setResponse({ path, data, error: null }) })
      .catch((err) => { if (!cancelled) setResponse({ path, data: null, error: err }) })
    return () => { cancelled = true }
  }, [path, attempt])

  const reload = useCallback(() => {
    setResponse({ path: null, data: null, error: null })
    setAttempt((count) => count + 1)
  }, [])

  // Shows newer data for the same path (e.g. a save's response) without a loading state
  const replace = useCallback((data) => {
    setResponse({ path, data, error: null })
  }, [path])

  // Re-reads the same path in the background; the current data stays visible meanwhile
  const refresh = useCallback(() => {
    if (!path) return Promise.resolve()
    return apiRequest(path)
      .then((data) => setResponse((prev) => (prev.path === path ? { path, data, error: null } : prev)))
      .catch(() => {})
  }, [path])

  const isCurrent = response.path === path
  return {
    data: isCurrent ? response.data : null,
    error: isCurrent ? response.error : null,
    isLoading: Boolean(path) && !isCurrent,
    reload,
    replace,
    refresh,
  }
}

// Wording and style for the backend's prediction_status.status values
const PREDICTION_STATUS = {
  current: { label: 'Prediction available', badge: 'border-emerald-200 bg-emerald-50 text-emerald-700', icon: CheckCircle2 },
  outdated: { label: 'Record changed — prediction needs to be regenerated', badge: 'border-amber-200 bg-amber-50 text-amber-800', icon: RotateCcw },
  awaiting_study_hours: { label: 'Waiting for student study hours', badge: 'border-sky-200 bg-sky-50 text-sky-700', icon: ListChecks },
  outside_model_range: { label: 'Prediction unavailable — academic record is outside the model\'s supported range', badge: 'border-rose-200 bg-rose-50 text-rose-700', icon: CircleAlert },
  not_predicted: { label: 'No prediction generated yet', badge: 'border-slate-200 bg-slate-50 text-slate-700', icon: Sparkles },
  incomplete_record: { label: 'Academic record incomplete', badge: 'border-slate-200 bg-slate-50 text-slate-700', icon: ClipboardList },
}

function PredictionStatusBadge({ status }) {
  const style = PREDICTION_STATUS[status]
  if (!style) return null
  const Icon = style.icon
  return (
    <span className={`ui-badge ${style.badge}`}>
      <Icon size={14} aria-hidden="true" />
      {style.label}
    </span>
  )
}

// The latest stored prediction counts as the student's prediction only while it matches the current record
const hasCurrentPrediction = (student) => student.prediction_status.status === 'current' && student.latest_prediction != null

const formatDateTime = (iso) => (iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : null)

// Loading and error states shared by every view that reads student data from the backend.
// A 401 is already handled by apiRequest (sign-in again), so it only needs a short message.
function ResourceState({ resource, what }) {
  if (resource.isLoading) {
    return <p role="status" className="flex items-center gap-2 text-sm text-slate-600"><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Loading {what}…</p>
  }
  if (resource.error) {
    return resource.error.status === 0
      ? <ServiceErrorNotice title="Server unavailable" message={`The ${what} could not be loaded because the server could not be reached.`} onRetry={resource.reload} />
      : <ServiceErrorNotice title={`Could not load the ${what}`} message={resource.error.message} onRetry={resource.reload} />
  }
  return null
}

// Ranges a real stored value may have (the backend's ACADEMIC_RANGES). They are wider than
// the model's training-supported ranges (MODEL_INPUT_FIELDS), which only the backend checks,
// and only when a prediction is requested.
const ACADEMIC_FIELDS = [
  { key: 'attendance', label: 'Attendance (%)', min: 0, max: 100 },
  { key: 'test1', label: 'Internal Test 1 (out of 40)', min: 0, max: 40 },
  { key: 'test2', label: 'Internal Test 2 (out of 40)', min: 0, max: 40 },
  { key: 'assignment', label: 'Assignment Score (out of 10)', min: 0, max: 10 },
]
const STUDY_HOURS_FIELD = { key: 'study_hours', label: 'Daily Study Hours', min: 0, max: 24 }

// One entered value checked against its academic range: [number, null] or [null, message]
function parseAcademicValue(raw, { label, min, max }) {
  if (String(raw).trim() === '') return [null, `Enter a value for ${label}.`]
  const value = Number(raw)
  if (!Number.isFinite(value)) return [null, `${label} must be a number.`]
  if (value < min || value > max) return [null, `${label} must be between ${min} and ${max}.`]
  return [value, null]
}

const modelRangeText = (key) => {
  const field = MODEL_INPUT_FIELDS.find((item) => item.key === key)
  return field ? `model trained on ${field.min}–${field.max}` : ''
}

function FormError({ message }) {
  if (!message) return null
  return (
    <div role="alert" className="mt-4 flex items-start gap-2 rounded-[18px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
      <CircleAlert size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <p>{message}</p>
    </div>
  )
}

// Faculty: the four faculty-provided values, saved with PUT /students/<id>/academic-record.
// Out-of-model-range values are valid academic values and are saved as entered.
function AcademicRecordForm({ studentId, record, onSaved, onCancel }) {
  const [values, setValues] = useState(() => Object.fromEntries(ACADEMIC_FIELDS.map(({ key }) => [key, record[key] ?? ''])))
  const [error, setError] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (isSaving) return
    const body = {}
    for (const field of ACADEMIC_FIELDS) {
      const [value, message] = parseAcademicValue(values[field.key], field)
      if (message) return setError(message)
      body[field.key] = value
    }
    setError('')
    setIsSaving(true)
    try {
      onSaved(await apiRequest(`/students/${studentId}/academic-record`, { method: 'PUT', body }))
    } catch (err) {
      setError(err.message)
      setIsSaving(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate data-testid="academic-record-form">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {ACADEMIC_FIELDS.map(({ key, label, min, max }) => (
          <div key={key} className="rounded-[18px] border border-slate-200 bg-white p-4">
            <label htmlFor={`record-${key}`} className="block text-sm font-medium text-slate-700 xl:min-h-[2.5rem]">{label}</label>
            <input id={`record-${key}`} type="number" inputMode="decimal" min={min} max={max} step="any" placeholder="Not entered" value={values[key]} onChange={(event) => setValues((prev) => ({ ...prev, [key]: event.target.value }))} disabled={isSaving} className={FIELD_INPUT} />
            <p className="mt-1 text-xs text-slate-500">Allowed {min}–{max} · {modelRangeText(key)}</p>
            {record[key] == null && <p className="mt-1 text-xs font-semibold text-amber-700">Currently not entered</p>}
          </div>
        ))}
      </div>
      <FormError message={error} />
      <div className="mt-5 flex flex-wrap gap-3">
        <button type="submit" disabled={isSaving} aria-busy={isSaving} className={BTN_PRIMARY}>
          {isSaving ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Saving…</> : 'Save academic record'}
        </button>
        <button type="button" onClick={onCancel} disabled={isSaving} className={BTN_SECONDARY}>Cancel</button>
      </div>
    </form>
  )
}

// Student: their own Daily Study Hours, saved with PUT /students/<own id>/study-hours.
// Any real value from 0 to 24 is accepted; the model's 1–5 range matters only for prediction.
function StudyHoursForm({ studentId, current, onSaved }) {
  const [value, setValue] = useState(current ?? '')
  const [error, setError] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const unchanged = String(value).trim() !== '' && Number(value) === current

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (isSaving) return
    const [hours, message] = parseAcademicValue(value, STUDY_HOURS_FIELD)
    if (message) return setError(message)
    setError('')
    setSaved(false)
    setIsSaving(true)
    try {
      onSaved(await apiRequest(`/students/${studentId}/study-hours`, { method: 'PUT', body: { study_hours: hours } }))
      setSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="mt-5 rounded-[18px] border border-blue-100 bg-blue-50/50 p-4" data-testid="study-hours-form">
      <label htmlFor="study-hours" className="text-sm font-semibold text-slate-700">Your Daily Study Hours</label>
      <p className="text-xs text-slate-500">Currently: {current ?? 'not entered'} · allowed 0–24 · {modelRangeText('study_hours')}</p>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <input id="study-hours" type="number" inputMode="decimal" min={0} max={24} step="any" placeholder="Not entered" value={value} onChange={(event) => { setValue(event.target.value); setSaved(false) }} disabled={isSaving} className={`${FIELD_INPUT} mt-0 w-36`} />
        <button type="submit" disabled={isSaving || unchanged} aria-busy={isSaving} className={BTN_PRIMARY}>
          {isSaving ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Saving…</> : current == null ? 'Save study hours' : 'Update study hours'}
        </button>
        {saved && <span role="status" className="text-sm font-semibold text-emerald-700">Saved</span>}
      </div>
      <FormError message={error} />
    </form>
  )
}

// What the signed-in user can do about each blocked or stale status
function statusGuidance(status, record, role) {
  if (status === 'incomplete_record') {
    const missing = ACADEMIC_FIELDS.filter(({ key }) => record[key] == null).map(({ label }) => label)
    return `Missing: ${missing.join(', ')}. ${role === 'faculty' ? 'Use "Edit academic record" below to complete it.' : 'Your faculty needs to enter these marks before a prediction can be generated.'}`
  }
  if (status === 'awaiting_study_hours') {
    return role === 'student' ? 'Enter your Daily Study Hours in the Academic record section below.' : 'The student needs to enter their Daily Study Hours before a prediction can be generated.'
  }
  if (status === 'outside_model_range') {
    return 'The stored values are valid academic values and are kept exactly as entered. The current model was trained on a narrower range, so it cannot produce a reliable prediction for this record.'
  }
  if (status === 'not_predicted') return 'The record is complete and within the model\'s supported range. Generate a prediction from the stored record.'
  if (status === 'outdated') return 'The record has changed since the last prediction. Generate a new prediction from the current record.'
  return null
}

// One student's stored academic record, prediction status and latest stored prediction,
// as returned by GET /students/<id>. Faculty can edit the four marks, the student their
// Daily Study Hours, and either can generate a prediction from the stored record.
// onDetailChange receives the updated GET /students/<id> shape after every change.
function StudentRecordDetails({ detail, onDetailChange }) {
  const { user } = useAuth()
  const { student, prediction_status: status, latest_prediction: latest } = detail
  const record = student.record
  const isCurrent = status.status === 'current'
  const recommendations = latest ? mapRecommendations(latest.recommendations) : []
  // A student only ever acts on their own record, identified by their account
  const targetId = user.role === 'student' ? user.student_id : student.id
  const [isEditing, setIsEditing] = useState(false)
  const [notice, setNotice] = useState('')
  const [recordNotice, setRecordNotice] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generateError, setGenerateError] = useState('')
  const canGenerate = ['not_predicted', 'outdated'].includes(status.status)
  const guidance = statusGuidance(status.status, record, user.role)
  // Faculty: the intervention being written, possibly started from a stored recommendation
  const [interventionDraft, setInterventionDraft] = useState(null)

  // Prefills a new intervention from a recommendation stored with the latest prediction
  // (nothing is regenerated: the text and the prediction id come from the stored data)
  const renderCreateFromRecommendation = (rec) => (rec.needsAttention ? (
    <CreateInterventionButton onClick={() => setInterventionDraft(draftFromRecommendation(rec, latest.id))} />
  ) : null)

  const updates = [
    record.faculty_updated_at && `Marks last updated${record.faculty_updated_by === user.id ? ' by you' : record.faculty_updated_by != null ? ' by faculty' : ''} ${formatDateTime(record.faculty_updated_at)}`,
    record.study_hours_updated_at && `Study hours last updated${record.study_hours_updated_by === user.id ? ' by you' : record.study_hours_updated_by != null ? ' by the student' : ''} ${formatDateTime(record.study_hours_updated_at)}`,
  ].filter(Boolean)

  // POST /students/<id>/predictions with no body: the backend predicts from the stored record only
  const generatePrediction = async () => {
    if (isGenerating) return
    setGenerateError('')
    setNotice('')
    setIsGenerating(true)
    try {
      await apiRequest(`/students/${targetId}/predictions`, { method: 'POST' })
      onDetailChange(await apiRequest(`/students/${targetId}`))
      setNotice('Prediction generated from the stored academic record.')
    } catch (err) {
      setGenerateError(err.message)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="space-y-6">
      <section className="ui-card flex flex-wrap items-center justify-between gap-3 p-4 sm:p-5" aria-label="Selected student record">
        <div className="min-w-0">
          <p className="ui-eyebrow">Student record</p>
          <h2 className="mt-1 truncate text-lg font-semibold text-slate-900">{student.name}</h2>
          <p className="mt-1 text-sm text-slate-600">
            Student ID: <span className="font-semibold text-slate-800">{student.roll_no}</span>
            {[student.program, student.section && `Section ${student.section}`].filter(Boolean).map((item) => ` · ${item}`).join('')}
          </p>
        </div>
        {status.status === 'current' && latest && (
          <span className="ui-badge border-emerald-200 bg-emerald-50 text-emerald-800">
            <CheckCircle2 size={14} aria-hidden="true" /> Current prediction
          </span>
        )}
      </section>

      <section className={CARD} data-testid="prediction-status">
        <SectionHeading title="Prediction status" />
        <PredictionStatusBadge status={status.status} />
        <p className="mt-3 text-sm leading-6 text-slate-600">{status.message}</p>
        {guidance && <p className="mt-2 text-sm leading-6 text-slate-700" data-testid="status-guidance">{guidance}</p>}
        {canGenerate && (
          <button type="button" onClick={generatePrediction} disabled={isGenerating} aria-busy={isGenerating} className={`${BTN_PRIMARY} mt-4`}>
            {isGenerating ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Generating…</> : <><Sparkles size={16} aria-hidden="true" /> Generate Prediction</>}
          </button>
        )}
        {notice && <p role="status" className="mt-4 rounded-[18px] border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-800">{notice}</p>}
        <FormError message={generateError} />
      </section>

      <section className={CARD} data-testid="academic-record">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <SectionHeading title="Academic record" description="Attendance and marks are entered by faculty; Daily Study Hours is entered by the student." />
          {user.role === 'faculty' && !isEditing && (
            <button type="button" onClick={() => { setIsEditing(true); setRecordNotice('') }} className={BTN_SECONDARY}>Edit academic record</button>
          )}
        </div>
        {isEditing ? (
          <AcademicRecordForm
            studentId={targetId}
            record={record}
            onCancel={() => setIsEditing(false)}
            onSaved={(updated) => {
              onDetailChange(updated)
              setIsEditing(false)
              setNotice('')
              setRecordNotice('Academic record saved.')
            }}
          />
        ) : (
          <InputsList inputs={record} wide />
        )}
        {recordNotice && !isEditing && <p role="status" className="mt-4 rounded-[18px] border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-800">{recordNotice}</p>}
        {updates.length > 0 && <p className="mt-3 text-xs leading-5 text-slate-500" data-testid="record-updates">{updates.join(' · ')}</p>}
        {user.role === 'student' && <StudyHoursForm studentId={targetId} current={record.study_hours} onSaved={onDetailChange} />}
      </section>

      {latest && (
        <>
          <section className={CARD} data-testid="latest-prediction">
            <SectionHeading title={isCurrent ? 'Latest prediction' : 'Last generated prediction'} description={`Generated ${formatDateTime(latest.created_at)}${latest.created_by_role ? ` by ${latest.created_by_role}` : ''}.`} />
            {!isCurrent && (
              <p role="note" className="mb-4 rounded-[18px] border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900">
                This prediction was generated from an earlier version of the academic record, so it does not describe the current record.
              </p>
            )}
            <div className="grid gap-3 lg:grid-cols-[1.25fr_1fr]">
              <div className="rounded-[20px] border border-blue-200 bg-blue-50/60 p-5 sm:p-6">
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-blue-800">Predicted final-exam marks</p>
                <p className="mt-2 text-4xl font-bold tracking-tight text-slate-950 sm:text-5xl" data-testid="latest-marks">
                  {latest.predicted_marks}<span className="ml-1 text-xl font-semibold text-slate-600 sm:text-2xl">/ 100</span>
                </p>
                <p className="mt-2 text-sm font-medium text-slate-700">Model estimate — not actual examination marks.</p>
              </div>
              <div className="flex flex-col justify-center rounded-[20px] border border-slate-200 bg-slate-50 p-5 sm:p-6">
                <p className="text-xs font-bold uppercase tracking-[0.16em] text-slate-500">Early-warning category</p>
                <div className="mt-2"><EarlyWarningBadge risk={latest} /></div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{latest.risk_basis}. {EARLY_WARNING_STYLES[latest.risk_code]?.meaning}</p>
              </div>
            </div>
            {latest.risk_disclaimer && <p className="mt-2 text-xs leading-6 text-slate-500">{latest.risk_disclaimer}</p>}
            <p className="mb-2 mt-5 text-sm font-semibold text-slate-500">Inputs used for this prediction</p>
            <InputsList inputs={latest.inputs} wide />
          </section>

          <section className={CARD}>
            <SectionHeading title="What contributed to this prediction?" description="SHAP contributions (in marks) stored with this prediction. They describe the model's attribution, not cause and effect." />
            <ShapContributionBars contributions={latest.shap_contributions} baseValue={latest.base_value} predictedMarks={latest.predicted_marks} />
          </section>

          <section className={CARD}>
            <SectionHeading title="Recommended Actions" description="Stored with this prediction; generated from its SHAP contributions. They are academic guidance and do not establish cause and effect." />
            {recommendations.length === 0
              ? <p className="text-sm text-slate-600">No specific intervention recommendations were generated for this prediction.</p>
              : <RecommendationGroups recommendations={recommendations} strongest={findStrongestNegative(recommendations)} renderFooter={user.role === 'faculty' ? renderCreateFromRecommendation : undefined} />}
          </section>
        </>
      )}

      {user.role === 'faculty' && <FacultyInterventions studentId={targetId} draft={interventionDraft} onDraftChange={setInterventionDraft} />}

      <PredictionHistory studentId={targetId} version={`${latest?.id ?? 'none'}|${MODEL_INPUT_FIELDS.map(({ key }) => record[key]).join(',')}`} />
    </div>
  )
}

// Current / Outdated for one stored prediction, as decided by the backend (is_current)
function HistoryStatusBadge({ isCurrent }) {
  const Icon = isCurrent ? CheckCircle2 : RotateCcw
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${isCurrent ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-800'}`}>
      <Icon size={13} aria-hidden="true" />
      {isCurrent ? 'Current' : 'Outdated'}
    </span>
  )
}

// One stored prediction exactly as it was generated: its own inputs, marks, SHAP and recommendations
function HistoryDetails({ prediction }) {
  const recommendations = mapRecommendations(prediction.recommendations)
  return (
    <div className="space-y-5" data-testid={`history-details-${prediction.id}`}>
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-3xl font-bold tracking-tight text-slate-950">
          {prediction.predicted_marks}<span className="ml-1 text-base font-semibold text-slate-500">/ 100 predicted marks</span>
        </p>
        <EarlyWarningBadge risk={prediction} />
        <HistoryStatusBadge isCurrent={prediction.is_current} />
      </div>
      <p className="text-sm text-slate-600">Model estimate, not actual examination marks.</p>
      <p className="text-sm leading-6 text-slate-700">{prediction.risk_basis}. Generated {formatDateTime(prediction.created_at)}{prediction.created_by_role ? ` by ${prediction.created_by_role}` : ''}.</p>
      {!prediction.is_current && (
        <p className="rounded-[16px] border border-amber-200 bg-amber-50 px-3 py-2 text-sm leading-6 text-amber-900">The academic record has changed since this prediction. The inputs below are the values it was generated from, not the current record.</p>
      )}
      <div>
        <p className="mb-2 text-sm font-semibold text-slate-500">Inputs used for this prediction</p>
        <InputsList inputs={prediction.inputs} wide />
      </div>
      <div>
        <p className="mb-3 text-sm font-semibold text-slate-500">SHAP contributions (marks)</p>
        <ShapContributionBars contributions={prediction.shap_contributions} baseValue={prediction.base_value} predictedMarks={prediction.predicted_marks} />
      </div>
      <div>
        <p className="mb-3 text-sm font-semibold text-slate-500">Recommendations</p>
        {recommendations.length === 0
          ? <p className="text-sm text-slate-600">No specific intervention recommendations were generated for this prediction.</p>
          : <RecommendationGroups recommendations={recommendations} strongest={findStrongestNegative(recommendations)} />}
      </div>
    </div>
  )
}

// Every stored prediction of one student, newest first, from GET /students/<id>/predictions.
// version changes whenever a prediction is generated or the record changes, so the history
// (and its backend-computed current/outdated flags) is read again.
function PredictionHistory({ studentId, version }) {
  const history = useApiResource(`/students/${studentId}/predictions`)
  const { refresh } = history
  const loadedVersion = useRef(version)
  const [openId, setOpenId] = useState(null)

  useEffect(() => {
    if (version === loadedVersion.current) return
    loadedVersion.current = version
    refresh()
  }, [version, refresh])

  const predictions = history.data?.predictions
  return (
    <section className={CARD} data-testid="prediction-history">
      <SectionHeading
        title="Prediction History"
        description={predictions?.length
          ? `${predictions.length} stored prediction${predictions.length === 1 ? '' : 's'}, newest first. Each keeps the inputs it was generated from.`
          : 'Every generated prediction is stored with the inputs it was generated from.'}
      />
      {!predictions ? (
        <ResourceState resource={history} what="prediction history" />
      ) : predictions.length === 0 ? (
        <p className="rounded-[18px] border border-dashed border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">No predictions have been generated for this student yet.</p>
      ) : (
        // relative: keeps the absolutely positioned sr-only header inside this scroll area on narrow screens
        <div className="relative overflow-x-auto rounded-xl border border-slate-200" tabIndex={0} role="region" aria-label="Scrollable prediction history table">
          <table className="ui-table w-full min-w-[36rem] text-left text-sm">
            <caption className="sr-only">Prediction history, newest first, with risk category and current status</caption>
            <thead className="sticky top-0 z-10">
              <tr className="border-b border-slate-200 text-xs uppercase tracking-[0.15em] text-slate-500">
                <th scope="col" className="py-2 pr-3 font-semibold">Date</th>
                <th scope="col" className="py-2 pr-3 text-right font-semibold">Predicted Marks</th>
                <th scope="col" className="py-2 pr-3 font-semibold">Risk</th>
                <th scope="col" className="py-2 pr-3 font-semibold">Status</th>
                <th scope="col" className="py-2 font-semibold"><span className="sr-only">Details</span></th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((prediction, index) => {
                const open = openId === prediction.id
                return (
                  <tr key={prediction.id} className={`border-b border-slate-100 align-middle ${open ? 'bg-blue-50/50' : ''}`} data-history-row={prediction.id}>
                    <td className="py-3 pr-3 text-slate-700">
                      {formatDateTime(prediction.created_at)}
                      {index === 0 && <span className="ml-2 rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-semibold text-blue-700">Latest</span>}
                    </td>
                    <td className="py-3 pr-3 text-right font-semibold text-slate-900">{prediction.predicted_marks}</td>
                    <td className="py-3 pr-3"><EarlyWarningBadge risk={prediction} /></td>
                    <td className="py-3 pr-3"><HistoryStatusBadge isCurrent={prediction.is_current} /></td>
                    <td className="py-3 text-right">
                      <button type="button" aria-expanded={open} aria-controls="history-details" onClick={() => setOpenId(open ? null : prediction.id)} className="inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-sm font-semibold text-blue-700 transition hover:bg-blue-50">
                        {open ? 'Hide details' : 'View details'}
                        <ChevronRight size={14} className={`transition ${open ? 'rotate-90' : ''}`} aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      {/* The selected prediction's details sit below the table, so they use the full card width on any screen */}
      {predictions?.some((prediction) => prediction.id === openId) && (
        <div id="history-details" className="mt-4 rounded-[20px] border border-slate-200 bg-slate-50/60 p-4 sm:p-5">
          <HistoryDetails prediction={predictions.find((prediction) => prediction.id === openId)} />
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
// Interventions (improvement tasks) stored in the database
// ---------------------------------------------------------------------------

// Must match INTERVENTION_CATEGORIES in app.py
const INTERVENTION_CATEGORIES = ['Attendance', 'Internal Tests', 'Assignments', 'Study', 'General']
const INTERVENTION_TITLE_MAX = 120
const INTERVENTION_DESCRIPTION_MAX = 1000

// Stored recommendations name a model feature; each maps to a task category and a short title
const FEATURE_INTERVENTION = {
  'Attendance (%)': { category: 'Attendance', title: 'Improve Attendance' },
  'Internal Test 1 (out of 40)': { category: 'Internal Tests', title: 'Strengthen Internal Test 1 Topics' },
  'Internal Test 2 (out of 40)': { category: 'Internal Tests', title: 'Strengthen Internal Test 2 Topics' },
  'Assignment Score (out of 10)': { category: 'Assignments', title: 'Improve Assignment Work' },
  'Daily Study Hours': { category: 'Study', title: 'Increase Daily Study Hours' },
}

// A due date is a calendar day (YYYY-MM-DD), shown without any time-zone shift
const formatDay = (day) => {
  if (!day) return null
  const [y, m, d] = day.split('-').map(Number)
  return new Date(y, m - 1, d).toLocaleDateString([], { dateStyle: 'medium' })
}
const formatDate = (iso) => (iso ? new Date(iso).toLocaleDateString([], { dateStyle: 'medium' }) : null)

function InterventionStatusBadge({ status }) {
  const completed = status === 'completed'
  const Icon = completed ? BadgeCheck : ListChecks
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${completed ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-sky-200 bg-sky-50 text-sky-700'}`}>
      <Icon size={14} aria-hidden="true" />
      {completed ? 'Completed' : 'Assigned'}
    </span>
  )
}

// Today as a calendar day (YYYY-MM-DD) in the browser's time zone, to compare with due dates
const todayDay = () => {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
}

// An assigned intervention whose due date has passed
const isOverdue = (intervention) => intervention.status === 'assigned' && Boolean(intervention.due_date) && intervention.due_date < todayDay()

function interventionCounts(interventions) {
  const completed = interventions.filter((item) => item.status === 'completed').length
  return { total: interventions.length, completed, open: interventions.length - completed, overdue: interventions.filter(isOverdue).length }
}

// A new intervention prefilled from a recommendation stored with prediction predictionId
// (nothing is regenerated: the text and the prediction id come from the stored data)
function draftFromRecommendation(rec, predictionId) {
  const preset = FEATURE_INTERVENTION[rec.title] || { category: 'General', title: `Improve ${rec.title}` }
  return { key: Date.now(), title: preset.title, description: rec.action || rec.description, category: preset.category, predictionId, feature: rec.title }
}

// Opens the intervention form (rendered by FacultyInterventions) and scrolls to it
function CreateInterventionButton({ onClick }) {
  return (
    <button
      type="button"
      onClick={() => {
        onClick()
        setTimeout(() => document.getElementById('intervention-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
      }}
      className={BTN_SECONDARY}
    >
      Create Intervention
    </button>
  )
}

function EmptyState({ message, note, action }) {
  return (
    <div className="rounded-[20px] border border-dashed border-slate-200 bg-slate-50 p-6 text-center">
      <p className="text-sm font-semibold leading-7 text-slate-700">{message}</p>
      {note && <p className="mt-1 text-sm leading-6 text-slate-500">{note}</p>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  )
}

// One stored intervention, with the prediction (and recommendation) it came from, if any.
// showStudent adds the student's name, for lists that span several students.
function InterventionItem({ intervention, action, showStudent = false }) {
  const meta = [
    intervention.category,
    `Assigned ${formatDate(intervention.created_at)}`,
    intervention.due_date && `Due ${formatDay(intervention.due_date)}`,
    intervention.completed_at && `Completed ${formatDateTime(intervention.completed_at)}`,
  ].filter(Boolean)
  return (
    <li className="rounded-[20px] border border-slate-200 bg-white p-4" data-intervention={intervention.id}>
      {showStudent && <p className="mb-1 text-xs font-semibold uppercase tracking-[0.15em] text-blue-600">{intervention.student_name} · {intervention.student_roll_no}</p>}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">{intervention.title}</h3>
        <span className="flex flex-wrap gap-2">
          {isOverdue(intervention) && <span className="inline-flex items-center rounded-full border border-rose-200 bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-700">Overdue</span>}
          <InterventionStatusBadge status={intervention.status} />
        </span>
      </div>
      <p className="mt-2 whitespace-pre-line text-sm leading-6 text-slate-700">{intervention.description}</p>
      <p className="mt-2 text-xs font-medium text-slate-500">{meta.join(' · ')}</p>
      {intervention.prediction_id != null && (
        <p className="mt-1 text-xs text-slate-500" data-testid="intervention-source">
          Based on Prediction #{intervention.prediction_id}{intervention.prediction_created_at ? ` · ${formatDate(intervention.prediction_created_at)}` : ''}
          {intervention.recommendation_feature ? ` · ${intervention.recommendation_feature} recommendation` : ''}
        </p>
      )}
      {action && <div className="mt-3">{action}</div>}
    </li>
  )
}

// Faculty: a new intervention for one student, blank or prefilled from a stored recommendation
function InterventionForm({ studentId, draft, onCreated, onCancel }) {
  const [title, setTitle] = useState(draft.title)
  const [description, setDescription] = useState(draft.description)
  const [category, setCategory] = useState(draft.category)
  const [dueDate, setDueDate] = useState('')
  const [error, setError] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (isSaving) return
    if (!title.trim()) return setError('Enter a title.')
    if (title.trim().length > INTERVENTION_TITLE_MAX) return setError(`Title must be at most ${INTERVENTION_TITLE_MAX} characters.`)
    if (!description.trim()) return setError('Enter a description or instruction.')
    if (description.trim().length > INTERVENTION_DESCRIPTION_MAX) return setError(`Description must be at most ${INTERVENTION_DESCRIPTION_MAX} characters.`)
    setError('')
    setIsSaving(true)
    try {
      const body = { title: title.trim(), description: description.trim(), category, due_date: dueDate || null }
      // The stored prediction (and its recommendation) this task came from; only references are sent
      if (draft.predictionId != null) {
        body.prediction_id = draft.predictionId
        body.recommendation_feature = draft.feature
      }
      const data = await apiRequest(`/students/${studentId}/interventions`, { method: 'POST', body })
      onCreated(data.intervention)
    } catch (err) {
      setError(err.message)
      setIsSaving(false)
    }
  }

  return (
    <form id="intervention-form" onSubmit={handleSubmit} noValidate className="scroll-mt-6 rounded-[20px] border border-blue-100 bg-blue-50/40 p-4" data-testid="intervention-form">
      <p className="text-sm font-semibold text-slate-900">New intervention</p>
      {draft.predictionId != null && (
        <p className="mt-1 text-xs text-slate-600">Started from the {draft.feature} recommendation stored with Prediction #{draft.predictionId}. You can edit the text before saving.</p>
      )}
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <label htmlFor="intervention-title" className="text-sm font-semibold text-slate-700">Title</label>
          <input id="intervention-title" type="text" maxLength={INTERVENTION_TITLE_MAX} value={title} onChange={(event) => setTitle(event.target.value)} disabled={isSaving} className={FIELD_INPUT} />
        </div>
        <div className="sm:col-span-2">
          <label htmlFor="intervention-description" className="text-sm font-semibold text-slate-700">Description / instruction</label>
          <textarea id="intervention-description" rows="3" maxLength={INTERVENTION_DESCRIPTION_MAX} value={description} onChange={(event) => setDescription(event.target.value)} disabled={isSaving} className={FIELD_INPUT} />
        </div>
        <div>
          <label htmlFor="intervention-category" className="text-sm font-semibold text-slate-700">Category</label>
          <select id="intervention-category" value={category} onChange={(event) => setCategory(event.target.value)} disabled={isSaving} className={FIELD_INPUT}>
            {INTERVENTION_CATEGORIES.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="intervention-due" className="text-sm font-semibold text-slate-700">Due date <span className="font-normal text-slate-500">(optional)</span></label>
          <input id="intervention-due" type="date" value={dueDate} onChange={(event) => setDueDate(event.target.value)} disabled={isSaving} className={FIELD_INPUT} />
        </div>
      </div>
      <FormError message={error} />
      <div className="mt-4 flex flex-wrap gap-3">
        <button type="submit" disabled={isSaving} aria-busy={isSaving} className={BTN_PRIMARY}>
          {isSaving ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Saving…</> : 'Assign intervention'}
        </button>
        <button type="button" onClick={onCancel} disabled={isSaving} className={BTN_SECONDARY}>Cancel</button>
      </div>
    </form>
  )
}

// Faculty: the selected student's stored interventions and the create form.
// draft is the form being written (null when closed); it is held by the parent so a
// recommendation card can open the form prefilled. onCreated (optional) is told about each new one.
function FacultyInterventions({ studentId, draft, onDraftChange, onCreated }) {
  const resource = useApiResource(`/students/${studentId}/interventions`)
  const [notice, setNotice] = useState('')
  const interventions = resource.data?.interventions
  const completed = interventions ? interventions.filter((item) => item.status === 'completed').length : 0

  return (
    <section className={CARD} data-testid="faculty-interventions">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <SectionHeading
          title="Interventions"
          description={interventions?.length ? `${interventions.length} assigned to this student · ${completed} completed · ${interventions.length - completed} open` : 'Improvement tasks assigned to this student. The student sees them in My Improvement Tasks.'}
        />
        {!draft && (
          <button type="button" onClick={() => { setNotice(''); onDraftChange({ key: Date.now(), title: '', description: '', category: 'General', predictionId: null, feature: null }) }} className={BTN_SECONDARY}>New intervention</button>
        )}
      </div>
      {draft && (
        <div className="mb-4">
          <InterventionForm
            key={draft.key}
            studentId={studentId}
            draft={draft}
            onCancel={() => onDraftChange(null)}
            onCreated={(created) => {
              resource.replace({ ...resource.data, interventions: [created, ...(interventions ?? [])] })
              onDraftChange(null)
              setNotice(`Intervention "${created.title}" assigned.`)
              onCreated?.(created)
            }}
          />
        </div>
      )}
      {notice && <p role="status" className="mb-4 rounded-[18px] border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm font-semibold text-emerald-800">{notice}</p>}
      {!interventions ? (
        <ResourceState resource={resource} what="interventions" />
      ) : interventions.length === 0 ? (
        <p className="rounded-[18px] border border-dashed border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">No interventions have been assigned to this student yet.</p>
      ) : (
        <ul className="space-y-3">
          {interventions.map((intervention) => <InterventionItem key={intervention.id} intervention={intervention} />)}
        </ul>
      )}
    </section>
  )
}

// Student practice pages run the model on sample inputs, not the student's stored record
function PracticeModeBanner() {
  return (
    <div role="note" className="mb-6 flex items-start gap-3 rounded-[20px] border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900">
      <FlaskConical size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <div>
        <p className="font-semibold">Practice Mode — Sample Scenario</p>
        <p className="mt-0.5">This page runs the live model on a sample scenario. It does not use or change your stored academic record, and its results are not saved.</p>
      </div>
    </div>
  )
}

// The five model inputs as compact value tiles, labelled as on the input forms.
// A value not entered yet (null in a stored academic record) is shown as such.
function InputsList({ inputs, wide = false }) {
  return (
    <dl className={`grid gap-2 ${wide ? 'grid-cols-2 sm:grid-cols-3 xl:grid-cols-5' : 'grid-cols-2 sm:grid-cols-3'}`}>
      {MODEL_INPUT_FIELDS.map(({ key, label }) => (
        <div key={key} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
          <dt className="text-xs leading-5 text-slate-500">{label}</dt>
          <dd className="mt-0.5 text-lg font-semibold text-slate-900">{inputs[key] ?? <span className="text-sm font-medium text-slate-500">Not entered</span>}</dd>
        </div>
      ))}
    </dl>
  )
}

// Page title block with breadcrumbs and an in-app back link (never browser history)
function PageHeader({ breadcrumbs = [], back, eyebrow, title, description, actions }) {
  const backClass = 'inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1.5 font-semibold text-slate-700 transition hover:border-blue-200 hover:text-blue-700'
  const backContent = <><ArrowLeft size={15} aria-hidden="true" />{back?.label}</>
  return (
    <header className="ui-page-header mb-6">
      {(back || breadcrumbs.length > 0) && (
        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          {back && (back.to
            ? <NavLink to={back.to} className={backClass}>{backContent}</NavLink>
            : <button type="button" onClick={back.onClick} className={backClass}>{backContent}</button>)}
          {breadcrumbs.length > 0 && (
            <nav aria-label="Breadcrumb">
              <ol className="flex flex-wrap items-center gap-1 text-slate-500">
                {breadcrumbs.map((crumb, index) => (
                  <li key={crumb.label} className="flex items-center gap-1">
                    {index > 0 && <ChevronRight size={14} aria-hidden="true" />}
                    {crumb.to && <NavLink to={crumb.to} className="hover:text-blue-700">{crumb.label}</NavLink>}
                    {!crumb.to && crumb.onClick && <button type="button" onClick={crumb.onClick} className="hover:text-blue-700">{crumb.label}</button>}
                    {!crumb.to && !crumb.onClick && <span aria-current="page" className="font-semibold text-slate-700">{crumb.label}</span>}
                  </li>
                ))}
              </ol>
            </nav>
          )}
        </div>
      )}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="min-w-0">
          {eyebrow && <p className="ui-eyebrow">{eyebrow}</p>}
          <h1 className="ui-page-title mt-1 text-2xl sm:text-3xl">{title}</h1>
          {description && <p className="ui-page-description mt-2 max-w-3xl">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap gap-3">{actions}</div>}
      </div>
    </header>
  )
}

// Numbered or plain heading for a page section
function SectionHeading({ step, title, description }) {
  return (
    <div className="mb-4 flex items-start gap-3">
      {step && <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white" aria-hidden="true">{step}</span>}
      <div>
        <h2 className="ui-section-title">{title}</h2>
        {description && <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p>}
      </div>
    </div>
  )
}

// Friendly notice for a failed call to the Flask prediction service
function ServiceErrorNotice({ title = 'Prediction service unavailable', message = 'The Flask prediction service isn\'t currently running. Start the backend and try again.', onRetry, retrying = false }) {
  return (
    <div role="alert" className="flex flex-col gap-3 rounded-[20px] border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-start gap-3">
        <CircleAlert size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
        <div>
          <p className="font-semibold">{title}</p>
          <p className="mt-1 leading-6">{message}</p>
        </div>
      </div>
      {onRetry && (
        <button type="button" onClick={onRetry} disabled={retrying} className="inline-flex shrink-0 items-center gap-2 rounded-full border border-rose-300 bg-white px-4 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 disabled:opacity-60">
          <RotateCcw size={14} aria-hidden="true" /> {retrying ? 'Retrying…' : 'Retry'}
        </button>
      )}
    </div>
  )
}

const formatShapMarks = (value) => `${Number(value) >= 0 ? '+' : ''}${Number(value).toFixed(2)} marks`

// Maps /recommend items to the fields used by the recommendation cards and exports (no fabricated fallback)
function mapRecommendations(apiRecommendations) {
  if (!Array.isArray(apiRecommendations)) return []
  return apiRecommendations.map((r, idx) => ({
    id: idx,
    title: r.feature || `Recommendation ${idx + 1}`,
    description: r.recommendation || r.action || '',
    priority: r.priority || '',
    action: r.action || '',
    currentValue: r.current_value,
    performanceLevel: r.performance_level,
    shapContribution: r.shap_contribution,
    needsAttention: r.priority === 'HIGH' || r.priority === 'MEDIUM',
  }))
}

// The most negative SHAP contribution among HIGH/MEDIUM items (display only; order is unchanged)
function findStrongestNegative(recommendations) {
  return recommendations
    .filter((rec) => rec.needsAttention && Number.isFinite(Number(rec.shapContribution)))
    .reduce((min, rec) => (min === null || Number(rec.shapContribution) < Number(min.shapContribution) ? rec : min), null)
}

const PRIORITY_STYLES = {
  HIGH: { card: 'border-rose-200 bg-rose-50/40', badge: 'border-rose-200 bg-rose-50 text-rose-700', label: 'HIGH priority', box: 'border-rose-100', boxLabel: 'text-rose-700', ring: 'ring-rose-300' },
  MEDIUM: { card: 'border-amber-200 bg-amber-50/40', badge: 'border-amber-200 bg-amber-50 text-amber-700', label: 'MEDIUM priority', box: 'border-amber-100', boxLabel: 'text-amber-700', ring: 'ring-amber-300' },
  POSITIVE: { card: 'border-emerald-200 bg-emerald-50/40', badge: 'border-emerald-200 bg-emerald-50 text-emerald-700', label: 'POSITIVE contributor', box: 'border-emerald-100', boxLabel: 'text-emerald-700' },
}
const DEFAULT_PRIORITY_STYLE = { card: 'border-slate-200 bg-slate-50', badge: 'border-slate-200 bg-slate-50 text-slate-700', label: 'Recommendation', box: 'border-slate-200', boxLabel: 'text-blue-600' }

// One /recommend item: feature, its current value and level, why it matters (the backend message) and the action
function RecommendationCard({ rec, isStrongest, footer }) {
  const style = PRIORITY_STYLES[rec.priority] || DEFAULT_PRIORITY_STYLE
  const details = [
    ['Current value', rec.currentValue != null ? rec.currentValue : null],
    ['Performance level', rec.performanceLevel || null],
    ['SHAP contribution', Number.isFinite(Number(rec.shapContribution)) && rec.shapContribution != null ? formatShapMarks(rec.shapContribution) : null],
  ].filter(([, value]) => value !== null)
  return (
    <article className={`flex h-full flex-col rounded-[20px] border p-4 sm:p-5 ${style.card} ${isStrongest ? `ring-2 ${style.ring}` : ''}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h5 className="text-base font-semibold leading-6 text-slate-900">{rec.title}</h5>
        <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${style.badge}`}>{style.label}</span>
      </div>
      {isStrongest && (
        <p className="mt-2 inline-flex w-fit items-center gap-1.5 rounded-lg border border-rose-200 bg-white px-2.5 py-1 text-xs font-semibold text-rose-800">
          <ArrowDown size={13} aria-hidden="true" /> Strongest negative contributor
        </p>
      )}
      {details.length > 0 && (
        <dl className="mt-4 grid gap-2 text-xs sm:grid-cols-3">
          {details.map(([label, value]) => (
            <div key={label} className="min-w-0 rounded-xl border border-slate-200 bg-white px-3 py-2">
              <dt className="text-slate-500">{label}</dt>
              <dd className="mt-0.5 break-words font-semibold text-slate-900">{value}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className="mt-4">
        <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Model explanation</p>
        <p className="mt-1 text-sm leading-6 text-slate-700">{rec.description}</p>
      </div>
      {rec.action && rec.action !== rec.description && (
        <div className={`mt-4 rounded-[16px] border bg-white p-3.5 ${style.box}`}>
          <p className={`text-xs font-bold uppercase tracking-wide ${style.boxLabel}`}>{rec.needsAttention ? 'Suggested next step' : 'Maintenance guidance'}</p>
          <p className="mt-1 text-sm leading-6 text-slate-700">{rec.action}</p>
        </div>
      )}
      {footer && <div className="mt-auto pt-3">{footer}</div>}
    </article>
  )
}

// Recommendations grouped by priority; backend order is kept inside each group
const RECOMMENDATION_GROUPS = [
  { priority: 'HIGH', title: 'High priority' },
  { priority: 'MEDIUM', title: 'Medium priority' },
  { priority: 'POSITIVE', title: 'Maintenance — positive contributors' },
]

function RecommendationGroups({ recommendations, strongest, renderFooter }) {
  return (
    <div className="space-y-6">
      {RECOMMENDATION_GROUPS.map(({ priority, title }) => {
        const items = recommendations.filter((rec) => rec.priority === priority)
        if (items.length === 0) return null
        return (
          <section key={priority} aria-labelledby={`recommendation-group-${priority.toLowerCase()}`}>
            <h4 id={`recommendation-group-${priority.toLowerCase()}`} className="mb-3 flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-800">
              <span className={`rounded-full border px-2.5 py-1 text-xs ${PRIORITY_STYLES[priority].badge}`}>{priority}</span>
              {title}
              <span className="text-xs font-medium text-slate-500">{items.length} {items.length === 1 ? 'recommendation' : 'recommendations'}</span>
            </h4>
            <div className="grid gap-4 md:grid-cols-2">
              {items.map((rec) => <RecommendationCard key={rec.id} rec={rec} isStrongest={strongest != null && rec.id === strongest.id} footer={renderFooter?.(rec)} />)}
            </div>
          </section>
        )
      })}
    </div>
  )
}

// SHAP values and feature order are displayed as returned; only bar lengths use a shared visual scale.
function ShapContributionBars({ contributions, baseValue, predictedMarks }) {
  const entries = Object.entries(contributions || {})
  const maxContribution = Math.max(...entries.map(([, value]) => Math.abs(Number(value))), 0)
  const hasBaseValue = baseValue != null && Number.isFinite(Number(baseValue))
  const hasPredictedMarks = predictedMarks != null && Number.isFinite(Number(predictedMarks))
  return (
    <div>
      <p className="mb-4 rounded-xl border border-blue-100 bg-blue-50/60 px-4 py-3 text-sm leading-6 text-slate-700">
        Positive SHAP values move the model output upward from its baseline; negative values move it downward. Bar lengths are scaled relative to the largest absolute contribution in this explanation; the displayed values are the returned SHAP values.
      </p>
      {(hasBaseValue || hasPredictedMarks) && (
        <dl className="mb-5 grid gap-3 sm:grid-cols-2">
          {hasBaseValue && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
              <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">SHAP baseline</dt>
              <dd className="mt-1 text-lg font-semibold tabular-nums text-slate-900">{Number(baseValue).toFixed(2)} marks</dd>
              <dd className="mt-1 text-xs leading-5 text-slate-600">Starting model output before these feature contributions are applied.</dd>
            </div>
          )}
          {hasPredictedMarks && (
            <div className="rounded-xl border border-blue-200 bg-blue-50/60 px-4 py-3">
              <dt className="text-xs font-semibold uppercase tracking-wide text-blue-800">Predicted model output</dt>
              <dd className="mt-1 text-lg font-semibold tabular-nums text-slate-900">{predictedMarks} marks</dd>
              <dd className="mt-1 text-xs leading-5 text-slate-600">Baseline plus the feature contributions shown below; values are displayed at the precision returned by the API.</dd>
            </div>
          )}
        </dl>
      )}
      {entries.length === 0 ? (
        <p className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">SHAP contributions are not available.</p>
      ) : (
        <ul className="space-y-4" aria-label="SHAP feature contributions">
          {entries.map(([name, val]) => {
            const numeric = Number(val)
            const positive = numeric > 0
            const negative = numeric < 0
            const width = `${maxContribution > 0 ? (Math.abs(numeric) / maxContribution) * 50 : 0}%`
            const Icon = positive ? ArrowUp : negative ? ArrowDown : Minus
            const direction = positive ? 'Raises prediction' : negative ? 'Lowers prediction' : 'No change'
            const tone = positive ? 'text-emerald-800' : negative ? 'text-amber-900' : 'text-slate-600'
            const barTone = positive ? 'bg-emerald-600' : negative ? 'bg-amber-600' : 'bg-slate-400'
            return (
              <li key={name} className="rounded-xl border border-slate-200 bg-white px-3 py-3 sm:px-4">
                <div className="grid gap-x-4 gap-y-1 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
                  <span className="text-sm font-semibold leading-5 text-slate-800">{name}</span>
                  <span className="inline-flex items-center gap-1.5 text-sm font-bold tabular-nums text-slate-950 sm:justify-self-end">
                    {formatShapMarks(numeric)}
                    <span className={`inline-flex items-center gap-1 text-xs font-semibold ${tone}`}>
                      <Icon size={13} aria-hidden="true" />{direction}
                    </span>
                  </span>
                </div>
                <div className="relative mt-3 h-3 rounded-full bg-slate-100" aria-hidden="true">
                  <div className="absolute inset-y-0 left-1/2 z-[1] w-0.5 -translate-x-1/2 bg-slate-500" />
                  {numeric !== 0 && (
                    <div
                      className={`absolute inset-y-0 ${positive ? 'left-1/2 rounded-r-full' : 'right-1/2 rounded-l-full'} ${barTone}`}
                      style={{ width }}
                    />
                  )}
                </div>
                <div className="mt-1 flex justify-between text-[11px] font-medium text-slate-500" aria-hidden="true">
                  <span>↓ Lowers output</span><span>0</span><span>Raises output ↑</span>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// 5-fold cross-validation results for the production model.
// MAE and RMSE are in final-exam marks (out of 100); R² is unitless.
const adminModelMetrics = [
  { label: 'MAE', value: '3.62 ± 0.07', detail: 'Mean absolute error (marks)' },
  { label: 'RMSE', value: '4.59 ± 0.09', detail: 'Root mean squared error (marks)' },
  { label: 'R²', value: '0.8357 ± 0.0084', detail: 'Coefficient of determination' },
]

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
      <SessionDemoStudents>
      <div className="app-shell">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard-selection" element={<DashboardSelectionPage />} />
          <Route path="/login" element={<LoginPage />} />
          {/* Every Student Portal page shares one sidebar, so no page is a navigation dead end */}
          <Route element={<RequireRole role="student"><StudentLayout /></RequireRole>}>
            <Route path="/student" element={<StudentHome />} />
            <Route path="/prediction" element={<PredictionRoute />} />
            <Route path="/what-if" element={<WhatIfPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/profile" element={<ProfilePage />} />
          </Route>
          <Route path="/faculty" element={<RequireRole role="faculty"><FacultyDashboard /></RequireRole>} />
          <Route path="/admin" element={<AdminDashboard />} />
        </Routes>
      </div>
      </SessionDemoStudents>
      </AuthProvider>
    </BrowserRouter>
  )
}

// The session-only demo state starts fresh for each signed-in account, so one
// user's demo results and assignments are never shown to the next user of the tab
function SessionDemoStudents({ children }) {
  const { user } = useAuth()
  return <DemoStudentsProvider key={user ? user.id : 'signed-out'}>{children}</DemoStudentsProvider>
}

// Brand block shared by both portal sidebars; links back to the landing page
function PortalBrand({ icon: Icon, title, subtitle }) {
  return (
    <NavLink to="/" aria-label="PredictWise home" className="flex items-center gap-3 rounded-2xl">
      <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 p-2.5 text-white"><Icon size={20} aria-hidden="true" /></div>
      <div>
        <p className="font-semibold text-slate-900">{title}</p>
        <p className="text-sm text-slate-500">{subtitle}</p>
      </div>
    </NavLink>
  )
}

// The signed-in account, sign out and the landing page, shown at the bottom of each portal sidebar
function PortalAccount() {
  const { user, logout } = useAuth()
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-slate-200 pt-4 lg:flex-col lg:items-stretch">
      <p className="flex-1 px-4 text-sm text-slate-600">Signed in as <span className="font-semibold text-slate-900">{user.username}</span></p>
      <button type="button" onClick={logout} className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:border-blue-200 hover:text-blue-700">
        <ArrowLeft size={18} aria-hidden="true" />Sign out
      </button>
      <NavLink to="/" className="flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium text-slate-600 transition hover:bg-slate-100 hover:text-slate-900">
        <House size={18} aria-hidden="true" />Home
      </NavLink>
    </div>
  )
}

const STUDENT_NAV = [
  { name: 'Dashboard', icon: LayoutDashboard, path: '/student' },
  { name: 'Prediction', icon: BrainCircuit, path: '/prediction' },
  { name: 'What-If Analysis', icon: ArrowRightLeft, path: '/what-if' },
  { name: 'My Improvement Tasks', icon: ListChecks, path: '/student', view: 'interventions' },
  { name: 'Reports', icon: FileText, path: '/reports' },
  { name: 'Profile', icon: UserRound, path: '/profile' },
]

// Practice pages run on a sample scenario rather than the student's stored record
const isPracticePage = (location) => ['/prediction', '/what-if'].includes(location.pathname)

function StudentLayout() {
  const location = useLocation()
  const currentView = new URLSearchParams(location.search).get('view')
  // The student shown is always the signed-in user's own record (never chosen in the UI)
  const { user } = useAuth()
  const studentResource = useApiResource(`/students/${user.student_id}`)
  const detail = studentResource.data
  return (
    <div className="app-shell">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="w-full border-b border-slate-200 bg-white/90 p-4 shadow-sm backdrop-blur lg:sticky lg:top-0 lg:h-screen lg:w-72 lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r lg:p-6">
          <PortalBrand icon={GraduationCap} title="Student Portal" subtitle="PredictWise" />
          <div className="mt-5 rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3 text-sm" data-testid="student-identity">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-blue-800">Signed-in student</p>
            {detail ? (
              <>
                <p className="mt-1 font-semibold text-slate-900">{detail.student.name}</p>
                <p className="text-slate-600">{detail.student.roll_no}</p>
                <p className="mt-1 text-slate-600">{hasCurrentPrediction(detail) ? `${detail.latest_prediction.predicted_marks} marks · ${detail.latest_prediction.risk_level}` : PREDICTION_STATUS[detail.prediction_status.status]?.label}</p>
              </>
            ) : (
              <p className="mt-1 text-slate-600">{studentResource.error ? 'Record unavailable' : 'Loading…'}</p>
            )}
          </div>
          <nav aria-label="Student Portal" className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:flex lg:flex-col">
            {STUDENT_NAV.map(({ name, icon: Icon, path, view }) => {
              // /student and /student?view=interventions share a path, so the view parameter decides which is active
              const isActive = location.pathname === path && (view ?? null) === currentView
              return (
                <Link key={name} to={view ? `${path}?view=${view}` : path} aria-current={isActive ? 'page' : undefined} className={`flex min-h-11 min-w-0 items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium leading-5 transition sm:gap-3 sm:px-4 ${isActive ? 'bg-blue-700 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}`}>
                  <Icon size={18} aria-hidden="true" />
                  <span className="min-w-0">{name}</span>
                </Link>
              )
            })}
          </nav>
          <PortalAccount />
        </aside>

        <main className="min-w-0 flex-1 p-4 sm:p-6 lg:p-8">
          <div className="mx-auto max-w-6xl">
            {isPracticePage(location) && <PracticeModeBanner />}
            <Outlet context={studentResource} />
          </div>
        </main>
      </div>
    </div>
  )
}

// Re-creates the Prediction page when another Demo Student is selected, so the form
// starts from that student's latest submitted inputs (or their predefined inputs)
function PredictionRoute() {
  const { selectedId } = useDemoStudents()
  return <PredictionPage key={selectedId} />
}

function LandingPage() {
  const stats = [
    { value: '5', label: 'Algorithms Tested' },
    { value: '5', label: 'Model Input Features' },
    { value: '2,000', label: 'Dataset Records' },
    { value: '0.8357', label: '5-Fold CV R²' },
  ]

  const steps = ['Enter Student Data', 'Preprocessing', 'AI Prediction', 'SHAP Explainability', 'Recommendations', 'Faculty Interventions']

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.16),_transparent_30%),linear-gradient(135deg,#f9fbff_0%,#f2f4f8_100%)]">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 p-2.5 text-white shadow-lg">
            <BrainCircuit size={20} />
          </div>
          <div>
            <p className="text-lg font-semibold text-slate-900">PredictWise</p>
            <p className="text-sm text-slate-500">AI-Based Student Analytics</p>
          </div>
        </div>
        <div className="flex items-center gap-4 md:gap-6">
          <a href="#portals" className="hidden text-sm font-medium text-slate-600 transition hover:text-blue-600 md:inline">Portals</a>
          <a href="#how-it-works" className="hidden text-sm font-medium text-slate-600 transition hover:text-blue-600 md:inline">How it works</a>
          <a href="#features" className="hidden text-sm font-medium text-slate-600 transition hover:text-blue-600 md:inline">Features</a>
          <NavLink to="/login" className="text-sm font-semibold text-slate-700 transition hover:text-blue-600">Sign in</NavLink>
          <NavLink to="/student" className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700">Student Portal</NavLink>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl px-6 pb-24 lg:px-8">
        <section className="grid items-center gap-10 rounded-[36px] border border-white/70 bg-white/80 p-8 shadow-[0_30px_90px_-35px_rgba(37,99,235,0.35)] backdrop-blur-xl lg:grid-cols-[1.05fr_0.95fr] lg:p-12">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.55 }}>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-4 py-2 text-sm font-semibold text-blue-700">
              <Sparkles size={16} /> AI-based student performance prediction system
            </div>
            <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
              Predict student performance with SHAP-based explainability.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">
              An explainable AI system for student performance analysis that helps identify risk early, explain every prediction, and support interventions with clarity.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <NavLink to="/student" className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 px-6 py-3 font-semibold text-white shadow-lg shadow-blue-200 transition hover:scale-[1.02]"><GraduationCap size={18} aria-hidden="true" />Open Student Portal</NavLink>
              <NavLink to="/faculty" className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 font-semibold text-slate-700 transition hover:border-blue-200 hover:text-blue-700"><Users size={18} aria-hidden="true" />Open Teacher Portal</NavLink>
              <NavLink to="/dashboard-selection" className="text-sm font-semibold text-slate-600 underline-offset-4 hover:text-blue-700 hover:underline">All dashboards</NavLink>
            </div>
            <div className="mt-10 flex flex-wrap gap-3">
              {['Early Prediction', 'Explainable AI', 'What-If Analysis'].map((item) => (
                <span key={item} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium text-slate-700">{item}</span>
              ))}
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, x: 22 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}>
            <div className="rounded-[30px] border border-slate-200 bg-gradient-to-br from-slate-950 via-blue-950 to-cyan-900 p-6 text-white shadow-2xl">
              <div className="rounded-[24px] border border-white/10 bg-white/10 p-5 backdrop-blur">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-blue-200">Project insight</p>
                    <p className="mt-2 text-3xl font-semibold">Model at a glance</p>
                  </div>

                  {/* Predicted marks are shown on the Prediction page; no global state here. */}
                  <div className="rounded-2xl bg-white/15 p-3"><Wand2 size={20} /></div>
                </div>
                  <div className="mt-6 grid gap-3 sm:grid-cols-2">
                    {[
                      ['2,000', 'Dataset records'],
                      ['5', 'Model input features'],
                      ['0.8357', '5-Fold CV R²'],
                    ].map(([value, label]) => (
                      <div key={label} className="rounded-2xl border border-white/10 bg-slate-900/40 p-3 text-sm font-medium text-slate-100">
                        <span className="block text-lg font-semibold text-white">{value}</span>
                        {label}
                      </div>
                    ))}
                  </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* The two sides of the demo, with the flow each one follows */}
        <section id="portals" className="mt-10 grid scroll-mt-6 gap-6 lg:grid-cols-2">
          {[
            { title: 'Student Portal', icon: GraduationCap, to: '/student', cta: 'Open Student Portal', text: 'Students sign in to see their own academic record, prediction status and latest prediction with its SHAP explanation.', steps: ['Sign in', 'View record', 'Explain (SHAP)', 'Explore What-If'] },
            { title: 'Teacher Portal', icon: Users, to: '/faculty', cta: 'Open Teacher Portal', text: 'Faculty sign in to see every student\'s record and prediction status, with the students that need attention first.', steps: ['Sign in', 'Review students', 'Analyze', 'View recommendations'] },
          ].map(({ title, icon: Icon, to, cta, text, steps: flow }) => (
            <div key={title} className="flex flex-col rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 p-2.5 text-white"><Icon size={20} aria-hidden="true" /></div>
                <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
              </div>
              <p className="mt-3 text-sm leading-7 text-slate-600">{text}</p>
              <ol className="mt-4 flex flex-wrap items-center gap-2 text-sm">
                {flow.map((step, index) => (
                  <li key={step} className="flex items-center gap-2">
                    {index > 0 && <ChevronRight size={14} className="text-slate-400" aria-hidden="true" />}
                    <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 font-medium text-slate-700">{step}</span>
                  </li>
                ))}
              </ol>
              <NavLink to={to} className={`${BTN_PRIMARY} mt-6 self-start`}>{cta} <ArrowRight size={16} aria-hidden="true" /></NavLink>
            </div>
          ))}
        </section>

        <section className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {stats.map((stat) => (
            <motion.div whileHover={{ y: -3, scale: 1.01 }} key={stat.label} className="rounded-[24px] border border-slate-200 bg-white/85 p-5 shadow-sm backdrop-blur">
              <p className="text-3xl font-semibold text-slate-900">{stat.value}</p>
              <p className="mt-2 text-sm text-slate-600">{stat.label}</p>
            </motion.div>
          ))}
        </section>

        <section id="how-it-works" className="mt-20 rounded-[32px] border border-slate-200 bg-white/80 p-8 shadow-sm">
          <div className="mb-8 max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-[0.25em] text-blue-600">How it works</p>
            <h2 className="mt-3 text-3xl font-semibold text-slate-900">From raw academic data to intervention-ready intelligence</h2>
            <p className="mt-3 text-sm leading-7 text-slate-600">Student data is entered as five model inputs: Attendance (%), Internal Test 1, Internal Test 2, Assignment Score and Daily Study Hours.</p>
          </div>
          <div className="grid gap-4 lg:grid-cols-6">
            {steps.map((step, index) => (
              <div key={step} className="rounded-[24px] border border-slate-200 bg-slate-50 p-4">
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 text-sm font-semibold text-white">0{index + 1}</div>
                <p className="text-sm font-semibold text-slate-900">{step}</p>
              </div>
            ))}
          </div>
        </section>

        <section id="features" className="mt-20 grid gap-6 lg:grid-cols-3">
          {[
            ['Early Prediction', 'Identify at-risk students before the term worsens.'],
            ['Explainable AI', 'SHAP-based evidence that makes every prediction interpretable.'],
            ['Personalized Recommendations', 'Feature-level suggestions based on the student\'s prediction and the factors contributing to it.'],
            ['What-If Analysis', 'Explore model predictions through practical scenario testing.'],
            ['Model Evaluation', 'Model performance evaluated with 5-fold cross-validation using MAE, RMSE and R².'],
            ['Reports', 'Model evaluation, dataset facts and early-warning methodology.'],
          ].map(([title, text]) => (
            <motion.div whileHover={{ y: -4 }} key={title} className="rounded-[24px] border border-slate-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-50 text-blue-600">
                <BadgeCheck size={20} />
              </div>
              <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
              <p className="mt-3 text-sm leading-7 text-slate-600">{text}</p>
            </motion.div>
          ))}
        </section>
      </main>

      <footer className="border-t border-slate-200 bg-white/80 px-6 py-8 text-center text-sm text-slate-600">
        University project: AI-based student performance prediction and decision support with explainable AI.
      </footer>
    </div>
  )
}

function DashboardSelectionPage() {
  const cards = [
    {
      title: 'Student Dashboard',
      description: 'Explore predicted final-exam marks and early-warning status with a scenario simulator, with links to SHAP explanations and recommendations.',
      icon: GraduationCap,
      accent: 'from-blue-600 via-blue-500 to-cyan-500',
      href: '/student',
      access: 'Student sign-in',
    },
    {
      title: 'Teacher Portal',
      description: 'Faculty workspace showing students from the student database with their prediction status, latest stored prediction, SHAP analysis and recommendations.',
      icon: BarChart3,
      accent: 'from-violet-600 via-indigo-500 to-cyan-500',
      href: '/faculty',
      access: 'Faculty sign-in',
    },
    {
      title: 'Administrator Dashboard',
      description: 'View the production model’s 5-fold cross-validation performance and whether the prediction API has the model and SHAP explainer loaded.',
      icon: ShieldCheck,
      accent: 'from-emerald-600 via-cyan-500 to-blue-600',
      href: '/admin',
      access: 'Open access',
    },
  ]

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.16),_transparent_30%),linear-gradient(135deg,#f7fbff_0%,#f4f6f8_100%)] px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="mb-10 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-sm font-semibold uppercase tracking-[0.25em] text-blue-600">Dashboard selection</p>
            <h1 className="mt-3 text-4xl font-semibold text-slate-900 sm:text-5xl">Explore the AI analytics experience</h1>
            <p className="mt-4 text-lg leading-8 text-slate-600">The Student and Teacher portals require signing in with a project account. The Administrator dashboard is open.</p>
          </div>
          <NavLink to="/" className={BTN_SECONDARY}><ArrowLeft size={15} aria-hidden="true" />Back to landing</NavLink>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {cards.map(({ title, description, icon: Icon, accent, href, access }) => (
            <motion.div key={title} whileHover={{ y: -6, scale: 1.01 }} className="rounded-[30px] border border-slate-200 bg-white/90 p-6 shadow-[0_24px_60px_-26px_rgba(15,23,42,0.3)] backdrop-blur">
              <div className={`inline-flex rounded-2xl bg-gradient-to-br ${accent} p-3 text-white`}>
                <Icon size={22} />
              </div>
              <h2 className="mt-5 text-2xl font-semibold text-slate-900">{title}</h2>
              <p className="mt-3 text-sm leading-7 text-slate-600">{description}</p>
              <div className="mt-8 flex items-center justify-between">
                <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-600">{access}</span>
                <NavLink to={href} className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700">Open Dashboard <ChevronRight size={16} /></NavLink>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

function LoginPage() {
  const { status, user, expired, login, logout, retryRestore } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState('')
  const sessionExpired = Boolean(location.state?.sessionExpired || expired)

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (isSubmitting) return
    if (!username.trim() || !password) {
      setError('Please enter your username and password.')
      return
    }
    setError('')
    setIsSubmitting(true)
    try {
      const signedIn = await login(username.trim(), password)
      navigate(ROLE_HOME[signedIn.role] || '/', { replace: true })
    } catch (err) {
      // The server's own message: the same for a wrong username or password
      setError(err.message)
      setPassword('')
      setIsSubmitting(false)
    }
  }

  let panel
  if (status === 'checking') {
    panel = (
      <p role="status" className="mt-8 flex items-center gap-2 text-sm text-slate-600">
        <LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Checking your session…
      </p>
    )
  } else if (status === 'unreachable') {
    panel = <div className="mt-8"><ServiceErrorNotice title="Server unavailable" message="Your session could not be checked because the server could not be reached. Start the Flask backend and try again." onRetry={retryRestore} /></div>
  } else if (status === 'signedIn') {
    panel = (
      <div className="mt-8 rounded-[24px] border border-slate-200 bg-slate-50 p-5">
        <p className="text-sm text-slate-600">Signed in as</p>
        <p className="mt-1 text-lg font-semibold text-slate-900">{user.username} <span className="ml-1 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-0.5 align-middle text-xs font-semibold capitalize text-blue-700">{user.role}</span></p>
        <div className="mt-5 flex flex-wrap gap-3">
          <NavLink to={ROLE_HOME[user.role] || '/'} className={BTN_PRIMARY}>Continue <ArrowRight size={16} aria-hidden="true" /></NavLink>
          <button type="button" onClick={logout} className={BTN_SECONDARY}>Sign out</button>
        </div>
      </div>
    )
  } else {
    panel = (
      <form onSubmit={handleSubmit} noValidate className="mt-8 space-y-5">
        {sessionExpired && !error && (
          <p role="status" className="rounded-[18px] border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">Your session has ended. Please sign in again.</p>
        )}
        <div>
          <label htmlFor="login-username" className="text-sm font-semibold text-slate-700">Username</label>
          <input id="login-username" type="text" autoComplete="username" autoCapitalize="none" spellCheck="false" value={username} onChange={(event) => setUsername(event.target.value)} disabled={isSubmitting} className={FIELD_INPUT} />
        </div>
        <div>
          <label htmlFor="login-password" className="text-sm font-semibold text-slate-700">Password</label>
          <div className="relative mt-2">
            <input id="login-password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} disabled={isSubmitting} className="ui-field pr-12" />
            <button
              type="button"
              aria-label={showPassword ? 'Hide password' : 'Show password'}
              aria-pressed={showPassword}
              onClick={() => setShowPassword((visible) => !visible)}
              disabled={isSubmitting}
              className="absolute inset-y-0 right-0 inline-flex min-h-11 items-center justify-center rounded-r-xl px-3 text-slate-600 hover:text-blue-700"
            >
              {showPassword ? <Eye size={18} aria-hidden="true" /> : <EyeOff size={18} aria-hidden="true" />}
            </button>
          </div>
        </div>
        {error && (
          <div role="alert" className="flex items-start gap-2 rounded-[18px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
            <CircleAlert size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
            <p>{error}</p>
          </div>
        )}
        <button type="submit" disabled={isSubmitting} aria-busy={isSubmitting} className={`${BTN_PRIMARY} w-full py-3 text-base`}>
          {isSubmitting
            ? <><LoaderCircle size={18} className="animate-spin" aria-hidden="true" /> Signing in…</>
            : 'Sign in'}
        </button>
      </form>
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.16),_transparent_30%),linear-gradient(135deg,#eef5ff_0%,#f8fafc_100%)] px-4 py-6 sm:py-10">
      <div className="grid w-full max-w-6xl overflow-hidden rounded-[28px] border border-white/70 bg-white/90 shadow-[0_30px_90px_-35px_rgba(15,23,42,0.3)] backdrop-blur-xl lg:grid-cols-[1.02fr_0.98fr]">
        <div className="bg-gradient-to-br from-blue-700 via-blue-600 to-cyan-600 p-6 text-white sm:p-10">
          <div className="inline-flex rounded-full bg-white/15 px-3 py-1 text-sm font-medium backdrop-blur">PredictWise</div>
          <h2 className="mt-5 text-2xl font-semibold sm:mt-6 sm:text-3xl">Sign in to your portal</h2>
          <p className="mt-3 max-w-md text-sm leading-7 text-blue-50">Faculty and students sign in with the accounts provided for this project.</p>
          <div className="mt-6 rounded-[20px] border border-white/20 bg-white/10 p-4 backdrop-blur sm:mt-10 sm:p-5">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-white/20 p-3"><BrainCog size={20} aria-hidden="true" /></div>
              <div>
                <p className="font-semibold">Role-based access</p>
                <p className="text-sm text-blue-100">The server checks every request: faculty can access all student records, and each student can access only their own.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 sm:p-10">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.25em] text-blue-600">Account access</p>
              <h1 className="mt-2 text-2xl font-semibold text-slate-900">Sign in</h1>
            </div>
            <div className="rounded-2xl bg-slate-100 p-2 text-slate-600"><CircleUserRound size={20} aria-hidden="true" /></div>
          </div>

          {panel}

          <div className="mt-8 border-t border-slate-100 pt-6">
            <NavLink to="/" className="text-sm font-semibold text-slate-600 underline-offset-4 hover:text-blue-700 hover:underline">Back to landing</NavLink>
          </div>
        </div>
      </div>
    </div>
  )
}

// The signed-in student's improvement tasks from the database (GET /students/<own id>/interventions).
// The student can mark an assigned task completed; nothing else about a task can be changed here.
function ImprovementTasksPage() {
  const { user } = useAuth()
  const resource = useApiResource(`/students/${user.student_id}/interventions`)
  const tasks = resource.data?.interventions
  const completed = tasks ? tasks.filter((item) => item.status === 'completed').length : 0
  // Per task: true while its completion is being saved, or the error message from the last attempt
  const [busyId, setBusyId] = useState(null)
  const [errors, setErrors] = useState({})

  const markComplete = async (task) => {
    if (busyId != null) return
    setBusyId(task.id)
    setErrors((prev) => ({ ...prev, [task.id]: '' }))
    try {
      const data = await apiRequest(`/interventions/${task.id}/complete`, { method: 'PUT' })
      resource.replace({ ...resource.data, interventions: tasks.map((item) => (item.id === task.id ? data.intervention : item)) })
    } catch (err) {
      // Already completed elsewhere (409): show the stored state; otherwise keep the task and show why
      if (err.status === 409) await resource.refresh()
      else setErrors((prev) => ({ ...prev, [task.id]: err.message }))
    } finally {
      setBusyId(null)
    }
  }

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Student Portal', to: '/student' }, { label: 'My Improvement Tasks' }]}
        back={{ to: '/student', label: 'Back to Student Dashboard' }}
        eyebrow="Student Portal"
        title="My Improvement Tasks"
        description="Improvement tasks your faculty assigned to you, saved with your account."
      />

      <section className={CARD} data-testid="student-tasks">
        <SectionHeading title="Tasks" description={tasks?.length ? `${tasks.length} assigned · ${completed} completed · ${tasks.length - completed} pending` : null} />
        {!tasks ? (
          <ResourceState resource={resource} what="improvement tasks" />
        ) : tasks.length === 0 ? (
          <div className="rounded-[20px] border border-dashed border-slate-200 bg-slate-50 p-6 text-center">
            <p className="text-sm font-semibold text-slate-700">No improvement tasks assigned yet</p>
            <p className="mt-1 text-sm text-slate-500">Improvement tasks created by your faculty will appear here.</p>
          </div>
        ) : (
          <ul className="space-y-3">
            {tasks.map((task) => (
              <InterventionItem
                key={task.id}
                intervention={task}
                action={task.status === 'assigned' ? (
                  <>
                    <button type="button" onClick={() => markComplete(task)} disabled={busyId != null} aria-busy={busyId === task.id} className={BTN_PRIMARY}>
                      {busyId === task.id ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Saving…</> : <><CheckCircle2 size={16} aria-hidden="true" /> Mark as Complete</>}
                    </button>
                    <FormError message={errors[task.id]} />
                  </>
                ) : null}
              />
            ))}
          </ul>
        )}
      </section>
    </>
  )
}

// /student shows the dashboard, or the improvement tasks with ?view=interventions
function StudentHome() {
  const [searchParams] = useSearchParams()
  return searchParams.get('view') === 'interventions' ? <ImprovementTasksPage /> : <StudentDashboard />
}

function StudentDashboard() {
  // The signed-in student's stored record, loaded once by StudentLayout
  const studentResource = useOutletContext()
  const detail = studentResource.data
  const { user } = useAuth()
  const tasksResource = useApiResource(`/students/${user.student_id}/interventions`)
  const tasks = tasksResource.data?.interventions
  const completedTasks = tasks ? tasks.filter((item) => item.status === 'completed').length : 0
  const latest = detail && hasCurrentPrediction(detail) ? detail.latest_prediction : null

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Student Portal' }, { label: 'Dashboard' }]}
        eyebrow="Student Portal"
        title="Student Dashboard"
        description="Your academic record, prediction status and latest stored prediction, with its SHAP explanation and recommendations."
      />

      {!detail ? (
        <section className={CARD}><ResourceState resource={studentResource} what="academic record" /></section>
      ) : (
        <>
          {/* The signed-in student and their current prediction status */}
          <section className="ui-card p-5 sm:p-6" data-testid="student-summary">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="ui-eyebrow">Your academic overview</p>
                <p className="mt-1 text-2xl font-semibold text-slate-950 sm:text-3xl">{detail.student.name}</p>
                <p className="mt-1 text-sm text-slate-600">
                  Student ID: <span className="font-semibold text-slate-800">{detail.student.roll_no}</span>
                  {[detail.student.program, detail.student.section && `Section ${detail.student.section}`].filter(Boolean).map((item) => ` · ${item}`).join('')}
                </p>
                {latest ? (
                  <div className="mt-4 flex flex-wrap items-center gap-3">
                    <p className="text-base font-semibold text-slate-900">
                      Predicted final-exam marks: {latest.predicted_marks} / 100
                      <span className="mt-1 block text-sm font-normal text-slate-600">Model estimate, not actual examination marks.</span>
                    </p>
                    <EarlyWarningBadge risk={latest} />
                  </div>
                ) : (
                  <div className="mt-3">
                    <p className="text-lg font-semibold text-slate-900">{PREDICTION_STATUS[detail.prediction_status.status]?.label}</p>
                    <p className="text-sm text-slate-600">{detail.prediction_status.message}</p>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-3">
                <NavLink to="/what-if" className={BTN_SECONDARY}>
                  Practice What-If scenarios <ArrowRightLeft size={16} aria-hidden="true" />
                </NavLink>
              </div>
            </div>
          </section>

          <div className="mt-6">
            <StudentRecordDetails detail={detail} onDetailChange={studentResource.replace} />
          </div>
        </>
      )}

      {/* Improvement tasks assigned by faculty, from the database */}
      <section className={`${CARD} mt-6`} data-testid="student-task-summary">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-semibold text-slate-900"><ListChecks size={18} className="text-blue-600" aria-hidden="true" />Improvement Tasks</h2>
            {!tasks ? (
              <div className="mt-1"><ResourceState resource={tasksResource} what="improvement tasks" /></div>
            ) : tasks.length === 0 ? (
              <p className="mt-1 text-sm text-slate-600">No improvement tasks assigned yet. Tasks created by your faculty will appear here.</p>
            ) : (
              <ul className="mt-2 flex flex-wrap gap-2 text-sm">
                {[
                  [`${tasks.length} assigned`, 'border-sky-200 bg-sky-50 text-sky-700'],
                  [`${completedTasks} completed`, 'border-emerald-200 bg-emerald-50 text-emerald-700'],
                  [`${tasks.length - completedTasks} pending`, 'border-slate-200 bg-slate-50 text-slate-700'],
                ].map(([label, tone]) => <li key={label} className={`rounded-full border px-3 py-1 font-semibold ${tone}`}>{label}</li>)}
              </ul>
            )}
          </div>
          <Link to="/student?view=interventions" className={`${BTN_SECONDARY} shrink-0`}>View Improvement Tasks <ArrowRight size={14} aria-hidden="true" /></Link>
        </div>
      </section>

      <section className={`${CARD} mt-6`}>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="ui-section-title">Model performance</p>
            <p className="mt-1 text-xs font-semibold uppercase tracking-wide text-slate-500">5-fold cross-validation</p>
            <p className="mt-1 text-sm leading-6 text-slate-600">Final Student Performance Regression Model. Mean ± standard deviation across 5 folds; MAE and RMSE are in final-exam marks out of 100. These describe the model overall, not an individual student's forecast.</p>
          </div>
          <div className="grid w-full shrink-0 grid-cols-1 gap-2 sm:grid-cols-3 lg:w-auto">
            {[
              ['MAE', '3.62 ± 0.07'],
              ['RMSE', '4.59 ± 0.09'],
              ['R²', '0.8357 ± 0.0084'],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center justify-between gap-3 rounded-[16px] border border-slate-200 bg-slate-50 px-3 py-2 text-left sm:block sm:text-center">
                <p className="text-xs text-slate-500">{label}</p>
                <p className="mt-0.5 text-sm font-semibold tabular-nums text-slate-900">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}

const FACULTY_VIEWS = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'student-accounts', label: 'Student Accounts', icon: Users },
  { id: 'create-student', label: 'Create Student', icon: UserPlus },
  { id: 'bulk-registration', label: 'Bulk Student Registration', icon: Users },
  { id: 'bulk-upload', label: 'Bulk Academic Upload', icon: Upload },
  { id: 'intervention', label: 'Student Intervention Center', icon: Users },
  { id: 'analysis', label: 'Student AI Analysis', icon: BrainCog },
  { id: 'management', label: 'Intervention Management', icon: ClipboardList },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'profile', label: 'Profile', icon: CircleUserRound },
]

// Early-warning bands, in display order, with their bar colours
const RISK_BANDS = [
  { code: 'high_concern', label: 'High Concern', bar: 'bg-rose-500', marker: 'bg-rose-500', text: 'text-rose-800' },
  { code: 'borderline', label: 'Borderline', bar: 'bg-amber-500', marker: 'bg-amber-500', text: 'text-amber-800' },
  { code: 'lower_concern', label: 'Lower Concern', bar: 'bg-emerald-500', marker: 'bg-emerald-500', text: 'text-emerald-800' },
]

const INTERVENTION_FILTERS = ['All', 'Open', 'Completed', 'Overdue']

function RiskDistribution({ counts, currentTotal, studentTotal, emptyAction, testId }) {
  return (
    <GlassCard className="p-5 sm:p-6" data-testid={testId}>
      <SectionHeading
        title="Early-warning distribution"
        description="Share of students with a current prediction in each model-based risk category."
      />
      {currentTotal === 0 ? (
        <EmptyState
          message={studentTotal === 0 ? 'No students are available yet.' : 'No student has a current prediction yet.'}
          note={studentTotal === 0
            ? 'Student records will appear here when they are available in the database.'
            : 'Students without a current prediction are not included in this distribution.'}
          action={emptyAction}
        />
      ) : (
        <div
          role="group"
          aria-label={`Early-warning distribution for ${currentTotal} students with a current prediction, out of ${studentTotal} students total.`}
          className="space-y-5"
        >
          {counts.map(({ code, label, bar, marker, text, count }) => {
            const percentage = Math.round((count / currentTotal) * 100)
            return (
              <div key={code}>
                <div className="mb-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
                  <span className="inline-flex items-center gap-2 text-sm font-semibold text-slate-800">
                    <span className={`h-2.5 w-2.5 rounded-full ${marker}`} aria-hidden="true" />
                    {label}
                  </span>
                  <span className={`text-sm font-semibold ${text}`}>
                    {count} {count === 1 ? 'student' : 'students'} · {percentage}%
                  </span>
                </div>
                <div
                  className="h-2.5 overflow-hidden rounded-full bg-slate-100"
                  role="progressbar"
                  aria-label={`${label}: ${count} of ${currentTotal} students`}
                  aria-valuemin={0}
                  aria-valuemax={currentTotal}
                  aria-valuenow={count}
                >
                  <div className={`h-full rounded-full ${bar}`} style={{ width: `${percentage}%` }} />
                </div>
              </div>
            )
          })}
          <p className="border-t border-slate-100 pt-3 text-xs leading-5 text-slate-500">
            Based on {currentTotal} current prediction{currentTotal === 1 ? '' : 's'} across {studentTotal} student{studentTotal === 1 ? '' : 's'}. Students without a current prediction are excluded.
          </p>
        </div>
      )}
    </GlassCard>
  )
}

// Two-letter avatar initials for a student's name
const initials = (name) => name.split(' ').filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase()

function FacultyDashboard() {
  // Every view reads the student database: GET /students (all students with their prediction
  // status), GET /students/<id> (one student in full) and GET /interventions (all interventions).
  const { user } = useAuth()
  // The active view lives in the URL (?view=...), so other pages can link straight to it
  const [searchParams, setSearchParams] = useSearchParams()
  const viewParam = searchParams.get('view')
  const activeView = FACULTY_VIEWS.some((view) => view.id === viewParam) ? viewParam : 'dashboard'

  const listResource = useApiResource('/students')
  const interventionsResource = useApiResource('/interventions')
  const realStudents = useMemo(() => listResource.data?.students ?? [], [listResource.data])
  const allInterventions = useMemo(() => interventionsResource.data?.interventions ?? [], [interventionsResource.data])
  // One selected student, shared by Student AI Analysis, Intervention Management and Reports
  const [selectedStudentId, setSelectedStudentId] = useState(null)
  const analysisId = selectedStudentId ?? realStudents[0]?.id ?? null
  const analysisStudent = realStudents.find((student) => student.id === analysisId) || null
  const detailResource = useApiResource(['analysis', 'management', 'reports'].includes(activeView) && analysisId != null ? `/students/${analysisId}` : null)
  const byMarks = (a, b) => a.latest_prediction.predicted_marks - b.latest_prediction.predicted_marks
  const realCurrent = realStudents.filter(hasCurrentPrediction).sort(byMarks)
  const realAlerts = realCurrent.filter((student) => ['high_concern', 'borderline'].includes(student.latest_prediction.risk_code))
  const realWithoutCurrent = realStudents.filter((student) => !hasCurrentPrediction(student))
  const countRealBand = (code) => realCurrent.filter((student) => student.latest_prediction.risk_code === code).length
  const riskCounts = RISK_BANDS.map((band) => ({
    ...band,
    count: countRealBand(band.code),
  }))
  const totals = interventionCounts(allInterventions)
  const interventionsFor = (studentId) => allInterventions.filter((item) => item.student_id === studentId)
  const [search, setSearch] = useState('')
  const [riskFilter, setRiskFilter] = useState('All')
  const [interventionFilter, setInterventionFilter] = useState('All')
  const [accountActionStudentId, setAccountActionStudentId] = useState(null)
  const [accountActionError, setAccountActionError] = useState('')
  // Intervention Management: the intervention being written for the selected student
  const [managementDraft, setManagementDraft] = useState(null)

  // Prediction server status from GET /: null while checking, then 'ok' or 'error'
  const [apiStatus, setApiStatus] = useState(null)
  const [statusCheck, setStatusCheck] = useState(0)

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE_URL}/`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => { if (!cancelled) setApiStatus(data.model_loaded && data.shap_loaded ? 'ok' : 'error') })
      .catch(() => { if (!cancelled) setApiStatus('error') })
    return () => { cancelled = true }
  }, [statusCheck])

  const retryStatus = () => {
    setApiStatus(null)
    setStatusCheck((count) => count + 1)
  }

  const changeStudentAccountStatus = async (student) => {
    const nextActive = !student.account_active
    setAccountActionStudentId(student.id)
    setAccountActionError('')
    try {
      const result = await requestStudentAccountStatusChange(student, nextActive, {
        confirmAction: (message) => window.confirm(message),
        request: apiRequest,
      })
      if (result) await refreshStudents()
    } catch (error) {
      setAccountActionError(error.message)
    } finally {
      setAccountActionStudentId(null)
    }
  }

  const { refresh: refreshStudents } = listResource
  const { refresh: refreshInterventions } = interventionsResource

  // Changing view also re-reads the overview data in the background, so changes made in
  // another view (a new prediction or intervention) are reflected without a reload
  const setActiveView = (viewId) => {
    setSearchParams(viewId === 'dashboard' ? {} : { view: viewId })
    window.scrollTo({ top: 0 })
    refreshStudents()
    refreshInterventions()
  }

  const selectStudent = (studentId) => {
    setSelectedStudentId(studentId)
    setManagementDraft(null)
  }

  const openStudentView = (studentId, viewId) => {
    selectStudent(studentId)
    setActiveView(viewId)
  }

  // Opens Student AI Analysis for a student from the database
  const openAnalysis = (studentId) => openStudentView(studentId, 'analysis')

  const linkButton = 'inline-flex items-center gap-1 rounded-full px-3 py-2 text-sm font-semibold text-blue-700 transition hover:bg-blue-50'

  // One database student in a dashboard list: marks and band only for a current prediction, otherwise the status
  const renderRealStudentRow = (student, dashed = false) => {
    const current = hasCurrentPrediction(student)
    const riskBorder = {
      high_concern: 'border-l-rose-500',
      borderline: 'border-l-amber-500',
      lower_concern: 'border-l-emerald-500',
    }[student.latest_prediction?.risk_code] || 'border-l-slate-300'
    return (
      <article
        key={student.id}
        className={`flex flex-col gap-4 rounded-xl border border-l-4 p-4 sm:flex-row sm:items-center sm:justify-between ${current ? riskBorder : 'border-l-slate-300'} ${dashed ? 'border-dashed bg-white' : 'bg-white'}`}
        data-real-student={student.id}
      >
        <div className="flex min-w-0 flex-1 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-5">
          <div className="min-w-0">
            <p className="break-words font-semibold text-slate-900">{student.name}</p>
            <p className="mt-0.5 break-all text-sm text-slate-500">{student.roll_no}</p>
          </div>
          <div className="flex flex-wrap items-center gap-3 sm:justify-end">
            {current ? (
              <>
                <EarlyWarningBadge risk={student.latest_prediction} />
                <div className="rounded-lg bg-slate-50 px-3 py-2">
                  <p className="text-xs font-medium text-slate-500">Predicted marks</p>
                  <p className="mt-0.5 text-sm font-semibold tabular-nums text-slate-900">
                    {student.latest_prediction.predicted_marks} <span className="font-medium text-slate-500">/ 100</span>
                  </p>
                </div>
              </>
            ) : (
              <PredictionStatusBadge status={student.prediction_status.status} />
            )}
          </div>
        </div>
        <button
          type="button"
          onClick={() => openAnalysis(student.id)}
          className="ui-btn ui-btn-secondary w-full shrink-0 sm:w-auto"
        >
          View student <ArrowRight size={14} aria-hidden="true" />
        </button>
      </article>
    )
  }

  // Student selector for the views that show one student (Management, Reports)
  const renderStudentPicker = (id) => (
    <div className="print:hidden">
      <label htmlFor={id} className="text-sm font-semibold text-slate-700">Student</label>
      <select id={id} value={analysisId ?? ''} onChange={(event) => selectStudent(Number(event.target.value))} className="mt-2 block w-full rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-900 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100 sm:w-80">
        {realStudents.map((student) => (
          <option key={student.id} value={student.id}>{student.name} ({student.roll_no})</option>
        ))}
      </select>
    </div>
  )

  const interventionCountsText = (counts) => (counts.total === 0
    ? 'No interventions assigned yet'
    : `${counts.total} assigned · ${counts.completed} completed · ${counts.open} open${counts.overdue ? ` · ${counts.overdue} overdue` : ''}`)

  const viewLabel = FACULTY_VIEWS.find((view) => view.id === activeView).label
  const headers = {
    dashboard: { title: 'What needs attention?', description: 'Students from the student database with their prediction status and latest stored prediction.' },
    'student-accounts': { title: 'Student Accounts', description: 'Review student account access and deactivate or reactivate accounts without changing student records.' },
    'create-student': { title: 'Create Student', description: 'Create a student identity and login account without adding academic data or predictions.' },
    'bulk-registration': { title: 'Bulk Student Registration', description: 'Validate, preview and create student login accounts from a CSV file.' },
    'bulk-upload': { title: 'Bulk Academic Upload', description: 'Validate, preview and import academic records for existing student accounts.' },
    intervention: { title: 'Student Intervention Center', description: 'Every student with their prediction status and interventions, lowest predicted marks first.' },
    analysis: { title: analysisStudent ? analysisStudent.name : 'Student AI Analysis', description: 'Stored academic record, prediction status and latest stored prediction for this student.' },
    management: { title: 'Intervention Management', description: 'Assign interventions from stored recommendations and track every student\'s interventions.' },
    analytics: { title: 'Analytics', description: 'Model performance, the early-warning distribution and an intervention summary from the student database.' },
    reports: { title: 'Reports', description: 'A consolidated report for one student: record, prediction, explanation, history and interventions.' },
    profile: { title: 'Profile', description: 'Faculty account overview.' },
  }
  const breadcrumbs = activeView === 'dashboard'
    ? [{ label: 'Teacher Portal' }, { label: 'Dashboard' }]
    : [
      { label: 'Teacher Portal', onClick: () => setActiveView('dashboard') },
      activeView === 'analysis' && analysisStudent ? { label: viewLabel, onClick: () => setActiveView('analysis') } : { label: viewLabel },
      ...(activeView === 'analysis' && analysisStudent ? [{ label: analysisStudent.name }] : []),
    ]

  const renderView = () => {
    switch (activeView) {
      case 'student-accounts': {
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        return (
          <section className={CARD} data-testid="student-account-management">
            <SectionHeading
              title="Student account access"
              description="Deactivation prevents sign-in and invalidates existing sessions. Academic records, predictions, interventions and audit history remain unchanged."
            />
            {accountActionError && (
              <p className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800" role="alert">
                {accountActionError}
              </p>
            )}
            {realStudents.length === 0 ? (
              <div className="mt-4"><EmptyState message="No students are available in the database yet." /></div>
            ) : (
              <ul className="mt-4 divide-y divide-slate-100">
                {realStudents.map((student) => {
                  const hasAccount = student.account_active !== null
                  const isPending = accountActionStudentId === student.id
                  return (
                    <li key={student.id} className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between" data-account-student={student.id}>
                      <div>
                        <p className="font-semibold text-slate-900">{student.name}</p>
                        <p className="text-sm text-slate-500">{student.roll_no}</p>
                      </div>
                      <div className="flex flex-wrap items-center gap-3">
                        <span
                          className={`ui-badge ${student.account_active === false ? 'border-amber-200 bg-amber-50 text-amber-800' : student.account_active === true ? 'border-emerald-200 bg-emerald-50 text-emerald-800' : 'border-slate-200 bg-slate-50 text-slate-600'}`}
                          data-testid={`account-status-${student.id}`}
                        >
                          {accountStatusLabel(student.account_active)}
                        </span>
                        {hasAccount && (
                          <button
                            type="button"
                            className={student.account_active ? BTN_SECONDARY : BTN_PRIMARY}
                            disabled={accountActionStudentId !== null}
                            onClick={() => changeStudentAccountStatus(student)}
                            aria-label={`${student.account_active ? 'Deactivate' : 'Reactivate'} ${student.name}`}
                          >
                            {isPending
                              ? 'Saving…'
                              : student.account_active ? 'Deactivate account' : 'Reactivate account'}
                          </button>
                        )}
                      </div>
                    </li>
                  )
                })}
              </ul>
            )}
          </section>
        )
      }
      case 'create-student':
        return <CreateStudentView onBack={() => setActiveView('dashboard')} onCreated={refreshStudents} />
      case 'bulk-registration':
        return <BulkStudentRegistrationView onBack={() => setActiveView('dashboard')} onRegistered={refreshStudents} />
      case 'bulk-upload':
        return <BulkUploadView onImported={() => { refreshStudents(); refreshInterventions() }} />
      case 'intervention': {
        // Every student, filtered by name/roll number and early-warning band, with their interventions
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        const query = search.trim().toLowerCase()
        const matches = realStudents.filter((student) => {
          const matchesSearch = !query || student.name.toLowerCase().includes(query) || student.roll_no.toLowerCase().includes(query)
          const current = hasCurrentPrediction(student)
          const matchesRisk = riskFilter === 'All'
            || (riskFilter === 'No current prediction' ? !current : current && student.latest_prediction.risk_level === riskFilter)
          return matchesSearch && matchesRisk
        })
        const ordered = [...matches.filter(hasCurrentPrediction).sort(byMarks), ...matches.filter((student) => !hasCurrentPrediction(student))]
        return (
          <div className="space-y-6">
            <div className="flex flex-wrap gap-3">
              <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100">
                <Search size={16} aria-hidden="true" />
                <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search name or roll number" aria-label="Search students" className="w-52 bg-transparent outline-none" />
              </div>
              <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)} aria-label="Filter by early-warning status" className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100">
                <option value="All">All students</option>
                <option value="High Concern">High Concern</option>
                <option value="Borderline">Borderline</option>
                <option value="Lower Concern">Lower Concern</option>
                <option value="No current prediction">No current prediction</option>
              </select>
            </div>

            {interventionsResource.error && <ResourceState resource={interventionsResource} what="interventions" />}
            {ordered.length === 0 && <EmptyState message="No student matches the current search and filter." />}

            <div className="grid gap-4 xl:grid-cols-2">
              {ordered.map((student) => {
                const current = hasCurrentPrediction(student)
                const studentInterventions = interventionsFor(student.id)
                const openItems = studentInterventions.filter((item) => item.status === 'assigned')
                return (
                  <motion.div key={student.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} data-center-student={student.id} className="flex flex-col rounded-[24px] border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 text-sm font-semibold text-white" aria-hidden="true">{initials(student.name)}</div>
                        <div>
                          <p className="font-semibold text-slate-900">{student.name}</p>
                          <p className="text-sm text-slate-500">{student.roll_no}{current ? ` · Predicted ${student.latest_prediction.predicted_marks} / 100` : ''}</p>
                        </div>
                      </div>
                      {current ? <EarlyWarningBadge risk={student.latest_prediction} /> : <PredictionStatusBadge status={student.prediction_status.status} />}
                    </div>
                    {!current && <p className="mt-3 text-sm text-slate-600">{student.prediction_status.message}</p>}
                    <p className="mt-4 text-sm font-semibold text-slate-700">Interventions</p>
                    <p className="mt-1 text-sm text-slate-600">{interventionCountsText(interventionCounts(studentInterventions))}</p>
                    {openItems.length > 0 && (
                      <ul className="mt-2 space-y-1 text-sm text-slate-700">
                        {openItems.slice(0, 2).map((item) => (
                          <li key={item.id} className="rounded-[14px] border border-slate-200 bg-slate-50 px-3 py-2">
                            <span className="font-semibold text-slate-900">{item.title}</span>
                            <span className="text-slate-500"> · {item.category}{item.due_date ? ` · due ${formatDay(item.due_date)}` : ''}</span>
                            {isOverdue(item) && <span className="ml-2 text-xs font-semibold text-rose-700">Overdue</span>}
                          </li>
                        ))}
                      </ul>
                    )}
                    <div className="mt-auto flex flex-wrap items-center gap-2 pt-5">
                      <button type="button" onClick={() => openAnalysis(student.id)} className={BTN_SECONDARY}>View AI Analysis</button>
                      <button type="button" onClick={() => openStudentView(student.id, 'management')} className={BTN_SECONDARY}>Manage interventions</button>
                      <button type="button" onClick={() => openStudentView(student.id, 'reports')} className={BTN_SECONDARY}>View report</button>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          </div>
        )
      }
      case 'analysis': {
        // The selected student's stored data, exactly as returned by GET /students/<id>
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        if (realStudents.length === 0) return <EmptyState message="No students found in the student database." />
        const detail = detailResource.data
        const current = detail && hasCurrentPrediction(detail) ? detail.latest_prediction : null
        return (
          <div className="space-y-6">
            {/* Student header: who, current result or status, and the student switcher */}
            <section className="rounded-[24px] border border-slate-200 bg-gradient-to-br from-slate-950 via-blue-950 to-cyan-900 p-6 text-white shadow-lg" data-testid="faculty-student-header">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="text-sm font-semibold text-cyan-300">{[analysisStudent.roll_no, analysisStudent.program, analysisStudent.section && `Section ${analysisStudent.section}`].filter(Boolean).join(' · ')}</p>
                  <p className="mt-1 text-3xl font-semibold">{analysisStudent.name}</p>
                  {current ? (
                    <div className="mt-2 flex flex-wrap items-center gap-3">
                      <p className="text-xl font-semibold">{current.predicted_marks} marks</p>
                      <EarlyWarningBadge risk={current} />
                    </div>
                  ) : detail && (
                    <div className="mt-3"><PredictionStatusBadge status={detail.prediction_status.status} /></div>
                  )}
                </div>
                <div>
                  <label htmlFor="faculty-student" className="text-xs font-semibold text-blue-100">Switch student</label>
                  <select id="faculty-student" value={analysisId} onChange={(event) => selectStudent(Number(event.target.value))} className="mt-1 block w-full rounded-full border border-white/30 bg-white px-3 py-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-300 sm:w-72">
                    {realStudents.map((student) => (
                      <option key={student.id} value={student.id}>{student.name} ({student.roll_no})</option>
                    ))}
                  </select>
                </div>
              </div>
            </section>

            {detail
              ? (
                <StudentRecordDetails
                  key={detail.student.id}
                  detail={detail}
                  onDetailChange={(updated) => {
                    detailResource.replace(updated)
                    // Keep the dashboard list's status and marks in step with the change
                    listResource.refresh()
                  }}
                />
              )
              : <section className={CARD}><ResourceState resource={detailResource} what="student record" /></section>}
          </div>
        )
      }
      case 'management': {
        // Create interventions from one student's stored recommendations; track everyone's interventions
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        if (realStudents.length === 0) return <EmptyState message="No students found in the student database." />
        const detail = detailResource.data
        const latest = detail?.latest_prediction ?? null
        const actionable = latest ? mapRecommendations(latest.recommendations).filter((rec) => rec.needsAttention) : []
        const shown = allInterventions.filter((item) => (
          interventionFilter === 'All'
          || (interventionFilter === 'Open' && item.status === 'assigned')
          || (interventionFilter === 'Completed' && item.status === 'completed')
          || (interventionFilter === 'Overdue' && isOverdue(item))
        ))
        return (
          <div className="space-y-6">
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {[
                ['Interventions', totals.total, 'Assigned to all students'],
                ['Open', totals.open, 'Not yet completed'],
                ['Completed', totals.completed, totals.total ? `${Math.round((totals.completed / totals.total) * 100)}% of assigned` : 'None assigned yet'],
                ['Overdue', totals.overdue, 'Open and past the due date'],
              ].map(([label, value, detailText]) => (
                <div key={label} className="rounded-[20px] border border-slate-200 bg-white p-4 shadow-sm" data-mgmt-tile={label}>
                  <p className="text-sm text-slate-500">{label}</p>
                  <p className="mt-1 text-2xl font-semibold text-slate-900">{value}</p>
                  <p className="text-xs text-slate-500">{detailText}</p>
                </div>
              ))}
            </div>

            <section className={CARD} data-testid="mgmt-create">
              <SectionHeading title="Assign from stored recommendations" description="Choose a student. The HIGH and MEDIUM recommendations stored with their latest prediction can start a new intervention; nothing is regenerated." />
              {renderStudentPicker('mgmt-student')}
              {!detail ? (
                <div className="mt-5"><ResourceState resource={detailResource} what="student record" /></div>
              ) : (
                <div className="mt-5 space-y-5">
                  {latest ? (
                    <div data-testid="mgmt-recommendations">
                      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm text-slate-600">
                        <span>{hasCurrentPrediction(detail) ? 'Latest prediction' : 'Last generated prediction (the record has changed since)'} #{latest.id}: <span className="font-semibold text-slate-900">{latest.predicted_marks} / 100</span></span>
                        <EarlyWarningBadge risk={latest} />
                      </div>
                      {actionable.length === 0
                        ? <p className="text-sm text-slate-600">No HIGH or MEDIUM recommendations were stored with this prediction.</p>
                        : <RecommendationGroups recommendations={actionable} strongest={findStrongestNegative(actionable)} renderFooter={(rec) => <CreateInterventionButton onClick={() => setManagementDraft(draftFromRecommendation(rec, latest.id))} />} />}
                    </div>
                  ) : (
                    <EmptyState
                      message={PREDICTION_STATUS[detail.prediction_status.status]?.label}
                      note={`${detail.prediction_status.message} Recommendations come from a stored prediction, generated in Student AI Analysis.`}
                      action={<button type="button" onClick={() => openAnalysis(detail.student.id)} className={BTN_SECONDARY}>Open Student AI Analysis <ArrowRight size={14} aria-hidden="true" /></button>}
                    />
                  )}
                  <FacultyInterventions key={detail.student.id} studentId={detail.student.id} draft={managementDraft} onDraftChange={setManagementDraft} onCreated={refreshInterventions} />
                </div>
              )}
            </section>

            <section className={CARD} data-testid="mgmt-all">
              <SectionHeading title="All interventions" description="Every student's interventions, newest first. Students mark them completed in My Improvement Tasks." />
              <div role="group" aria-label="Filter interventions" className="mb-4 flex flex-wrap gap-2">
                {INTERVENTION_FILTERS.map((filter) => (
                  <button key={filter} type="button" aria-pressed={interventionFilter === filter} onClick={() => setInterventionFilter(filter)} className={`rounded-full border px-4 py-1.5 text-sm font-semibold transition ${interventionFilter === filter ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-200 bg-white text-slate-700 hover:border-blue-200 hover:text-blue-700'}`}>{filter}</button>
                ))}
              </div>
              {!interventionsResource.data ? (
                <ResourceState resource={interventionsResource} what="interventions" />
              ) : shown.length === 0 ? (
                <EmptyState message={allInterventions.length === 0 ? 'No interventions have been assigned yet.' : 'No interventions match this filter.'} />
              ) : (
                <ul className="space-y-3">
                  {shown.map((item) => (
                    <InterventionItem key={item.id} intervention={item} showStudent action={<button type="button" onClick={() => openAnalysis(item.student_id)} className={`${linkButton} -ml-3`}>View student <ArrowRight size={14} aria-hidden="true" /></button>} />
                  ))}
                </ul>
              )}
            </section>
          </div>
        )
      }
      case 'analytics': {
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        const statusCounts = Object.keys(PREDICTION_STATUS).map((code) => [code, realStudents.filter((student) => student.prediction_status.status === code).length])
        const categoryCounts = INTERVENTION_CATEGORIES.map((category) => [category, allInterventions.filter((item) => item.category === category).length]).filter(([, count]) => count > 0)
        const maxCategory = Math.max(1, ...categoryCounts.map(([, count]) => count))
        return (
          <div className="space-y-6">
            <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
              {/* Real 5-fold cross-validation results of the production regression model */}
              <GlassCard className="p-6">
                <p className="text-sm font-semibold text-slate-500">Model Performance</p>
                <h4 className="mt-1 text-lg font-semibold text-slate-900">5-Fold Cross-Validation</h4>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  {[
                    ['MAE', '3.62 ± 0.07'],
                    ['RMSE', '4.59 ± 0.09'],
                    ['R²', '0.8357 ± 0.0084'],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-[18px] border border-slate-200 bg-slate-50 p-4 text-center">
                      <p className="text-sm text-slate-500">{label}</p>
                      <p className="mt-2 text-xl font-semibold text-slate-900">{value}</p>
                    </div>
                  ))}
                </div>
                <p className="mt-4 text-sm leading-7 text-slate-600">Performance measured using 5-fold cross-validation on the project dataset. MAE and RMSE are in final-exam marks (out of 100).</p>
              </GlassCard>

              <RiskDistribution
                counts={riskCounts}
                currentTotal={realCurrent.length}
                studentTotal={realStudents.length}
                testId="analytics-risk"
                emptyAction={realStudents.length > 0
                  ? <button type="button" onClick={() => setActiveView('analysis')} className={BTN_SECONDARY}>Open Student AI Analysis <ArrowRight size={14} aria-hidden="true" /></button>
                  : null}
              />
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <GlassCard className="p-6" data-testid="analytics-status">
                <p className="text-sm font-semibold text-slate-500">Prediction status of all students</p>
                <ul className="mt-4 space-y-2">
                  {statusCounts.map(([code, count]) => (
                    <li key={code} className="flex items-center justify-between gap-3">
                      <PredictionStatusBadge status={code} />
                      <span className="text-sm font-semibold text-slate-900">{count}</span>
                    </li>
                  ))}
                </ul>
              </GlassCard>

              <GlassCard className="p-6" data-testid="analytics-interventions">
                <p className="text-sm font-semibold text-slate-500">Interventions</p>
                {!interventionsResource.data ? (
                  <div className="mt-4"><ResourceState resource={interventionsResource} what="interventions" /></div>
                ) : (
                  <>
                    <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                      {[['Assigned', totals.total], ['Completed', totals.completed], ['Open', totals.open], ['Overdue', totals.overdue]].map(([label, value]) => (
                        <div key={label} className="rounded-[16px] border border-slate-200 bg-slate-50 px-3 py-2 text-center">
                          <p className="text-xs text-slate-500">{label}</p>
                          <p className="text-lg font-semibold text-slate-900">{value}</p>
                        </div>
                      ))}
                    </div>
                    {categoryCounts.length > 0 && (
                      <div className="mt-5 space-y-3">
                        {categoryCounts.map(([category, count]) => (
                          <div key={category}>
                            <div className="mb-1 flex items-center justify-between text-sm text-slate-600"><span>{category}</span><span>{count}</span></div>
                            <div className="h-2 rounded-full bg-slate-100"><div className="h-2 rounded-full bg-blue-500" style={{ width: `${Math.round((count / maxCategory) * 100)}%` }} /></div>
                          </div>
                        ))}
                      </div>
                    )}
                    <p className="mt-4 text-xs leading-5 text-slate-500">Assignment and completion are recorded; whether an intervention improved a student's performance is not measured.</p>
                  </>
                )}
              </GlassCard>
            </div>

            <div className="grid gap-6 xl:grid-cols-3">
              <GlassCard className="p-6">
                <p className="text-sm font-semibold text-slate-500">Department Performance</p>
                <div className="mt-4"><EmptyState message="Department performance needs department records, which this project does not store." /></div>
              </GlassCard>

              <GlassCard className="p-6">
                <p className="text-sm font-semibold text-slate-500">Attendance Trends</p>
                <div className="mt-4"><EmptyState message="Attendance trends need attendance records over time; only each student's current record is stored." /></div>
              </GlassCard>

              <GlassCard className="p-6">
                <p className="text-sm font-semibold text-slate-500">Heatmap</p>
                <div className="mt-4"><EmptyState message="A heatmap is not implemented in this project." /></div>
              </GlassCard>
            </div>
          </div>
        )
      }
      case 'reports': {
        // Everything comes from the stored record, stored predictions and stored interventions
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        if (realStudents.length === 0) return <EmptyState message="No students found in the student database." />
        const detail = detailResource.data
        const latest = detail?.latest_prediction ?? null
        const isCurrent = Boolean(detail) && hasCurrentPrediction(detail)
        const recommendations = latest ? mapRecommendations(latest.recommendations) : []
        const positive = recommendations.filter((rec) => rec.priority === 'POSITIVE')
        const negative = recommendations.filter((rec) => rec.needsAttention)
        const studentInterventions = interventionsFor(analysisId)
        const reportSection = (id, title, description, children) => (
          <section id={id} className={`${CARD} scroll-mt-6 break-inside-avoid`} data-report={id}>
            <SectionHeading title={title} description={description} />
            {children}
          </section>
        )
        return (
          <div className="space-y-6">
            {/* Report header: which student, and how to navigate or print the report */}
            <section className={CARD}>
              <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                {renderStudentPicker('faculty-report-student')}
                <div className="flex flex-wrap gap-2 print:hidden">
                  {[['report-student', 'Student'], ['report-prediction', 'Prediction'], ['report-history', 'History'], ['report-intervention', 'Intervention'], ['report-department', 'Department']].map(([id, label]) => (
                    <a key={id} href={`#${id}`} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:border-blue-200 hover:text-blue-700">{label}</a>
                  ))}
                  {detail && <button type="button" onClick={() => window.print()} className={BTN_SECONDARY}><FileText size={14} aria-hidden="true" />Print report</button>}
                </div>
              </div>
            </section>

            {!detail ? (
              <section className={CARD}><ResourceState resource={detailResource} what="student record" /></section>
            ) : (
              <>
                {reportSection('report-student', 'Student Report', 'Student information, stored academic record and prediction status.', (
                  <div className="space-y-5">
                    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-100 bg-blue-50/50 p-4">
                      <div>
                        <p className="text-lg font-semibold text-slate-900">{detail.student.name}</p>
                        <p className="mt-1 text-sm text-slate-600">
                          Student ID: <span className="font-semibold text-slate-800">{detail.student.roll_no}</span>
                          {[detail.student.program, detail.student.section && `Section ${detail.student.section}`].filter(Boolean).map((item) => ` · ${item}`).join('')}
                        </p>
                      </div>
                      <PredictionStatusBadge status={detail.prediction_status.status} />
                    </div>
                    <InputsList inputs={detail.student.record} wide />
                    <p className="text-sm leading-6 text-slate-600">{detail.prediction_status.message}</p>
                  </div>
                ))}

                {reportSection('report-prediction', 'Prediction Report', 'The stored prediction with its SHAP explanation and recommendations.', latest ? (
                  <div className="space-y-6">
                    {!isCurrent && (
                      <p role="note" className="rounded-[18px] border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900">This is the last generated prediction. The academic record has changed since, so it does not describe the current record.</p>
                    )}
                    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-blue-100 bg-blue-50/50 p-4 text-sm text-slate-700">
                      <span>
                        Predicted final-exam marks: <span className="text-2xl font-bold tabular-nums text-slate-950" data-testid="report-marks">{latest.predicted_marks} / 100</span>
                        <span className="mt-1 block text-xs text-slate-600">Model estimate, not actual examination marks.</span>
                      </span>
                      <EarlyWarningBadge risk={latest} />
                      <span className="text-xs text-slate-500">Prediction #{latest.id} · generated {formatDateTime(latest.created_at)}{latest.created_by_role ? ` by ${latest.created_by_role}` : ''}</span>
                    </div>
                    <p className="text-sm leading-7 text-slate-700">{latest.risk_basis}. {EARLY_WARNING_STYLES[latest.risk_code]?.meaning}</p>
                    {latest.risk_disclaimer && <p className="text-xs leading-6 text-slate-500">{latest.risk_disclaimer}</p>}
                    <div>
                      <p className="mb-2 text-sm font-semibold text-slate-700">Inputs used for this prediction</p>
                      <InputsList inputs={latest.inputs} wide />
                    </div>
                    <div data-testid="report-shap">
                      <p className="mb-3 text-sm font-semibold text-slate-700">SHAP contributions (marks)</p>
                      <ShapContributionBars contributions={latest.shap_contributions} baseValue={latest.base_value} predictedMarks={latest.predicted_marks} />
                      <p className="mt-3 text-xs leading-6 text-slate-500">SHAP describes the model's attribution for this prediction, not cause and effect. The regression model provides no confidence score, so none is shown.</p>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="rounded-[18px] border border-emerald-100 bg-emerald-50/60 p-4">
                        <p className="flex items-center gap-2 text-sm font-semibold text-emerald-800"><ArrowUp size={14} aria-hidden="true" />Positive contributors</p>
                        <ul className="mt-2 space-y-1 text-sm text-emerald-800">
                          {positive.length === 0 && <li>No positive SHAP contributors above +0.5 marks</li>}
                          {positive.map((rec) => <li key={rec.id}>{rec.title} ({formatShapMarks(rec.shapContribution)})</li>)}
                        </ul>
                      </div>
                      <div className="rounded-[18px] border border-amber-100 bg-amber-50/60 p-4">
                        <p className="flex items-center gap-2 text-sm font-semibold text-amber-800"><ArrowDown size={14} aria-hidden="true" />Risk contributors</p>
                        <ul className="mt-2 space-y-1 text-sm text-amber-800">
                          {negative.length === 0 && <li>No negative SHAP contributors below -0.5 marks</li>}
                          {negative.map((rec) => <li key={rec.id}>{rec.title} ({formatShapMarks(rec.shapContribution)}, {rec.priority} priority)</li>)}
                        </ul>
                      </div>
                    </div>
                    <div data-testid="report-recommendations">
                      <p className="mb-3 text-sm font-semibold text-slate-700">Stored recommendations</p>
                      {recommendations.length === 0
                        ? <p className="text-sm text-slate-600">No specific intervention recommendations were generated for this prediction.</p>
                        : <RecommendationGroups recommendations={recommendations} strongest={findStrongestNegative(recommendations)} />}
                    </div>
                  </div>
                ) : (
                  <EmptyState
                    message="No prediction has been generated for this student yet."
                    note={detail.prediction_status.message}
                    action={<button type="button" onClick={() => openAnalysis(detail.student.id)} className={BTN_SECONDARY}>Open Student AI Analysis <ArrowRight size={14} aria-hidden="true" /></button>}
                  />
                ))}

                <div id="report-history" className="scroll-mt-6 break-inside-avoid" data-report="report-history">
                  <PredictionHistory studentId={detail.student.id} version={`${latest?.id ?? 'none'}|${MODEL_INPUT_FIELDS.map(({ key }) => detail.student.record[key]).join(',')}`} />
                </div>

                {reportSection('report-intervention', 'Intervention Report', interventionCountsText(interventionCounts(studentInterventions)), !interventionsResource.data ? (
                  <ResourceState resource={interventionsResource} what="interventions" />
                ) : studentInterventions.length === 0 ? (
                  <EmptyState
                    message="No interventions have been assigned to this student yet."
                    action={<button type="button" onClick={() => setActiveView('management')} className={BTN_SECONDARY}>Open Intervention Management <ArrowRight size={14} aria-hidden="true" /></button>}
                  />
                ) : (
                  <ul className="space-y-3" data-testid="report-interventions">
                    {studentInterventions.map((item) => <InterventionItem key={item.id} intervention={item} />)}
                  </ul>
                ))}

                {reportSection('report-department', 'Department Reports', null, (
                  <EmptyState message="Department-level reports require department records, which this project does not store." />
                ))}
              </>
            )}
          </div>
        )
      }
      case 'profile':
        return (
          <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
            <GlassCard className="p-6">
              <p className="text-sm font-semibold uppercase tracking-[0.25em] text-blue-600">Faculty account overview</p>
              <div className="mt-6 rounded-[24px] border border-slate-200 bg-slate-50 p-5">
                <div className="flex items-center gap-3">
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 text-white" aria-hidden="true"><CircleUserRound size={26} /></div>
                  <div>
                    <p className="font-semibold text-slate-900">{user.username}</p>
                    <p className="text-sm capitalize text-slate-500">{user.role} account</p>
                  </div>
                </div>
              </div>
            </GlassCard>

            <GlassCard className="p-6">
              <p className="text-sm font-semibold uppercase tracking-[0.25em] text-blue-600">Faculty focus areas</p>
              <div className="mt-5 space-y-3">
                {['Monitor students with High Concern early-warning status', 'Review intervention outcomes weekly', 'Prepare department performance notes', 'Share AI insights with academic council'].map((item) => (
                  <div key={item} className="rounded-[18px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-700">{item}</div>
                ))}
              </div>
            </GlassCard>
          </div>
        )
      default: {
        // Students from the database (GET /students) and interventions (GET /interventions)
        if (!listResource.data) return <section className={CARD}><ResourceState resource={listResource} what="student list" /></section>
        const studentsWithInterventions = realStudents
          .map((student) => ({ student, counts: interventionCounts(interventionsFor(student.id)) }))
          .filter(({ counts }) => counts.total > 0)
        return (
          <div className="space-y-6">
            <section aria-label="Student account and data actions" className="flex flex-wrap gap-3">
              <button type="button" onClick={() => setActiveView('create-student')} className={BTN_PRIMARY}>Create Student</button>
              <button type="button" onClick={() => setActiveView('bulk-registration')} className={BTN_SECONDARY}>Bulk Student Registration</button>
              <button type="button" onClick={() => setActiveView('bulk-upload')} className={BTN_SECONDARY}>Bulk Academic Upload</button>
            </section>
            <section aria-label="Faculty overview statistics" data-testid="dashboard-statistics">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {[
                  { label: 'Students', value: String(realStudents.length), detail: `${realCurrent.length} with a current prediction`, icon: Users, iconTone: 'bg-blue-50 text-blue-700' },
                  { label: 'Needing attention', value: String(realAlerts.length), detail: `${countRealBand('high_concern')} High Concern · ${countRealBand('borderline')} Borderline`, icon: CircleAlert, iconTone: 'bg-rose-50 text-rose-700', tone: realAlerts.length ? 'text-rose-700' : null },
                  { label: 'Interventions', value: interventionsResource.data ? String(totals.total) : '—', detail: interventionsResource.data ? `${totals.open} open${totals.overdue ? ` · ${totals.overdue} overdue` : ''}` : interventionsResource.isLoading ? 'Loading intervention data…' : 'Intervention data unavailable', icon: ClipboardList, iconTone: 'bg-amber-50 text-amber-800', action: () => setActiveView('management') },
                  { label: 'Completed interventions', value: interventionsResource.data ? String(totals.completed) : '—', detail: interventionsResource.data ? (totals.total ? `${Math.round((totals.completed / totals.total) * 100)}% of assigned` : 'None assigned yet') : interventionsResource.isLoading ? 'Loading intervention data…' : 'Intervention data unavailable', icon: CheckCircle2, iconTone: 'bg-emerald-50 text-emerald-800', tone: totals.completed ? 'text-emerald-700' : null },
                ].map((metric) => (
                  <article key={metric.label} className="ui-card flex min-w-0 flex-col p-4 sm:p-5">
                    <div className="flex items-start justify-between gap-3">
                      <h2 className="max-w-[12rem] text-sm font-semibold leading-5 text-slate-600">{metric.label}</h2>
                      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${metric.iconTone}`} aria-hidden="true">
                        <metric.icon size={18} />
                      </span>
                    </div>
                    <p className={`mt-3 text-3xl font-semibold tabular-nums tracking-tight ${metric.tone || 'text-slate-900'}`}>{metric.value}</p>
                    <p className="mt-1 min-h-10 text-sm leading-5 text-slate-500">{metric.detail}</p>
                    {metric.action && (
                      <button type="button" onClick={metric.action} className={`${linkButton} -ml-3 mt-2 self-start`}>
                        Open Intervention Management <ArrowRight size={14} aria-hidden="true" />
                      </button>
                    )}
                  </article>
                ))}
              </div>
            </section>

            <RiskDistribution
              counts={riskCounts}
              currentTotal={realCurrent.length}
              studentTotal={realStudents.length}
              testId="dashboard-risk-distribution"
              emptyAction={realStudents.length > 0
                ? <button type="button" onClick={() => setActiveView('analysis')} className={BTN_SECONDARY}>Open Student AI Analysis <ArrowRight size={14} aria-hidden="true" /></button>
                : null}
            />

            <section className={CARD} data-testid="real-needs-attention">
              <SectionHeading title="Students needing attention" description="Current High Concern and Borderline predictions, ordered from lowest predicted marks first. Risk is identified by category text as well as color." />
              <div className="space-y-3">
                {realStudents.length === 0 && <EmptyState message="No students are available in the database yet." />}
                {realStudents.length > 0 && realCurrent.length === 0 && <EmptyState message="No student has a current prediction yet." />}
                {realCurrent.length > 0 && realAlerts.length === 0 && <EmptyState message="No student with a current prediction is in the High Concern or Borderline band." />}
                {realAlerts.map((student) => renderRealStudentRow(student))}
              </div>
            </section>

            <section className={CARD} data-testid="dashboard-assignments">
              <SectionHeading title="Interventions by student" description="Interventions assigned to each student and how many they have completed." />
              {!interventionsResource.data ? (
                <ResourceState resource={interventionsResource} what="interventions" />
              ) : studentsWithInterventions.length === 0 ? (
                <EmptyState
                  message="No interventions have been assigned yet."
                  note="Assign them from a student's stored recommendations in Intervention Management or Student AI Analysis."
                  action={<button type="button" onClick={() => setActiveView('management')} className={BTN_SECONDARY}>Open Intervention Management <ArrowRight size={14} aria-hidden="true" /></button>}
                />
              ) : (
                <ul className="space-y-3">
                  {studentsWithInterventions.map(({ student, counts }) => (
                    <li key={student.id} className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0">
                        <p className="break-words font-semibold text-slate-900">{student.name}</p>
                        <p className="mt-0.5 break-all text-sm text-slate-500">{student.roll_no}</p>
                        <p className="mt-2 text-sm text-slate-600">{interventionCountsText(counts)}</p>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 sm:justify-end">
                        <span className="ui-badge border-blue-200 bg-blue-50 text-blue-800">{counts.total} assigned</span>
                        <span className="ui-badge border-amber-200 bg-amber-50 text-amber-800">{counts.open} open</span>
                        <span className="ui-badge border-emerald-200 bg-emerald-50 text-emerald-800">{counts.completed} completed</span>
                        {counts.overdue > 0 && <span className="ui-badge border-rose-200 bg-rose-50 text-rose-800">{counts.overdue} overdue</span>}
                        <button type="button" onClick={() => openStudentView(student.id, 'management')} className="ui-btn ui-btn-secondary w-full sm:ml-2 sm:w-auto">
                          Manage interventions <ArrowRight size={14} aria-hidden="true" />
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className={CARD} data-testid="real-students">
              <SectionHeading title="All Students" description="Current predictions first (lowest predicted marks first), then students without a current prediction." />
              <div className="space-y-3">
                {realStudents.length === 0 && <EmptyState message="No students found in the student database." />}
                {realCurrent.map((student) => renderRealStudentRow(student))}
                {realWithoutCurrent.map((student) => renderRealStudentRow(student, true))}
              </div>
            </section>
          </div>
        )
      }
    }
  }

  return (
    <div className="app-shell">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="w-full border-b border-slate-200 bg-white/90 p-4 shadow-sm backdrop-blur lg:sticky lg:top-0 lg:h-screen lg:w-72 lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r lg:p-6 print:hidden">
          <PortalBrand icon={BrainCircuit} title="Teacher Portal" subtitle="Faculty workspace" />
          <nav aria-label="Teacher Portal" className="mt-5 flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
            {FACULTY_VIEWS.map((item) => {
              const Icon = item.icon
              const isActive = activeView === item.id
              return (
                <button key={item.id} type="button" aria-current={isActive ? 'page' : undefined} onClick={() => setActiveView(item.id)} className={`flex shrink-0 items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm font-medium transition ${isActive ? 'bg-blue-600 text-white shadow-lg shadow-blue-200' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}`}>
                  <Icon size={18} aria-hidden="true" />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </nav>
          <PortalAccount />
        </aside>

        <main className="min-w-0 flex-1 p-4 sm:p-6 lg:p-8">
          <div className="mx-auto max-w-6xl">
            {apiStatus === 'error' && (
              <div className="mb-6">
                <ServiceErrorNotice message="The Flask prediction service isn't currently running. Start the backend and try again. No sample or fallback results are shown." onRetry={retryStatus} />
              </div>
            )}
            {apiStatus === null && <p className="mb-6 inline-flex items-center gap-2 text-sm text-slate-500"><LoaderCircle size={14} className="animate-spin" aria-hidden="true" />Checking prediction service…</p>}

            <PageHeader
              breadcrumbs={breadcrumbs}
              back={activeView === 'dashboard' ? null : { onClick: () => setActiveView('dashboard'), label: 'Back to Faculty Dashboard' }}
              eyebrow={activeView === 'analysis' ? 'Student AI Analysis' : 'Teacher Portal'}
              title={headers[activeView].title}
              description={headers[activeView].description}
            />

            {renderView()}
          </div>
        </main>
      </div>
    </div>
  )
}

function AdminDashboard() {
  // Reads GET / from the backend; null while checking, 'error' if unreachable.
  const [apiStatus, setApiStatus] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE_URL}/`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data) => { if (!cancelled) setApiStatus(data) })
      .catch(() => { if (!cancelled) setApiStatus('error') })
    return () => { cancelled = true }
  }, [])

  let statusText = 'System status: checking…'
  let statusClass = 'bg-slate-100 text-slate-600'
  if (apiStatus === 'error') {
    statusText = 'System status: API unreachable'
    statusClass = 'bg-rose-50 text-rose-700'
  } else if (apiStatus) {
    const ready = apiStatus.model_loaded && apiStatus.shap_loaded
    statusText = `System status: Model ${apiStatus.model_loaded ? 'loaded' : 'not loaded'} · SHAP ${apiStatus.shap_loaded ? 'loaded' : 'not loaded'}`
    statusClass = ready ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
  }

  return (
    <div className="app-shell p-4 sm:p-6 lg:p-8">
      <div className="mx-auto max-w-6xl">
      <PageHeader
        breadcrumbs={[{ label: 'Home', to: '/' }, { label: 'Dashboards', to: '/dashboard-selection' }, { label: 'Administrator' }]}
        back={{ to: '/dashboard-selection', label: 'Back to dashboard selection' }}
        eyebrow="Administrator dashboard"
        title="Model information and system overview"
        actions={<div role="status" aria-live="polite" className={`ui-badge max-w-full whitespace-normal ${statusClass}`}>{statusText}</div>}
      />

      <GlassCard className="p-5">
        <h2 className="ui-section-title">Model performance</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-3">
          {adminModelMetrics.map((metric) => (
            <div key={metric.label} className="rounded-[18px] border border-slate-200 bg-slate-50 p-5">
              <p className="text-sm font-medium text-slate-500">{metric.label}</p>
              <p className="mt-3 text-3xl font-semibold text-slate-900">{metric.value}</p>
              <p className="mt-2 text-sm text-slate-500">{metric.detail}</p>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm text-slate-500">
          Mean ± standard deviation across 5-fold cross-validation. MAE and RMSE are measured in final-exam marks out of 100.
        </p>
      </GlassCard>
      </div>
    </div>
  )
}

function PredictionPage() {
  // The page starts from the selected Demo Student's latest successful prediction
  // (its submitted inputs and API responses), or from their predefined inputs
  const { selectedId, results, beginRequest, recordResult } = useDemoStudents()
  const demoStudent = DEMO_STUDENTS.find((student) => student.id === selectedId)
  const saved = results[selectedId]
  const startInputs = saved?.inputs ?? demoStudent.inputs

  const [attendance, setAttendance] = useState(startInputs.attendance)
  // Internal Test 1 (0-40)
  const [internalMarks, setInternalMarks] = useState(startInputs.test1)
  // Internal Test 2 (0-40)
  const [test2, setTest2] = useState(startInputs.test2)
  // Assignment score (0-10)
  const [assignments, setAssignments] = useState(startInputs.assignment)
  const [studyHours, setStudyHours] = useState(startInputs.study_hours)
  const [analyzed, setAnalyzed] = useState(Boolean(saved))
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState('')
  const [predictedMarks, setPredictedMarks] = useState(saved?.predict.predicted_marks ?? null)
  const [shapContributions, setShapContributions] = useState(saved?.explain.shap_contributions ?? null)
  const [shapBaseValue, setShapBaseValue] = useState(saved?.explain.base_value ?? null)
  const [apiRecommendations, setApiRecommendations] = useState(saved?.recommend.recommendations ?? [])
  // Early-warning classification returned by /predict
  const [earlyWarning, setEarlyWarning] = useState(saved?.predict.risk_code ? saved.predict : null)

  // SHAP contributor lists come from /recommend, so they use the backend's
  // thresholds (above +0.5 / below -0.5 marks), order and feature names.
  const strengthCards = apiRecommendations
    .filter((item) => item.priority === 'POSITIVE')
    .map((item) => `${item.feature} (${formatShapMarks(item.shap_contribution)})`)

  if (strengthCards.length === 0) {
    strengthCards.push('No positive SHAP contributors above +0.5 marks')
  }

  const improvementCards = apiRecommendations
    .filter((item) => item.priority === 'HIGH' || item.priority === 'MEDIUM')
    .map((item) => `${item.feature} (${formatShapMarks(item.shap_contribution)}, ${item.priority} priority)`)

  if (improvementCards.length === 0) {
    improvementCards.push('No negative SHAP contributors below -0.5 marks')
  }

  const noRecommendationsMessage = 'No specific intervention recommendations were generated for this prediction.'

  const confidenceBullets = [
    'Validation MAE: 3.62 ± 0.07',
    'Validation RMSE: 4.59 ± 0.09',
    'Validation R²: 0.8357 ± 0.0084',
    'Feature contributions shown via SHAP explanations (marks)',
  ]

  // Prepare recommendations from API response (no fabricated fallback)
  const recommendationsToRender = mapRecommendations(apiRecommendations)

  const strongestNegative = findStrongestNegative(recommendationsToRender)

  // Handler to call backend APIs
  const handlePredict = async () => {
    setApiError('')

    // Map frontend inputs to backend expected fields
    // Use the raw user inputs directly (they now correspond to the model's features)
    const rawAttendance = attendance
    const rawTest1 = internalMarks
    const rawTest2 = test2
    const rawAssignments = assignments
    const rawStudyHours = studyHours

    if (rawAttendance === '' || rawTest1 === '' || rawTest2 === '' || rawAssignments === '' || rawStudyHours === '') {
      return setApiError('Please fill in all numeric fields before predicting.')
    }

    const test1 = Number(rawTest1)
    const test2Val = Number(rawTest2)
    const assignmentScore = Number(rawAssignments)
    const study_hours = Number(rawStudyHours)
    const att = Number(rawAttendance)

    // Frontend validation
    // Ranges match the model's training data and the backend validation
    if (!Number.isFinite(att) || att < 52 || att > 100) return setApiError('Attendance must be between 52 and 100.')
    if (!Number.isFinite(test1) || test1 < 18 || test1 > 40) return setApiError('Internal Test 1 must be between 18 and 40.')
    if (!Number.isFinite(test2Val) || test2Val < 16 || test2Val > 40) return setApiError('Internal Test 2 must be between 16 and 40.')
    if (!Number.isFinite(assignmentScore) || assignmentScore < 4 || assignmentScore > 10) return setApiError('Assignment Score must be between 4 and 10.')
    if (!Number.isFinite(study_hours) || study_hours < 1 || study_hours > 5) return setApiError('Daily Study Hours must be between 1 and 5.')

    const payload = {
      attendance: att,
      test1,
      test2: test2Val,
      assignment: assignmentScore,
      study_hours,
    }

    // The result belongs to the Demo Student selected when the prediction was submitted
    const requestToken = beginRequest(selectedId)

    setIsLoading(true)
    try {
      const { predict: predictJson, explain: explainJson, recommend: recommendJson } = await fetchLivePrediction(payload)

      setPredictedMarks(predictJson.predicted_marks ?? null)
      setEarlyWarning(predictJson.risk_code ? predictJson : null)
      setShapContributions(explainJson.shap_contributions ?? null)
      setShapBaseValue(explainJson.base_value ?? null)
      setApiRecommendations(recommendJson.recommendations ?? [])
      setAnalyzed(true)
      // Replaces this Demo Student's previous result; a failed request keeps the last successful one
      recordResult(selectedId, requestToken, { inputs: payload, predict: predictJson, explain: explainJson, recommend: recommendJson })
    } catch (err) {
      if (err.isServerResponse) {
        setApiError(err.message)
      } else {
        // Friendly message for connectivity issues
        setApiError(PREDICTION_CONNECTION_ERROR)
      }
      setPredictedMarks(null)
      setEarlyWarning(null)
      setShapContributions(null)
      setShapBaseValue(null)
      setApiRecommendations([])
    } finally {
      setIsLoading(false)
    }
  }

  // Same labels as the recommendation cards on the page
  const guidanceLabel = (rec) => (rec.needsAttention ? 'Recommended intervention' : 'Maintenance guidance')

  const handleDownloadCSV = () => {
    try {
      const csvRow = (values) => values.map((value) => `"${String(value ?? '').replace(/"/g, '""')}"`).join(',')
      const headers = ['Attendance', 'Internal Test 1', 'Internal Test 2', 'Assignment Score', 'Study Hours', 'Predicted Marks', 'Early-Warning Status', 'Risk Basis', 'Early-Warning Disclaimer']
      const row = [attendance ?? '', internalMarks ?? '', test2 ?? '', assignments ?? '', studyHours ?? '', predictedMarks ?? '', earlyWarning?.risk_level ?? '', earlyWarning?.risk_basis ?? '', earlyWarning?.risk_disclaimer ?? '']

      // Additional sections follow the original summary row, separated by blank lines
      const shapRows = shapContributions && Object.keys(shapContributions).length
        ? Object.entries(shapContributions).map(([name, val]) => csvRow([name, Number(val).toFixed(4)]))
        : [csvRow(['Not available', ''])]
      const recommendationRows = recommendationsToRender.length
        ? recommendationsToRender.map((rec) => csvRow([
          rec.priority, rec.title, rec.currentValue, rec.performanceLevel,
          Number.isFinite(Number(rec.shapContribution)) ? Number(rec.shapContribution).toFixed(4) : '',
          rec.description, guidanceLabel(rec), rec.action,
        ]))
        : [csvRow([noRecommendationsMessage])]

      const csvContent = [
        headers.join(','),
        csvRow(row),
        '',
        csvRow(['SHAP Feature Contributions']),
        csvRow(['Feature', 'SHAP Contribution (marks)']),
        ...shapRows,
        '',
        csvRow(['Recommendations']),
        csvRow(['Priority', 'Feature', 'Current Value', 'Performance Level', 'SHAP Contribution (marks)', 'Explanation', 'Guidance Type', 'Recommended Action']),
        ...recommendationRows,
      ].join('\n')

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'student-performance-report.csv'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err)
      alert('Unable to generate CSV export.')
    }
  }

  const handleShareReport = async () => {
    const shareText = [
      'AI-Driven Student Performance Prediction Report',
      `Predicted Marks: ${predictedMarks ?? 'Not available'}`,
      `Attendance: ${attendance ?? 'Not available'}`,
      `Internal Test 1: ${internalMarks ?? 'Not available'}`,
      `Internal Test 2: ${test2 ?? 'Not available'}`,
      `Assignment Score: ${assignments ?? 'Not available'}`,
      `Study Hours: ${studyHours ?? 'Not available'}`,
      `Early-Warning Status: ${earlyWarning?.risk_level ?? 'Not available'}`,
      `Risk Basis: ${earlyWarning?.risk_basis ?? 'Not available'}`,
      ...(earlyWarning?.risk_disclaimer ? [earlyWarning.risk_disclaimer] : []),
      '',
      'SHAP Feature Contributions:',
      ...(shapContributions && Object.keys(shapContributions).length
        ? Object.entries(shapContributions).map(([name, val]) => `- ${name}: ${formatShapMarks(val)}`)
        : ['Not available']),
      '',
      'Recommendations:',
      ...(recommendationsToRender.length
        ? recommendationsToRender.flatMap((rec) => [
          `- [${rec.priority}] ${rec.title} (current value ${rec.currentValue}, ${rec.performanceLevel}, SHAP ${formatShapMarks(rec.shapContribution)})`,
          `  ${rec.description}`,
          ...(rec.action && rec.action !== rec.description ? [`  ${guidanceLabel(rec)}: ${rec.action}`] : []),
        ])
        : [noRecommendationsMessage]),
    ].join('\n')

    try {
      if (navigator.share) {
        await navigator.share({
          title: 'AI-Driven Student Performance Prediction Report',
          text: shareText,
        })
        return
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(shareText)
        alert('Report summary copied to clipboard.')
        return
      }

      alert('Sharing is unavailable in this browser.')
    } catch (err) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        try {
          await navigator.clipboard.writeText(shareText)
          alert('Report summary copied to clipboard.')
          return
        } catch (clipboardErr) {
          console.error(clipboardErr)
        }
      }
      console.error(err)
      alert('Unable to share report.')
    }
  }

  const handleDownloadPDF = () => {
    try {
      const doc = new jsPDF()
      const title = 'AI-Driven Student Performance Prediction Report'
      let y = 16
      // Writes wrapped text, starting a new page first if the block would not fit
      const write = (content, size = 11, lineHeight = 7, style = 'normal') => {
        doc.setFont('helvetica', style)
        doc.setFontSize(size)
        const lines = doc.splitTextToSize(String(content), 180)
        if (y + lines.length * lineHeight > 285) { doc.addPage(); y = 16 }
        doc.text(lines, 14, y)
        y += lines.length * lineHeight
      }

      write(title, 16, 8)

      write(`Attendance: ${attendance}`, 12)
      write(`Internal Test 1: ${internalMarks}`, 12)
      write(`Internal Test 2: ${test2}`, 12)
      write(`Assignment Score: ${assignments}`, 12)
      write(`Study Hours: ${studyHours}`, 12)
      y += 2
      write(`Predicted Marks: ${predictedMarks ?? 'Not available'}`, 12)
      write(`Early-Warning Status: ${earlyWarning?.risk_level ?? 'Not available'}`, 12)
      write(`Risk Basis: ${earlyWarning?.risk_basis ?? 'Not available'}`, 12)
      if (earlyWarning?.risk_disclaimer) write(earlyWarning.risk_disclaimer, 9, 5)
      y += 5

      write('SHAP Feature Contributions:', 13)
      if (shapContributions && Object.keys(shapContributions).length) {
        Object.entries(shapContributions).forEach(([name, val]) => write(`${name}: ${formatShapMarks(val)}`))
      } else {
        write('Not available')
        y += 2
      }

      write('Positive SHAP contributors:', 13)
      if (Array.isArray(strengthCards) && strengthCards.length) {
        strengthCards.forEach((s) => write(s))
      } else {
        write('Not available')
        y += 2
      }

      write('Negative SHAP contributors:', 13)
      if (Array.isArray(improvementCards) && improvementCards.length) {
        improvementCards.forEach((it) => write(it))
      } else {
        write('Not available')
        y += 2
      }

      write('Personalized AI recommendations:', 13)
      if (Array.isArray(recommendationsToRender) && recommendationsToRender.length) {
        recommendationsToRender.forEach((r) => {
          write(`${r.priority || ''} ${r.title}`.trim(), 11, 6, 'bold')
          if (r.description) write(r.description, 10.5, 5.5)
          if (r.action && r.action !== r.description) write(`${guidanceLabel(r)}: ${r.action}`, 10.5, 5.5)
          const details = [
            r.currentValue != null ? `Current value: ${r.currentValue}` : null,
            r.performanceLevel ? `Performance level: ${r.performanceLevel}` : null,
            Number.isFinite(Number(r.shapContribution)) && r.shapContribution != null ? `SHAP contribution: ${formatShapMarks(r.shapContribution)}` : null,
          ].filter(Boolean)
          if (details.length) write(details.join('   |   '), 10, 5.5)
          y += 4
        })
      } else {
        write(noRecommendationsMessage)
        y += 2
      }

      doc.save('student-performance-report.pdf')
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err)
      // Informative fallback for missing dependency
      alert('Unable to generate PDF. If jspdf is not installed run: npm install jspdf')
    }
  }

  // Same ids, ranges and handlers as before; shown in the model's feature order
  const inputFields = [
    { id: 'prediction-attendance', label: 'Attendance (%)', min: 52, max: 100, value: attendance, set: setAttendance },
    { id: 'prediction-internal-test-1', label: 'Internal Test 1 (out of 40)', min: 18, max: 40, value: internalMarks, set: setInternalMarks },
    { id: 'prediction-internal-test-2', label: 'Internal Test 2 (out of 40)', min: 16, max: 40, step: '0.1', value: test2, set: setTest2 },
    { id: 'prediction-assignment-score', label: 'Assignment Score (out of 10)', min: 4, max: 10, step: 0.1, value: assignments, set: setAssignments },
    { id: 'prediction-daily-study-hours', label: 'Daily Study Hours', min: 1, max: 5, step: '0.1', value: studyHours, set: setStudyHours },
  ]
  const isConnectionError = apiError === PREDICTION_CONNECTION_ERROR
  const exportActions = [
    ['Download PDF', handleDownloadPDF],
    ['Export CSV', handleDownloadCSV],
    ['Share Report', handleShareReport],
    ['Print Report', () => window.print()],
  ]

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Student Portal', to: '/student' }, { label: 'Prediction' }]}
        back={{ to: '/student', label: 'Back to Student Dashboard' }}
        eyebrow="AI Prediction Analysis"
        title="Explainable AI prediction report"
        description="This page explains the model's prediction and the input factors that contributed to it, using SHAP."
      />

      <section className={CARD}>
        <SectionHeading step="1" title="Practice scenario" description="Try the model with any values. Your stored academic record and its prediction are shown on your Dashboard." />
        <p className="rounded-2xl bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600">
          {saved
            ? <>Showing your latest practice result ({saved.predict.predicted_marks} marks · {saved.predict.risk_level}). The form shows the inputs of that result.</>
            : <>No practice result yet. The form starts from sample inputs.</>}
          {' '}Each new practice prediction replaces the previous one and is not saved.
        </p>
      </section>

      <section className={`${CARD} mt-6`}>
        <SectionHeading step="2" title="Enter academic inputs" description="The five model inputs, within the ranges the model was trained on." />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {inputFields.map(({ id, label, min, max, step, value, set }) => (
            <div key={id} className="rounded-[18px] border border-slate-200 bg-white p-4">
              <label htmlFor={id} className="block text-sm font-medium text-slate-700 xl:min-h-[2.5rem]">{label}</label>
              <input
                id={id}
                type="number"
                min={min}
                max={max}
                step={step}
                value={value}
                onChange={(event) => {
                  const nextValue = event.target.value
                  set(nextValue === '' ? '' : Number(nextValue))
                }}
                className={FIELD_INPUT}
              />
              <p className="mt-1 text-xs text-slate-500">Range {min}–{max}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 flex flex-col gap-4 border-t border-slate-100 pt-6 sm:flex-row sm:items-center">
          <div className="flex items-center gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm font-semibold text-white" aria-hidden="true">3</span>
            <button type="button" onClick={handlePredict} disabled={isLoading} aria-busy={isLoading} className={`${BTN_PRIMARY} px-6 py-3 text-base`}>
              {isLoading
                ? <><LoaderCircle size={18} className="animate-spin" aria-hidden="true" /> Predicting...</>
                : <><Sparkles size={18} aria-hidden="true" /> Predict Performance</>}
            </button>
          </div>
          <p className="text-sm leading-6 text-slate-500">Runs the live model: prediction, SHAP explanation and recommendations.</p>
        </div>
        {apiError && (
          <div className="mt-4">
            {isConnectionError
              ? <ServiceErrorNotice onRetry={handlePredict} retrying={isLoading} />
              : (
                <div role="alert" className="rounded-[20px] border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
                  <p className="font-semibold">Could not generate a prediction</p>
                  <p className="mt-1 leading-6">{apiError}</p>
                </div>
              )}
          </div>
        )}
      </section>

      <div className="mt-8" aria-live="polite" aria-busy={isLoading}>
        {/* A failed request clears the page's result values, so no empty result card is shown */}
        {!analyzed || predictedMarks === null ? (
          isLoading ? (
            <div className={`${CARD} flex items-center gap-4`}>
              <LoaderCircle size={24} className="shrink-0 animate-spin text-blue-600" aria-hidden="true" />
              <div>
                <p className="font-semibold text-slate-900">Generating prediction…</p>
                <p className="text-sm text-slate-600">Running the prediction, SHAP explanation and recommendations on the live model.</p>
              </div>
            </div>
          ) : (
            <div className="rounded-[24px] border border-slate-200 bg-gradient-to-br from-slate-950 via-blue-950 to-cyan-900 p-6 text-white shadow-lg">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-white/15 p-3"><ShieldCheck size={18} aria-hidden="true" /></div>
                <div>
                  <p className="text-sm font-semibold text-cyan-300">Analysis preview</p>
                  <h3 className="text-2xl font-semibold">Your report will appear here</h3>
                </div>
              </div>
              <p className="mt-4 text-sm leading-7 text-slate-300">Enter the five model inputs to generate a prediction, early-warning status, SHAP explanation and academic recommendations.</p>
            </div>
          )
        ) : (
          <div className={`space-y-6 transition-opacity ${isLoading ? 'opacity-60' : ''}`}>
            {/* Result summary: the headline numbers first, then where to go next */}
            <section className="ui-card p-5 sm:p-7" aria-label="Practice prediction result">
              <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(20rem,0.9fr)]">
                <div className="rounded-[20px] border border-blue-200 bg-blue-50/60 p-5 sm:p-6">
                  <p className="text-xs font-bold uppercase tracking-[0.16em] text-blue-800">Practice prediction result</p>
                  <p className="mt-2 text-5xl font-bold tracking-tight text-slate-950 sm:text-6xl">
                    {predictedMarks !== null ? predictedMarks : '—'}
                    <span className="ml-1 text-2xl font-semibold text-slate-600 sm:text-3xl">/ 100</span>
                  </p>
                  <p className="mt-2 text-sm font-semibold text-slate-800">Predicted final-exam marks — not actual examination marks.</p>
                  <div className="mt-4 flex flex-wrap items-center gap-3">
                    <EarlyWarningBadge risk={earlyWarning} />
                    <span className="text-sm text-slate-700">{earlyWarning?.risk_basis ?? 'Risk basis not available'}</span>
                  </div>
                  <p className="mt-4 text-sm leading-6 text-slate-700">{EARLY_WARNING_STYLES[earlyWarning?.risk_code]?.meaning ?? 'Early-warning interpretation is not available.'}</p>
                  {earlyWarning?.risk_disclaimer && <p className="mt-2 text-xs leading-5 text-slate-600">{earlyWarning.risk_disclaimer}</p>}
                </div>
                <dl className="grid content-start gap-3 sm:grid-cols-2">
                  {[
                    ['Model', 'Final Student Performance Regression Model'],
                    ['Prediction Scope', 'Practice scenario (browser session only, not stored)'],
                    ['Validation MAE', '3.62'],
                    ['Validation RMSE', '4.59'],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-[16px] border border-slate-200 bg-slate-50 px-3 py-3">
                      <dt className="text-xs font-medium text-slate-500">{label}</dt>
                      <dd className="mt-1 text-sm font-semibold text-slate-900">{value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              <div className="mt-5 flex flex-wrap gap-3">
                <NavLink to="/what-if" className={`${BTN_PRIMARY} no-underline`}>Explore What-If <ArrowRightLeft size={16} aria-hidden="true" /></NavLink>
                <a href="#recommended-actions" className={`${BTN_SECONDARY} no-underline`}>Jump to Recommended Actions <ArrowDown size={16} aria-hidden="true" /></a>
              </div>
            </section>

            <section className={CARD}>
              <SectionHeading step="4" title="What contributed to this prediction?" description="The SHAP baseline is the starting model output. Each contribution moves that output up or down in marks; the values below are the ones returned by the explanation API." />
              {shapContributions && Object.keys(shapContributions).length
                ? <ShapContributionBars contributions={shapContributions} baseValue={shapBaseValue} predictedMarks={predictedMarks} />
                : <p className="text-sm text-slate-600">SHAP contributions are not available.</p>}
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <div className="rounded-[18px] border border-emerald-100 bg-emerald-50/60 p-4">
                  <p className="flex items-center gap-2 text-sm font-semibold text-emerald-800"><ArrowUp size={14} aria-hidden="true" />Positive SHAP contributors</p>
                  <p className="mt-1 text-xs text-emerald-800/80">Features that raised this prediction by more than 0.5 marks.</p>
                  <ul className="mt-3 space-y-1.5 text-sm font-medium text-emerald-800">
                    {strengthCards.map((item) => <li key={item}>{item}</li>)}
                  </ul>
                </div>
                <div className="rounded-[18px] border border-amber-100 bg-amber-50/60 p-4">
                  <p className="flex items-center gap-2 text-sm font-semibold text-amber-800"><ArrowDown size={14} aria-hidden="true" />Negative SHAP contributors</p>
                  <p className="mt-1 text-xs text-amber-800/80">Features that lowered this prediction by more than 0.5 marks.</p>
                  <ul className="mt-3 space-y-1.5 text-sm font-medium text-amber-800">
                    {improvementCards.map((item) => <li key={item}>{item}</li>)}
                  </ul>
                </div>
              </div>
              <p className="mt-4 text-xs leading-6 text-slate-500">Feature contributions shown above are in marks as returned by the explanation API. The early-warning status is based only on the predicted final-exam marks; SHAP describes the model's attribution for this prediction, not cause and effect.</p>
            </section>

            <section id="recommended-actions" className={`${CARD} scroll-mt-6`}>
              <SectionHeading step="5" title="Recommended Actions" description="Targeted academic guidance from the recommendation engine, grouped by priority." />
              <div className="mb-6 rounded-[18px] border border-blue-100 bg-blue-50 p-4 text-sm leading-7 text-slate-600">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-700">Overall early-warning status:</span>
                  <EarlyWarningBadge risk={earlyWarning} />
                </div>
                <p className="mt-2">Risk level indicates the student's overall early-warning concern and is based on the predicted final-exam marks. SHAP contributions show which input features influenced this individual prediction. These are different measures: the recommendations below follow the SHAP contributions, which do not set the risk level and describe the model's attribution rather than cause and effect.</p>
              </div>
              {recommendationsToRender.length === 0 && (
                <p className="rounded-[18px] border border-dashed border-slate-200 bg-slate-50 p-4 text-sm leading-7 text-slate-600">{noRecommendationsMessage}</p>
              )}
              <RecommendationGroups recommendations={recommendationsToRender} strongest={strongestNegative} />
              <p className="mt-6 text-sm leading-7 text-slate-600">The model predicts final exam marks based on the five inputs above. The recommendations above are based on the features that contributed most strongly to this prediction according to the SHAP explanation. They are intended as academic guidance and do not establish cause-and-effect relationships.</p>
            </section>

            <div className="grid gap-6 lg:grid-cols-2">
              <section className={CARD}>
                <SectionHeading step="6" title="Export the analysis" description="Download, share or print this prediction report." />
                <div className="flex flex-wrap gap-3">
                  {exportActions.map(([label, onClick]) => (
                    <button key={label} type="button" onClick={onClick} className={BTN_SECONDARY}>{label}</button>
                  ))}
                </div>
              </section>

              <section className={CARD}>
                <SectionHeading title="Model notes" description="Model validation metrics (regression), from 5-fold cross-validation." />
                <ul className="flex flex-wrap gap-2">
                  {confidenceBullets.map((item) => (
                    <li key={item} className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">{item}</li>
                  ))}
                </ul>
              </section>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

function WhatIfPage() {
  // The current scenario is the latest practice prediction's inputs, or the sample inputs (not the stored record yet)
  const { selectedId, results } = useDemoStudents()
  const demoStudent = DEMO_STUDENTS.find((student) => student.id === selectedId)
  const saved = results[selectedId]
  const baseline = saved?.inputs ?? demoStudent.inputs
  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Student Portal', to: '/student' }, { label: 'Prediction', to: '/prediction' }, { label: 'What-If Analysis' }]}
        back={{ to: '/prediction', label: 'Back to Prediction' }}
        eyebrow="What-if analysis"
        title="Interactive scenario simulator"
        description="Adjust the five model inputs within their training-supported ranges. Each scenario is scored by the same prediction model and early-warning classification used on the Prediction page."
      />
      <WhatIfSimulator key={selectedId} baseline={baseline} fromPrediction={Boolean(saved)} />
    </>
  )
}

function WhatIfSimulator({ baseline, fromPrediction }) {
  const [scenario, setScenario] = useState(baseline)
  const { result, error, isLoading, retry } = useScenarioPrediction(scenario)
  const { result: baseResult, error: baseError, isLoading: baseLoading, retry: retryBase } = useScenarioPrediction(baseline)

  const change = result && baseResult ? result.predicted_marks - baseResult.predicted_marks : null
  const changedFields = MODEL_INPUT_FIELDS.filter(({ key }) => scenario[key] !== baseline[key])
  const serviceError = error || baseError

  return (
    <div className="mt-6 grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
      <section className={CARD}>
        <SectionHeading title="What-If scenario inputs" description={`Practice only — starting point: ${fromPrediction ? 'your latest practice prediction inputs' : 'sample inputs'}. Changed inputs are marked; your stored academic record is unchanged.`} />
        <div className="space-y-4">
          {MODEL_INPUT_FIELDS.map(({ key, label, min, max, step }) => {
            const changed = scenario[key] !== baseline[key]
            return (
              <div key={key} className={`rounded-[18px] border p-4 transition ${changed ? 'border-blue-300 bg-blue-50/60' : 'border-slate-200 bg-slate-50'}`}>
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-sm font-medium text-slate-600">
                  <label htmlFor={`what-if-${key}`}>{label}</label>
                  <span className="flex items-center gap-2">
                    {changed && <span className="ui-badge border-blue-200 bg-white text-blue-800">Changed from {baseline[key]}</span>}
                    <output htmlFor={`what-if-${key}`} className="text-base font-semibold tabular-nums text-slate-900">{scenario[key]}</output>
                  </span>
                </div>
                <input id={`what-if-${key}`} type="range" min={min} max={max} step={step} value={scenario[key]} onChange={(event) => setScenario((prev) => ({ ...prev, [key]: Number(event.target.value) }))} aria-label={label} aria-valuetext={`${scenario[key]}; ${changed ? `changed from ${baseline[key]}` : 'same as starting value'}`} className="w-full accent-blue-600" />
                <div className="mt-1 flex justify-between text-xs text-slate-400">
                  <span>{min}</span>
                  <span>{max}</span>
                </div>
              </div>
            )
          })}
        </div>
        <p className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-950">All slider values are hypothetical practice scenarios. They are not actual academic results and are not saved to your student record.</p>
        <button type="button" onClick={() => setScenario(baseline)} disabled={changedFields.length === 0} className={`${BTN_SECONDARY} mt-4`}>
          <RotateCcw size={14} aria-hidden="true" /> Reset to current scenario
        </button>
      </section>

      <div className="space-y-6">
        {serviceError && <ServiceErrorNotice onRetry={() => { retry(); retryBase() }} />}

        <section className={CARD}>
          <SectionHeading title="Scenario comparison" />
          <div className="space-y-3">
            <div className="rounded-[18px] border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-800">Starting scenario · predicted marks</p>
              <p className="text-xs text-slate-500">Practice inputs · {fromPrediction ? 'latest practice prediction inputs' : 'sample inputs'}</p>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <p className="text-3xl font-bold tabular-nums text-slate-950">{baseResult ? `${baseResult.predicted_marks} / 100` : baseLoading ? 'Updating…' : '—'}</p>
                {baseResult && <EarlyWarningBadge risk={baseResult} />}
              </div>
            </div>

            <div className="rounded-[18px] border border-blue-200 bg-blue-50/60 p-4">
              <p className="text-sm font-semibold text-blue-900">What-If scenario · predicted marks</p>
              <p className="text-xs text-slate-500">{changedFields.length ? `Changed: ${changedFields.map(({ label }) => label).join(', ')}` : 'No inputs changed yet — move a slider to explore.'}</p>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <p className="text-3xl font-bold tabular-nums text-slate-950">{result ? `${result.predicted_marks} / 100` : isLoading ? 'Updating prediction…' : 'Prediction not available'}</p>
                {isLoading
                  ? <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-semibold text-slate-600"><LoaderCircle size={14} className="animate-spin" aria-hidden="true" />Updating…</span>
                  : result && <EarlyWarningBadge risk={result} />}
              </div>
              {result && <p className="mt-2 text-sm text-slate-600">{result.risk_basis}</p>}
            </div>

            <div className="rounded-[18px] border border-slate-200 bg-white p-4">
              <p className="text-sm font-semibold text-slate-500">Predicted change</p>
              <p className="mt-1 flex items-center gap-2 text-2xl font-semibold text-slate-900" role="status" aria-live="polite">
                {change !== null && change !== 0 && (change > 0 ? <ArrowUp size={20} className="text-emerald-600" aria-hidden="true" /> : <ArrowDown size={20} className="text-amber-600" aria-hidden="true" />)}
                {change !== null ? `${change >= 0 ? '+' : ''}${change.toFixed(2)} marks` : isLoading || baseLoading ? 'Updating…' : '—'}
              </p>
              <p className="mt-1 text-xs leading-5 text-slate-500">This is the difference between two model predictions. It does not represent an expected real-world improvement.</p>
            </div>
          </div>
          {result?.risk_disclaimer && <p className="mt-4 text-xs leading-6 text-slate-500">{result.risk_disclaimer}</p>}
        </section>

      </div>
    </div>
  )
}

function ReportsPage() {
  // Thresholds mirror classify_early_warning in the backend
  const earlyWarningCategories = [
    { code: 'high_concern', label: 'High Concern', rule: 'Predicted marks below 40' },
    { code: 'borderline', label: 'Borderline', rule: 'Predicted marks from 40 through 44' },
    { code: 'lower_concern', label: 'Lower Concern', rule: 'Predicted marks above 44' },
  ]

  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Student Portal', to: '/student' }, { label: 'Reports' }]}
        back={{ to: '/student', label: 'Back to Student Dashboard' }}
        eyebrow="Reports"
        title="Model & Prediction Report"
      />

        <GlassCard className="p-5 sm:p-6">
          <p className="ui-eyebrow">Model evaluation</p>
          <h3 className="text-lg font-semibold text-slate-900">5-Fold Cross-Validation</h3>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {adminModelMetrics.map((metric) => (
              <div key={metric.label} className="rounded-[18px] border border-slate-200 bg-slate-50 p-4 sm:p-5">
                <p className="text-sm font-medium text-slate-500">{metric.label}</p>
                <p className="mt-2 text-2xl font-semibold tabular-nums text-slate-900 sm:text-3xl">{metric.value}</p>
                <p className="mt-2 text-sm text-slate-500">{metric.detail}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm text-slate-500">
            Results are means ± standard deviations across 5 folds. MAE and RMSE are measured in final-exam marks out of 100.
          </p>
        </GlassCard>

        <div className="mt-6 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <GlassCard className="p-5 sm:p-6">
            <h2 className="ui-section-title">Model &amp; dataset facts</h2>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div className="rounded-[18px] border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm text-slate-500">Dataset</p>
                <p className="mt-1 text-lg font-semibold text-slate-900">2,000 records</p>
              </div>
              <div className="rounded-[18px] border border-slate-200 bg-slate-50 p-4">
                <p className="text-sm text-slate-500">Target</p>
                <p className="mt-1 text-lg font-semibold text-slate-900">Final Exam Marks (out of 100)</p>
              </div>
            </div>
            <p className="mt-5 text-sm font-semibold text-slate-700">Input features and supported ranges</p>
            <div className="mt-3 space-y-2">
              {MODEL_INPUT_FIELDS.map(({ key, label, min, max }) => (
                <div key={key} className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 rounded-[14px] border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm">
                  <span className="text-slate-700">{label}</span>
                  <span className="font-semibold text-slate-900">{min}–{max}</span>
                </div>
              ))}
            </div>
          </GlassCard>

          <div className="space-y-6">
            <GlassCard className="p-5 sm:p-6">
              <h2 className="ui-section-title">Early-warning categories</h2>
              <ul className="mt-4 space-y-3">
                {earlyWarningCategories.map(({ code, label, rule }) => {
                  const style = EARLY_WARNING_STYLES[code]
                  const Icon = style.icon
                  return (
                    <li key={code} className="flex flex-col items-start justify-between gap-2 rounded-[18px] border border-slate-200 bg-slate-50 px-4 py-3 sm:flex-row sm:items-center">
                      <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm font-semibold ${style.badge}`} aria-label={`Early-warning category: ${label}`}>
                        <Icon size={16} aria-hidden="true" />{label}
                      </span>
                      <span className="text-sm text-slate-600 sm:text-right">{rule}</span>
                    </li>
                  )
                })}
              </ul>
              <p className="mt-4 text-xs leading-5 text-slate-500">
                These are model-based early-warning categories derived from predicted final-exam marks. They are not an official university pass/fail determination.
              </p>
            </GlassCard>

            <GlassCard className="p-5 sm:p-6">
              <h2 className="ui-section-title">Your prediction report</h2>
              <p className="mt-3 text-sm leading-7 text-slate-600">Your stored prediction, SHAP explanation, recommendations and prediction history are on your Dashboard, generated from your academic record.</p>
              <NavLink to="/student" className={`${BTN_PRIMARY} mt-4`}>
                Go to Dashboard <ArrowRight size={14} aria-hidden="true" />
              </NavLink>
            </GlassCard>
          </div>
        </div>
    </>
  )
}

function ProfilePage() {
  const studentResource = useOutletContext()
  const { user } = useAuth()
  const student = studentResource.data?.student
  return (
    <>
      <PageHeader
        breadcrumbs={[{ label: 'Student Portal', to: '/student' }, { label: 'Profile' }]}
        back={{ to: '/student', label: 'Back to Student Dashboard' }}
        eyebrow="Profile"
        title="Student profile"
      />
      {!student ? (
        <section className={CARD}><ResourceState resource={studentResource} what="student profile" /></section>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <GlassCard className="flex flex-col items-center justify-center p-6 text-center">
            <div className="flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 text-white">
              <CircleUserRound size={44} aria-hidden="true" />
            </div>
            <h2 className="mt-6 text-2xl font-semibold text-slate-900">{student.name}</h2>
            <p className="mt-1 text-sm text-slate-500">{student.roll_no}</p>
          </GlassCard>

          <GlassCard className="p-6">
            <p className="text-sm font-semibold uppercase tracking-[0.25em] text-blue-600">Account</p>
            <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
              {[['Username', user.username], ['Roll number', student.roll_no], ['Program', student.program || '—'], ['Section', student.section || '—']].map(([label, value]) => (
                <div key={label} className="rounded-[16px] border border-slate-200 bg-slate-50 px-3 py-2">
                  <dt className="text-xs text-slate-500">{label}</dt>
                  <dd className="font-semibold text-slate-900">{value}</dd>
                </div>
              ))}
            </dl>
            <NavLink to="/student" className={`${BTN_PRIMARY} mt-6`}>
              View my academic record <ArrowRight size={14} aria-hidden="true" />
            </NavLink>
          </GlassCard>
        </div>
      )}
    </>
  )
}

function GlassCard({ children, className = '', ...rest }) {
  return (
    <div {...rest} className={`ui-card ${className}`}>{children}</div>
  )
}

export default App
