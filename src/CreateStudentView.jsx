import { useState } from 'react'
import { ArrowLeft, CircleAlert, LoaderCircle, UserPlus } from 'lucide-react'
import { apiRequest } from './api'

const CARD = 'ui-card p-5 sm:p-6'
const PRIMARY_BUTTON = 'ui-btn ui-btn-primary'
const SECONDARY_BUTTON = 'ui-btn ui-btn-secondary'
const FIELD_INPUT = 'ui-field mt-2'

function CreateStudentView({ onBack, onCreated }) {
  const [form, setForm] = useState({
    name: '',
    roll_no: '',
    username: '',
    password: '',
  })
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  const update = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }))
    setError('')
    setSuccess('')
  }

  const submit = async (event) => {
    event.preventDefault()
    setError('')
    setSuccess('')
    setIsSaving(true)
    try {
      const response = await apiRequest('/students/accounts', {
        method: 'POST',
        body: form,
      })
      setSuccess(
        `${response.student.name} (${response.student.roll_no}) was created. ` +
        `The student can sign in with username "${response.student.username}" and the password you set.`,
      )
      setForm({ name: '', roll_no: '', username: '', password: '' })
      await onCreated?.()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div className="space-y-5" data-testid="create-student-view">
      <section className={CARD}>
        <h2 className="text-lg font-semibold text-slate-900">Create Student</h2>
        <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-600">
          Create a student identity and login account. This does not add academic data,
          predictions, or interventions. Use the roll number as Student ID in Bulk Academic Upload.
        </p>
        <form className="mt-5 space-y-4" onSubmit={submit}>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-sm font-semibold text-slate-700" htmlFor="student-create-name">
              Student full name
              <input
                id="student-create-name"
                className={FIELD_INPUT}
                value={form.name}
                onChange={update('name')}
                maxLength={120}
                autoComplete="name"
                required
              />
            </label>
            <label className="text-sm font-semibold text-slate-700" htmlFor="student-create-roll">
              Student roll number / Student ID
              <input
                id="student-create-roll"
                className={FIELD_INPUT}
                value={form.roll_no}
                onChange={update('roll_no')}
                maxLength={80}
                required
              />
            </label>
            <label className="text-sm font-semibold text-slate-700" htmlFor="student-create-username">
              Login username
              <input
                id="student-create-username"
                className={FIELD_INPUT}
                value={form.username}
                onChange={update('username')}
                minLength={3}
                maxLength={80}
                pattern="[A-Za-z0-9._-]{3,80}"
                autoComplete="username"
                required
              />
              <span className="mt-1 block text-xs font-normal text-slate-500">
                3–80 letters, numbers, dots, underscores, or hyphens.
              </span>
            </label>
            <label className="text-sm font-semibold text-slate-700" htmlFor="student-create-password">
              Initial password
              <input
                id="student-create-password"
                type="password"
                className={FIELD_INPUT}
                value={form.password}
                onChange={update('password')}
                minLength={8}
                maxLength={256}
                autoComplete="new-password"
                required
              />
              <span className="mt-1 block text-xs font-normal text-slate-500">
                At least 8 characters. The app stores only a password hash.
              </span>
            </label>
          </div>

          {error && (
            <p role="alert" className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
              <CircleAlert size={17} className="mt-0.5 shrink-0" aria-hidden="true" />{error}
            </p>
          )}
          {success && (
            <p role="status" className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
              {success}
            </p>
          )}
          <div className="flex flex-wrap gap-3">
            <button type="submit" disabled={isSaving} aria-busy={isSaving} className={PRIMARY_BUTTON}>
              {isSaving
                ? <><LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> Creating…</>
                : <><UserPlus size={16} aria-hidden="true" /> Create Student</>}
            </button>
            <button type="button" onClick={onBack} disabled={isSaving} className={SECONDARY_BUTTON}>
              <ArrowLeft size={16} aria-hidden="true" /> Return to Faculty Dashboard
            </button>
          </div>
        </form>
      </section>
    </div>
  )
}

export default CreateStudentView
