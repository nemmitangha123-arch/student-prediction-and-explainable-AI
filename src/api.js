// Backend API base
export const API_BASE_URL = "http://127.0.0.1:5000"

export const SERVER_CONNECTION_ERROR = 'Unable to connect to the server. Please make sure the Flask backend is running.'

// The sign-in token lives in sessionStorage: it survives a page reload but
// ends with the browser tab, and is never shared with other sites.
const TOKEN_KEY = 'predictwise.authToken'

export function getStoredToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function storeToken(token) {
  try {
    sessionStorage.setItem(TOKEN_KEY, token)
  } catch {
    // Storage blocked: the session then lasts until the next reload only
  }
}

export function clearStoredToken() {
  try {
    sessionStorage.removeItem(TOKEN_KEY)
  } catch {
    // Nothing stored
  }
}

// Called when the server rejects the stored token (expired or signed out elsewhere)
let unauthorizedHandler = null

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler
}

// status is the HTTP status, or 0 when the server could not be reached
export class ApiError extends Error {
  constructor(message, status, response = null) {
    super(message)
    this.status = status
    this.response = response
  }
}

// One JSON or multipart request to the backend. Adds "Authorization: Bearer <token>" when
// signed in (unless auth is false) and throws ApiError with the server's own
// message on failure. A 401 for a request that sent a token clears the token
// and notifies the unauthorized handler.
export async function apiRequest(path, { method = 'GET', body, auth = true } = {}) {
  const token = auth ? getStoredToken() : null
  const headers = {}
  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData
  if (body !== undefined && !isFormData) headers['Content-Type'] = 'application/json'
  if (token) headers.Authorization = `Bearer ${token}`

  let res
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : isFormData ? body : JSON.stringify(body),
    })
  } catch {
    throw new ApiError(SERVER_CONNECTION_ERROR, 0)
  }

  const json = await res.json().catch(() => null)

  if (res.status === 401 && token) {
    clearStoredToken()
    unauthorizedHandler?.()
  }

  if (!res.ok) {
    throw new ApiError(
      json?.message || `The server returned an error (HTTP ${res.status}).`,
      res.status,
      json,
    )
  }
  return json
}
