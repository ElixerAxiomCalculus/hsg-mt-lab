const defaultApiUrl = import.meta.env.DEV
  ? 'http://localhost:8000'
  : 'https://hsg-mt-lab.onrender.com'

export const API_URL = (import.meta.env.VITE_API_URL || defaultApiUrl).replace(/\/$/, '')

export class ApiFailure extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const token = localStorage.getItem('hsg-token')
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = (await response.json()) as { detail?: string; error?: { message?: string } }
      message = body.detail ?? body.error?.message ?? message
    } catch {
      // Keep the HTTP status message when the body is not JSON.
    }
    throw new ApiFailure(response.status, message)
  }
  return response.json() as Promise<T>
}
