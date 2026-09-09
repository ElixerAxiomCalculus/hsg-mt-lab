import { FormEvent, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Atom, KeyRound, ShieldCheck } from 'lucide-react'
import { api } from '../api/client'

export function Login() {
  const navigate = useNavigate()
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true); setError('')
    const form = new FormData(event.currentTarget)
    try {
      const result = await api<{ access_token: string }>('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ username: form.get('username'), password: form.get('password') }) })
      localStorage.setItem('hsg-token', result.access_token)
      navigate('/experiments')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to sign in')
    } finally { setBusy(false) }
  }
  return <div className="page login-page"><div className="login-card"><div className="login-symbol"><Atom /></div><span className="section-kicker"><span>SECURE</span> PERSONAL RESEARCH ACCESS</span><h2>Sign in to the lab</h2><p>Experiment controls and stored research records require an authenticated session.</p><form onSubmit={submit}><label>Username<input name="username" autoComplete="username" required /></label><label>Password<input name="password" type="password" autoComplete="current-password" required /></label>{error && <div className="form-error">{error}</div>}<button className="button login-submit" disabled={busy}><KeyRound size={16} />{busy ? 'Verifying…' : 'Sign in'}</button></form><div className="login-note"><ShieldCheck size={16} /><span>Credentials are verified by the API. Secrets never enter the frontend bundle.</span></div></div></div>
}

