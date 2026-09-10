import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, FlaskConical, LockKeyhole, Plus } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { StatusPill } from '../components/Status'
import type { Experiment, ModelStatus } from '../types'

type Exchange = 'NSE' | 'BSE' | 'NASDAQ'

function formatCapital(value: string, currency: string) {
  return new Intl.NumberFormat(currency === 'USD' ? 'en-US' : 'en-IN', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(Number(value))
}

export function Experiments() {
  const signedIn = Boolean(localStorage.getItem('hsg-token'))
  const [creating, setCreating] = useState(false)
  const [exchange, setExchange] = useState<Exchange>('NSE')
  const client = useQueryClient()
  const navigate = useNavigate()
  const query = useQuery({
    queryKey: ['experiments'],
    queryFn: () => api<Experiment[]>('/api/v1/experiments'),
    enabled: signedIn,
  })
  const model = useQuery({
    queryKey: ['model'],
    queryFn: () => api<ModelStatus>('/api/v1/system/model'),
  })
  const nasdaqReady =
    model.data?.status === 'READY' && model.data.trained_exchanges?.includes('NASDAQ')
  const create = useMutation({
    mutationFn: (payload: Record<string, unknown>) =>
      api<Experiment>('/api/v1/experiments', {
        method: 'POST',
        body: JSON.stringify(payload),
      }),
    onSuccess: async (item) => {
      setCreating(false)
      await client.invalidateQueries({ queryKey: ['experiments'] })
      navigate(`/experiments/${item.id}`)
    },
  })

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    create.mutate({
      name: form.get('name'),
      exchange,
      base_currency: exchange === 'NASDAQ' ? 'USD' : 'INR',
      initial_capital: form.get('capital'),
      duration: Number(form.get('duration')),
      duration_unit: form.get('duration_unit'),
      max_positions: Number(form.get('max_positions')),
      max_position_pct: form.get('max_position_pct'),
      scanner_frequency_seconds: Number(form.get('scan_interval')),
      confidence_threshold: form.get('confidence'),
      transaction_cost_bps: form.get('costs'),
      slippage_bps: form.get('slippage'),
    })
  }

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <span className="section-kicker"><span>RUNS</span> IMMUTABLE RESEARCH CONFIGURATIONS</span>
          <h2>Experiment registry</h2>
          <p>Run isolated NSE, BSE, or model-compatible NASDAQ research experiments.</p>
        </div>
        <button className="button" disabled={!signedIn} onClick={() => setCreating((value) => !value)}>
          <Plus size={16} />New experiment
        </button>
      </div>

      {creating && (
        <form className="experiment-form" onSubmit={submit}>
          <label>
            Experiment name
            <input name="name" defaultValue="Two-day HSG-MT study" required minLength={3} />
          </label>
          <label>
            Market universe
            <select name="exchange" value={exchange} onChange={(event) => setExchange(event.target.value as Exchange)}>
              <option value="NSE">NSE</option>
              <option value="BSE">BSE</option>
              <option value="NASDAQ" disabled={!nasdaqReady}>
                {nasdaqReady ? 'NASDAQ' : 'NASDAQ — requires NASDAQ-trained model'}
              </option>
            </select>
          </label>
          <label>
            Initial capital ({exchange === 'NASDAQ' ? 'USD' : 'INR'})
            <input name="capital" type="number" min="1" defaultValue="1000000" required />
          </label>
          <label>
            Duration
            <input name="duration" type="number" min="1" defaultValue="2" required />
          </label>
          <label>
            Duration unit
            <select name="duration_unit" defaultValue="CALENDAR_DAYS">
              <option value="CALENDAR_HOURS">Calendar hours</option>
              <option value="CALENDAR_DAYS">Calendar days</option>
              <option value="MARKET_SESSIONS">Market sessions</option>
            </select>
          </label>
          <label>Max positions<input name="max_positions" type="number" min="1" defaultValue="5" /></label>
          <label>Max position %<input name="max_position_pct" type="number" min="0.01" max="1" step="0.01" defaultValue="0.20" /></label>
          <label>Scan interval (seconds)<input name="scan_interval" type="number" min="30" defaultValue="300" /></label>
          <label>Confidence threshold<input name="confidence" type="number" min="0" max="1" step="0.01" defaultValue="0.65" /></label>
          <label>Transaction cost (bps)<input name="costs" type="number" min="0" defaultValue="10" /></label>
          <label>Slippage (bps)<input name="slippage" type="number" min="0" defaultValue="5" /></label>
          {!nasdaqReady && (
            <div className="form-note">NASDAQ is prepared for future use and unlocks only after a checkpoint is trained with <code>--markets NASDAQ</code>.</div>
          )}
          {create.isError && <div className="form-error">{create.error.message}</div>}
          <div className="form-actions">
            <button type="button" className="button button-secondary" onClick={() => setCreating(false)}>Cancel</button>
            <button className="button" disabled={create.isPending}>{create.isPending ? 'Creating…' : 'Create draft'}</button>
          </div>
        </form>
      )}

      {!signedIn && (
        <div className="auth-gate">
          <LockKeyhole size={26} />
          <div><h3>Authentication required</h3><p>Sign in to inspect or control experiments. Public system health remains visible.</p></div>
          <Link className="button" to="/login">Sign in<ArrowRight size={16} /></Link>
        </div>
      )}

      <div className="experiment-list">
        {query.data?.map((item) => (
          <Link to={`/experiments/${item.id}`} className="experiment-row" key={item.id}>
            <span className="run-icon"><FlaskConical /></span>
            <div className="run-name"><strong>{item.name}</strong><span>{item.exchange ?? 'NSE'} · {item.id}</span></div>
            <div><span className="table-label">Capital</span><strong>{formatCapital(item.initial_capital, item.base_currency ?? 'INR')}</strong></div>
            <div><span className="table-label">Duration</span><strong>{item.duration} {item.duration_unit.toLowerCase().replaceAll('_', ' ')}</strong></div>
            <StatusPill value={item.state} />
            <ArrowRight size={17} />
          </Link>
        ))}
      </div>
    </div>
  )
}
