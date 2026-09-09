import { useQuery } from '@tanstack/react-query'
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Bot,
  BriefcaseBusiness,
  CandlestickChart,
  DatabaseZap,
  FileArchive,
  ScanSearch,
} from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { EmptyState, StatusPill } from '../components/Status'
import type { Experiment } from '../types'

const tabs = [
  ['Overview', Activity],
  ['Portfolio', BriefcaseBusiness],
  ['Trades', CandlestickChart],
  ['Candidates', ScanSearch],
  ['HSG-MT', Bot],
  ['Metrics', BarChart3],
  ['Agents', DatabaseZap],
  ['Reports', FileArchive],
] as const

function formatCapital(item: Experiment) {
  const currency = item.base_currency ?? 'INR'
  return new Intl.NumberFormat(currency === 'USD' ? 'en-US' : 'en-IN', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(Number(item.initial_capital))
}

export function ExperimentDetail() {
  const { experimentId = '' } = useParams()
  const query = useQuery({
    queryKey: ['experiment', experimentId],
    queryFn: () => api<Experiment>(`/api/v1/experiments/${experimentId}`),
  })
  const item = query.data
  const failed = item?.state === 'FAILED'

  return (
    <div className="page">
      <Link className="back-link" to="/experiments"><ArrowLeft size={15} />All experiments</Link>
      <div className="detail-heading">
        <div>
          <span className="section-kicker"><span>RUN</span> {item?.id ?? experimentId}</span>
          <h2>{item?.name ?? 'Loading experiment…'}</h2>
        </div>
        {item && <StatusPill value={item.state} />}
      </div>
      <div className="tabs" role="tablist">
        {tabs.map(([label, Icon]) => (
          <button key={label} className={`tab ${label === 'Overview' ? 'active' : ''}`}>
            <Icon size={15} />{label}
          </button>
        ))}
      </div>
      <div className="detail-grid">
        <div className="panel detail-configuration">
          <div className="panel-title"><div><span>RUN CONFIGURATION</span><h3>Immutable parameters</h3></div></div>
          {item ? (
            <dl>
              <div><dt>Market</dt><dd>{item.exchange ?? 'NSE'}</dd></div>
              <div><dt>Initial capital</dt><dd>{formatCapital(item)}</dd></div>
              <div><dt>Duration</dt><dd>{item.duration} {item.duration_unit.toLowerCase().replaceAll('_', ' ')}</dd></div>
              <div><dt>Created</dt><dd>{new Date(item.created_at).toLocaleString('en-IN')}</dd></div>
              <div><dt>Error state</dt><dd>{item.error_code ?? 'None'}</dd></div>
            </dl>
          ) : <p>Loading…</p>}
        </div>
        <div className="panel">
          <div className="panel-title"><div><span>OBSERVATIONS</span><h3>Experiment telemetry</h3></div></div>
          <EmptyState
            title={failed ? (item?.error_code ?? 'Experiment failed') : 'No observations recorded'}
            detail={failed
              ? 'The run stopped safely without generating a prediction or trade.'
              : 'Worker observations will appear after the experiment begins.'}
          />
        </div>
      </div>
    </div>
  )
}
