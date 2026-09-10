import { useQuery } from '@tanstack/react-query'
import {
  BrainCircuit,
  CalendarClock,
  CircleDot,
  Cpu,
  Database,
  Radio,
  ServerCog,
} from 'lucide-react'
import { api } from '../api/client'
import { StatusPill } from '../components/Status'
import type { AlgorithmStatus, MarketStatus, ModelStatus, WorkerStatus } from '../types'

export function System() {
  const algorithm = useQuery({
    queryKey: ['algorithm'],
    queryFn: () => api<AlgorithmStatus>('/api/v1/system/algorithm'),
  })
  const model = useQuery({
    queryKey: ['model'],
    queryFn: () => api<ModelStatus>('/api/v1/system/model'),
  })
  const market = useQuery({
    queryKey: ['market'],
    queryFn: () => api<MarketStatus>('/api/v1/system/market'),
    refetchInterval: 60_000,
  })
  const worker = useQuery({
    queryKey: ['worker'],
    queryFn: () => api<WorkerStatus>('/api/v1/system/worker'),
    refetchInterval: 10_000,
  })

  const databaseStatus = worker.data ? 'CONNECTED' : (worker.isError ? 'UNAVAILABLE' : 'CHECKING')

  return (
    <div className="page">
      <div className="page-intro">
        <div>
          <span className="section-kicker"><span>OPS</span> OBSERVABLE RUNTIME</span>
          <h2>System health</h2>
          <p>General health remains available even when the scientific implementation is missing.</p>
        </div>
      </div>
      <div className="health-grid">
        <Health
          icon={<Cpu />}
          label="HSG-MT adapter"
          value={algorithm.data?.status ?? 'CHECKING'}
          detail="Dynamic contract validation"
        />
        <Health
          icon={<BrainCircuit />}
          label="Active model"
          value={model.data?.status ?? 'CHECKING'}
          detail={model.data?.model_version ?? 'Run offline pretraining from the terminal'}
        />
        <Health
          icon={<Radio />}
          label="Market session"
          value={market.data?.status ?? 'CHECKING'}
          detail="Asia/Kolkata exchange calendar"
        />
        <Health
          icon={<ServerCog />}
          label="Autonomous worker"
          value={worker.data?.status ?? (worker.isError ? 'UNAVAILABLE' : 'CHECKING')}
          detail={worker.data?.heartbeat ? `Heartbeat ${worker.data.heartbeat}` : 'No heartbeat recorded'}
        />
        <Health
          icon={<Database />}
          label="MongoDB Atlas"
          value={databaseStatus}
          detail="Persistent experiment state"
        />
      </div>
      <div className="panel system-details">
        <div className="panel-title">
          <div><span>SCIENTIFIC BOUNDARY</span><h3>Algorithm capabilities</h3></div>
        </div>
        <div className="contract-grid">
          {Object.entries(algorithm.data?.capabilities ?? {}).map(([name, available]) => (
            <div key={name}>
              <CircleDot size={15} className={available ? 'text-good' : 'text-muted'} />
              <span>{name.replaceAll('_', ' ')}</span>
              <strong>{available ? 'AVAILABLE' : 'NOT PROVIDED'}</strong>
            </div>
          ))}
        </div>
        <div className="next-session">
          <CalendarClock size={18} />
          <span>Next regular or overridden session</span>
          <strong>
            {market.data?.next_market_open
              ? new Date(market.data.next_market_open).toLocaleString('en-IN', {
                  timeZone: 'Asia/Kolkata',
                })
              : '—'}
          </strong>
        </div>
      </div>
    </div>
  )
}

function Health({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode
  label: string
  value: string
  detail: string
}) {
  return (
    <article className="health-card">
      <div className="health-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <StatusPill value={value} />
        <p>{detail}</p>
      </div>
    </article>
  )
}
