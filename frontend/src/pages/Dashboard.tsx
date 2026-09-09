import { useQuery } from '@tanstack/react-query'
import { Activity, ArrowUpRight, CalendarClock, Cpu, Database, IndianRupee, Radio, WalletCards } from 'lucide-react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/client'
import { EmptyState, StatusPill } from '../components/Status'
import type { AlgorithmStatus, DatasetInspection, Experiment, MarketStatus } from '../types'

const formatDate = (value?: string) => value ? new Intl.DateTimeFormat('en-IN', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Kolkata' }).format(new Date(value)) : '—'
const formatMoney = (value?: string, currency = 'INR') => value ? new Intl.NumberFormat(currency === 'USD' ? 'en-US' : 'en-IN', { style: 'currency', currency, maximumFractionDigits: 0 }).format(Number(value)) : '—'

export function Dashboard() {
  const algorithm = useQuery({ queryKey: ['algorithm'], queryFn: () => api<AlgorithmStatus>('/api/v1/system/algorithm') })
  const market = useQuery({ queryKey: ['market'], queryFn: () => api<MarketStatus>('/api/v1/system/market'), refetchInterval: 60_000 })
  const datasets = useQuery({ queryKey: ['datasets'], queryFn: () => api<DatasetInspection[]>('/api/v1/datasets') })
  const experiments = useQuery({ queryKey: ['experiments'], queryFn: () => api<Experiment[]>('/api/v1/experiments'), enabled: Boolean(localStorage.getItem('hsg-token')) })
  const current = experiments.data?.find((item) => ['RUNNING', 'WAITING_FOR_MARKET', 'INITIALIZING'].includes(item.state))

  return (
    <div className="page dashboard-page">
      <section className="system-ribbon">
        <div><span className="system-label"><Cpu size={15} />Algorithm</span><StatusPill value={algorithm.data?.status ?? (algorithm.isError ? 'UNAVAILABLE' : 'CHECKING')} /></div>
        <div><span className="system-label"><Radio size={15} />NSE/BSE session</span><StatusPill value={market.data?.status ?? (market.isError ? 'UNAVAILABLE' : 'CHECKING')} /></div>
        <div className="ribbon-grow"><span className="system-label"><CalendarClock size={15} />Next market open</span><strong>{formatDate(market.data?.next_market_open)}</strong></div>
        <div><span className="system-label"><Database size={15} />Datasets</span><strong>{datasets.data?.length ?? '—'} registered</strong></div>
      </section>

      <section className="lead-grid">
        <div className="lead-panel">
          <div className="section-kicker"><span>01</span> EXPERIMENT CONTROL</div>
          <div className="lead-heading">
            <div><h2>{current?.name ?? 'No active experiment'}</h2><p>{current ? 'Autonomous worker state is persisted independently of this browser.' : 'Create an experiment after installing and training the HSG-MT algorithm.'}</p></div>
            <StatusPill value={current?.state ?? 'IDLE'} />
          </div>
          <div className="metric-grid">
            <Metric label="Starting capital" value={formatMoney(current?.initial_capital, current?.base_currency)} icon={<IndianRupee />} />
            <Metric label="Portfolio equity" value="N/A" icon={<WalletCards />} />
            <Metric label="Total return" value="N/A" icon={<ArrowUpRight />} />
            <Metric label="Open positions" value="0" icon={<Activity />} />
          </div>
        </div>
        <div className="algorithm-panel">
          <div className="spectral-heading"><div className="orbit-mark"><span /><span /><span /></div><div><span>HSG-MT CORE</span><h3>{algorithm.data?.status === 'INSTALLED' ? 'Scientific engine online' : 'Algorithm not installed'}</h3></div></div>
          <p className="algorithm-copy">The research shell is operational. Predictions and trading remain locked until <code>backend/algo.py</code> satisfies the documented contract.</p>
          <div className="hash-block"><span>ALGORITHM SHA-256</span><code>{algorithm.data?.algorithm_sha256 ?? 'Checking adapter…'}</code></div>
          <div className="capability-list">
            {['pretrain', 'predict', 'online_update', 'save_checkpoint'].map((name) => <div key={name}><span>{name.replace('_', ' ')}</span><i className={algorithm.data?.capabilities[name] ? 'cap-on' : ''} /></div>)}
          </div>
        </div>
      </section>

      <section className="chart-grid">
        <div className="panel chart-panel">
          <div className="panel-title"><div><span>PORTFOLIO EQUITY</span><h3>Capital trajectory</h3></div><span className="data-tag">NO OBSERVATIONS</span></div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={[]} margin={{ top: 10, right: 8, left: 0, bottom: 0 }}>
                <defs><linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#78f2c2" stopOpacity={0.25} /><stop offset="100%" stopColor="#78f2c2" stopOpacity={0} /></linearGradient></defs>
                <CartesianGrid stroke="#1b2c29" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="time" stroke="#60736f" tickLine={false} axisLine={false} />
                <YAxis stroke="#60736f" tickLine={false} axisLine={false} />
                <Tooltip />
                <Area dataKey="equity" type="monotone" stroke="#78f2c2" fill="url(#equityFill)" />
              </AreaChart>
            </ResponsiveContainer>
            <div className="chart-empty"><Activity size={25} /><strong>Awaiting experiment telemetry</strong><span>Equity, benchmark, and trade markers will appear here.</span></div>
          </div>
        </div>
        <div className="panel event-panel">
          <div className="panel-title"><div><span>LATEST ACTIVITY</span><h3>Agent event stream</h3></div><span className="live-label"><i /> LIVE</span></div>
          <EmptyState title="No research events yet" detail="Events are written by the autonomous worker and survive browser disconnects." />
          <div className="event-legend"><span><i className="good-dot" />Validated</span><span><i className="warn-dot" />Guardrail</span><span><i className="bad-dot" />Failure</span></div>
        </div>
      </section>
    </div>
  )
}

function Metric({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return <div className="metric"><span className="metric-icon">{icon}</span><div><span>{label}</span><strong>{value}</strong></div></div>
}
