import { AlertTriangle, CheckCircle2, CircleOff, LoaderCircle } from 'lucide-react'

export function StatusPill({ value }: { value: string }) {
  const type = ['INSTALLED', 'OPEN', 'RUNNING', 'COMPLETED', 'CONNECTED'].includes(value)
    ? 'good'
    : ['NOT_INSTALLED', 'FAILED', 'UNAVAILABLE', 'CONTRACT_INVALID'].includes(value)
      ? 'bad'
      : 'neutral'
  const Icon = type === 'good' ? CheckCircle2 : type === 'bad' ? CircleOff : LoaderCircle
  return <span className={`status-pill status-${type}`}><Icon size={13} />{value.replaceAll('_', ' ')}</span>
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><AlertTriangle size={20} /><div><strong>{title}</strong><p>{detail}</p></div></div>
}

