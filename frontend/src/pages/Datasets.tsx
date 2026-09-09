import { useQuery } from '@tanstack/react-query'
import { AlertCircle, CheckCircle2, FileSpreadsheet, Fingerprint, RefreshCw } from 'lucide-react'
import { api } from '../api/client'
import type { DatasetInspection } from '../types'

const size = (bytes: number) => bytes > 1_000_000 ? `${(bytes / 1_000_000).toFixed(1)} MB` : `${(bytes / 1_000).toFixed(1)} KB`

export function Datasets() {
  const query = useQuery({ queryKey: ['datasets'], queryFn: () => api<DatasetInspection[]>('/api/v1/datasets') })
  return <div className="page">
    <div className="page-intro"><div><span className="section-kicker"><span>DATA</span> LOCAL CSV REGISTRY</span><h2>Training-source inspection</h2><p>Schema detection, temporal coverage, and content fingerprints are calculated from files already present in <code>/data</code>.</p></div><button className="button button-secondary" onClick={() => void query.refetch()}><RefreshCw size={16} />Reinspect</button></div>
    {query.isError && <div className="callout error"><AlertCircle size={19} />The dataset service is unavailable. Start the API and try again.</div>}
    <div className="dataset-stack">
      {query.data?.map((item) => <article className="dataset-card" key={item.filename}>
        <div className="dataset-icon"><FileSpreadsheet /></div>
        <div className="dataset-main">
          <div className="dataset-title"><div><h3>{item.filename}</h3><span>{size(item.file_size)} · {item.row_count.toLocaleString('en-IN')} rows</span></div><span className="type-tag">{item.detected_type.replaceAll('_', ' ')}</span></div>
          <div className="dataset-facts"><div><span>Date range</span><strong>{item.min_date?.slice(0,10) ?? 'Unknown'} → {item.max_date?.slice(0,10) ?? 'Unknown'}</strong></div><div><span>Symbols</span><strong>{item.unique_symbols?.toLocaleString('en-IN') ?? 'N/A'}</strong></div><div><span>Invalid prices</span><strong>{item.invalid_price_rows}</strong></div><div><span>Duplicates</span><strong>{item.duplicate_rows}</strong></div></div>
          <div className="column-list">{item.normalized_columns.map((column) => <code key={column}>{column}</code>)}</div>
          <div className="fingerprint"><Fingerprint size={14} /><span>{item.sha256}</span></div>
          {item.warnings.length ? <div className="warning-list">{item.warnings.map((warning) => <span key={warning}><AlertCircle size={14} />{warning}</span>)}</div> : <div className="validated"><CheckCircle2 size={14} />No blocking quality warnings</div>}
        </div>
      </article>)}
    </div>
  </div>
}

