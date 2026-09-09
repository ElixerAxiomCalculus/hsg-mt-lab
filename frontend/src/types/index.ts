export type AlgorithmStatus = {
  status: 'NOT_INSTALLED' | 'CONTRACT_INVALID' | 'INSTALLED'
  algorithm_sha256: string
  capabilities: Record<string, boolean>
  errors: string[]
}

export type ModelStatus = {
  status: 'READY' | 'NOT_TRAINED' | 'INCOMPATIBLE' | 'UNAVAILABLE'
  model_version: string | null
  created_at: string | null
  algorithm_sha256?: string
  trained_exchanges?: string[]
}

export type MarketStatus = {
  status: 'OPEN' | 'CLOSED'
  timestamp: string
  timezone: string
  next_market_open: string
  session: { name: string; opens_at: string; closes_at: string } | null
}

export type DatasetInspection = {
  filename: string
  sha256: string
  file_size: number
  row_count: number
  columns: string[]
  normalized_columns: string[]
  detected_type: string
  min_date: string | null
  max_date: string | null
  unique_symbols: number | null
  null_counts: Record<string, number>
  duplicate_rows: number
  invalid_price_rows: number
  invalid_dates: number
  warnings: string[]
}

export type Experiment = {
  id: string
  name: string
  state: string
  initial_capital: string
  exchange?: 'NSE' | 'BSE' | 'NASDAQ'
  base_currency?: 'INR' | 'USD'
  duration: number
  duration_unit: string
  created_at: string
  error_code?: string
}
