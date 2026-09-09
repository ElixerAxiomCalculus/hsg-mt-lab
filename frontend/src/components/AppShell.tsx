import {
  Activity,
  Atom,
  ChevronRight,
  Database,
  FlaskConical,
  Gauge,
  LogIn,
  Menu,
  ServerCog,
  ShieldCheck,
  X,
} from 'lucide-react'
import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'

const navigation = [
  { to: '/', label: 'Overview', icon: Gauge },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/system', label: 'System health', icon: ServerCog },
]

const titles: Record<string, string> = {
  '/': 'Research overview',
  '/experiments': 'Experiments',
  '/datasets': 'Dataset registry',
  '/system': 'System health',
  '/login': 'Secure access',
}

export function AppShell() {
  const [open, setOpen] = useState(false)
  const location = useLocation()
  const signedIn = Boolean(localStorage.getItem('hsg-token'))
  const title = location.pathname.startsWith('/experiments/')
    ? 'Experiment detail'
    : (titles[location.pathname] ?? 'HSG-MT Lab')

  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><Atom size={22} /></div>
          <div>
            <strong>HSG–MT</strong>
            <span>RESEARCH LAB</span>
          </div>
          <button className="icon-button close-nav" onClick={() => setOpen(false)} aria-label="Close navigation"><X size={19} /></button>
        </div>
        <nav aria-label="Primary navigation">
          <div className="nav-label">Workspace</div>
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === '/'} onClick={() => setOpen(false)}>
              <Icon size={18} />
              <span>{label}</span>
              <ChevronRight className="nav-arrow" size={15} />
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="safety-seal"><ShieldCheck size={17} /><span>Simulation boundary active</span></div>
          <p>No brokerage connections<br />No real-money execution</p>
        </div>
      </aside>
      {open && <button className="nav-scrim" onClick={() => setOpen(false)} aria-label="Close navigation" />}
      <div className="main-column">
        <header className="topbar">
          <div className="title-row">
            <button className="icon-button menu-button" onClick={() => setOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
            <div><span className="eyebrow">HYPERSPECTRUM GEOMETRY</span><h1>{title}</h1></div>
          </div>
          <div className="top-actions">
            <div className="paper-badge"><span className="pulse-dot" />PAPER TRADING / RESEARCH ONLY</div>
            <NavLink className="account-button" to="/login">
              {signedIn ? <Activity size={16} /> : <LogIn size={16} />}
              {signedIn ? 'Authenticated' : 'Sign in'}
            </NavLink>
          </div>
        </header>
        <main><Outlet /></main>
      </div>
    </div>
  )
}
