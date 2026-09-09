import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { Dashboard } from './pages/Dashboard'
import { Datasets } from './pages/Datasets'
import { Experiments } from './pages/Experiments'
import { ExperimentDetail } from './pages/ExperimentDetail'
import { Login } from './pages/Login'
import { System } from './pages/System'

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="datasets" element={<Datasets />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="experiments/:experimentId" element={<ExperimentDetail />} />
        <Route path="system" element={<System />} />
        <Route path="login" element={<Login />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
