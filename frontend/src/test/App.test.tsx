import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import App from '../App'

test('keeps the paper-trading boundary visible', () => {
  render(<QueryClientProvider client={new QueryClient()}><MemoryRouter><App /></MemoryRouter></QueryClientProvider>)
  expect(screen.getByText(/PAPER TRADING \/ RESEARCH ONLY/i)).toBeInTheDocument()
})

