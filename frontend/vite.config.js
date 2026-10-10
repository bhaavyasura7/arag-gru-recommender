import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/users': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
      '/feed': 'http://127.0.0.1:8000',
      '/search': 'http://127.0.0.1:8000',
      '/feedback': 'http://127.0.0.1:8000',
      '/runs': 'http://127.0.0.1:8000',
      '/results': 'http://127.0.0.1:8000'
    }
  }
})
