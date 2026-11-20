import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The FastAPI server serves the production build from dist/. During
// development, `npm run dev` proxies the API and both WebSockets to it.
const BACKEND = process.env.CROWDGUARD_BACKEND ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': BACKEND,
      '/ws': { target: BACKEND.replace(/^http/, 'ws'), ws: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 800,
  },
});
