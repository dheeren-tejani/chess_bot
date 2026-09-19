import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev uses the SAME relative URLs as production: Vite forwards them
    // to your local backend, exactly like the Netlify edge function does
    // in production. No URL ever appears in frontend code.
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/healthz': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
});