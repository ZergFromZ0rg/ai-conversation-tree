import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/conversations": "http://127.0.0.1:8000",
      "/edges": "http://127.0.0.1:8000"
    }
  }
});
