import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          graph: ["@xyflow/react", "@dagrejs/dagre"],
          markdown: ["react-markdown"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/conversations": "http://127.0.0.1:8000",
      "/edges": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
      "/concepts": "http://127.0.0.1:8000"
    }
  }
});
