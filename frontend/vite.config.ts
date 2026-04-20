import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 7869,
    allowedHosts: ["www.gunterdong.top", "gunterdong.top"],
    hmr: false,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8889",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
