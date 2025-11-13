import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { resolve } from "path"

// https://vite.dev/config/
export default defineConfig({
  base: "/ai/ui/",
  plugins: [react()],
  server: {
    allowedHosts: [".com"]
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
})
