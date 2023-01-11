import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import Markdown from 'vite-plugin-md'
import Components from 'unplugin-vue-components/vite'
import prism from 'markdown-it-prism'

// https://vitejs.dev/config/
export default defineConfig({
  base: './',
  plugins: [
    vue({
      include: [/\.vue$/, /\.md$/],
    }),
    Markdown({
      markdownItUses: [prism]
    }),
    Components({
      dirs: [
        'src/components',
        'src/md'
      ],
      extensions: [
        'vue',
        'md'
      ]
    })
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  build: {
    rollupOptions: {
      external: ['onnxruntime-web'],
    },
  }
})
