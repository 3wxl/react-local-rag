/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // 语义化颜色令牌：通过 CSS 变量驱动，支持 light/dark/sepia 三套主题
        bg: {
          DEFAULT: "var(--bg)",
          elevated: "var(--bg-elevated)",
          hover: "var(--bg-hover)",
          active: "var(--bg-active)",
        },
        line: "var(--border)",
        ink: {
          DEFAULT: "var(--text)",
          muted: "var(--text-muted)",
          faint: "var(--text-faint)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
          soft: "var(--accent-soft)",
          text: "var(--accent-text)",
        },
        avatar: {
          aiBg: "var(--avatar-ai-bg)",
          aiFg: "var(--avatar-ai-fg)",
          userBg: "var(--avatar-user-bg)",
          userFg: "var(--avatar-user-fg)",
        },
      },
    },
  },
  plugins: [],
};
