import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: "var(--surface)",
        border: "var(--border)",
        muted: "var(--muted)",
        "muted-foreground": "var(--muted-foreground)",
        primary: {
          DEFAULT: "#7065E8",
          hover: "#5E53D9",
          dark: "#4E42B8",
          light: "#EEECFC",
        },
        navy: {
          DEFAULT: "#171638",
          muted: "#36345C",
          light: "#85839A",
        },
        finguard: {
          rail: "#5448C8",
          surface: "#FFFFFF",
          bg: "#F4F3FA",
          card: "#FFFFFF",
          border: "#E9E7F2",
          text: "#171638",
          muted: "#85839A",
          success: "#239B79",
          error: "#D95465",
          warning: "#D9A441",
        },
      },
      borderRadius: {
        "3xl": "1.5rem",
        "4xl": "2rem",
      },
      boxShadow: {
        card: "0 4px 20px -2px rgba(23, 22, 56, 0.05), 0 2px 6px -1px rgba(23, 22, 56, 0.02)",
        "card-hover": "0 10px 30px -4px rgba(23, 22, 56, 0.08), 0 4px 12px -2px rgba(23, 22, 56, 0.04)",
        rail: "0 12px 36px 0 rgba(84, 72, 200, 0.25)",
      },
    },
  },
  plugins: [],
};

export default config;

