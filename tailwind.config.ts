import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    // Add "./src/pages/**/*.{js,ts,jsx,tsx,mdx}" below if you also use the pages directory inside src
  ],
  theme: {
    extend: {
      // You can add theme customizations here later if needed
      // Example:
      // colors: {
      //   primary: '#...',
      // },
    },
  },
  plugins: [
     // You can add Tailwind plugins here later if needed
  ],
};

export default config; 