export interface ColorPalette {
  background: string;
  surface: string;
  border: string;
  primaryAccent: string;
  secondaryAccent: string;
  optionalAccent: string;
  textPrimary: string; // For main text (off-white/light gray)
  textTitles: string;  // For titles/headings (#F8F805)
}

export const batmanTheme: ColorPalette = {
  background: "#000000",       // Dark Knight Black
  surface: "#232834",          // Gotham Gray
  border: "#36454F",           // Shadowy Charcoal
  primaryAccent: "#FFD700",    // Batarang Gold
  secondaryAccent: "#BA8E23",  // Dark Yellow
  optionalAccent: "#003366",   // Batman Blue (optional)
  textPrimary: "#E0E0E0",      // Light Gray (for readability)
  textTitles: "#F8F805"        // Soft Yellow
}; 