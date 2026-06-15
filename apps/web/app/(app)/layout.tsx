import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SloughGPT",
  description: "AI‑augmented chat & training platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head />
      <body>{children}</body>
    </html>
  );
}
