import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "chess_ml",
  description: "Personalized chess coaching — local MVP",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
