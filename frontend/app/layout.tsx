import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Revenue Recovery Autopilot",
  description:
    "Recover more revenue from failed payments, automatically — but safely. " +
    "Razorpay AI Buildathon — Track 3.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-950 antialiased`}>
        {children}
      </body>
    </html>
  );
}
