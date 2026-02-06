import React from "react"
import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import { Toaster } from "@/components/ui/sonner"

import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })

export const metadata: Metadata = {
  title: 'AI Fraud & Anomaly Detector',
  description: 'Real-time AI-powered fraud detection and anomaly monitoring for trading and risk teams',
}

export const viewport: Viewport = {
  themeColor: '#0a0a0a',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body suppressHydrationWarning className={`${inter.variable} font-sans antialiased bg-background text-foreground`}>
        {children}
        <Toaster position="bottom-right" richColors />
      </body>
    </html>
  )
}
