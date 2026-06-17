import "../globals.css";
import type { Metadata } from "next";
import AppLayout from "@/components/AppLayout";

export const metadata: Metadata = {
  title: "SloughGPT",
  description: "AI‑augmented chat & training platform",
};

export default function AppLayoutWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppLayout>{children}</AppLayout>;
}
