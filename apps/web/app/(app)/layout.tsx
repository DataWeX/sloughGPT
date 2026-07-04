import "../globals.css";
import AppLayout from "@/components/AppLayout";

export const dynamic = 'force-dynamic'

export default function AppLayoutWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppLayout>{children}</AppLayout>;
}
