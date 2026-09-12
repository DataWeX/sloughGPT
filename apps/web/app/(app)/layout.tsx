import "../globals.css";
import AppLayout from "@/components/AppLayout";
import { ConsciousnessQuickActionsWrapper } from "@/components/consciousness/ConsciousnessQuickActionsWrapper";

export const dynamic = 'force-dynamic'

export default function AppLayoutWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <AppLayout>{children}</AppLayout>
      <ConsciousnessQuickActionsWrapper />
    </>
  );
}
