'use client'
export const dynamic = 'force-dynamic'

import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Skeleton } from '@sloughgpt/strui'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@sloughgpt/strui'
import { Button, Spinner, Breadcrumbs, IconTrash, IconDownload } from '@sloughgpt/strui'
import { useTrainingJob } from '@/hooks/useTrainingJob'
import { JobStatusCard } from '@/components/training/JobStatusCard'
import { JobLossCard } from '@/components/training/JobLossCard'
import { JobDetailsCard } from '@/components/training/JobDetailsCard'

export default function TrainingJobDetailPage() {
  const {
    jobId,
    job,
    loading,
    fetchError,
    summaryText,
    summaryLoading,
    showDelete,
    setShowDelete,
    badge,
    fetchJob,
    handleExport,
    handleLoadCheckpoint,
    handleDelete,
    handleResume,
    handleStop,
    handleDownloadCheckpoint,
    handleTryInChat,
  } = useTrainingJob()

  const headerRight = (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="sm"
        onClick={fetchJob}
        disabled={loading}
        aria-label="Refresh job status"
      >
        <Spinner className="h-4 w-4" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={handleExport}
        disabled={!job}
        aria-label="Export job details"
      >
        <IconDownload className="h-4 w-4" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="text-destructive"
        onClick={() => setShowDelete(true)}
        disabled={!job}
        aria-label="Delete job"
      >
        <IconTrash className="h-4 w-4" />
      </Button>
    </div>
  )

  return (
    <PageContainer
      title={loading ? '...' : job?.name || jobId}
      headerRight={headerRight}
      loading={loading}
      loadingContent={
        <div className="space-y-3">
          <Skeleton className="h-32 rounded-lg" />
          <Skeleton className="h-48 rounded-lg" />
        </div>
      }
    >
      <Breadcrumbs
        items={[
          { label: 'Training', href: '/training' },
          { label: loading ? '...' : job?.name || jobId },
        ]}
        className="mb-3"
      />

      {!loading && fetchError ? (
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-sm text-destructive mb-2">{fetchError}</p>
            <Button size="sm" variant="outline" onClick={fetchJob}>
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : !loading && !job ? (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Job not found
          </CardContent>
        </Card>
      ) : job ? (
        <>
          {/* Plain-language explanation when completed */}
          {job.status === 'completed' && job.explanation && (
            <div className="rounded-lg border border-success/20 bg-success/5 p-4">
              <p className="text-sm font-medium text-success">{job.explanation}</p>
            </div>
          )}

          {/* Summary card */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Summary</CardTitle>
            </CardHeader>
            <CardContent>
              {summaryLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-4 w-4/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              ) : summaryText ? (
                <div className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
                  {summaryText}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Summary not available</p>
              )}
            </CardContent>
          </Card>

          <JobStatusCard
            job={job}
            badge={badge}
            onResume={handleResume}
            onStop={handleStop}
            onLoadCheckpoint={handleLoadCheckpoint}
            onDownloadCheckpoint={handleDownloadCheckpoint}
            onTryInChat={handleTryInChat}
          />

          <JobLossCard job={job} />

          <JobDetailsCard job={job} />
        </>
      ) : null}

      <AlertDialog open={showDelete} onOpenChange={setShowDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete job</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &ldquo;{job?.name || job?.id}&rdquo;? This cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageContainer>
  )
}
