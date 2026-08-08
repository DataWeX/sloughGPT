'use client'

import { type FieldError, type UseFormRegisterReturn } from 'react-hook-form'
import { Input, Textarea, cn } from '@sloughgpt/strui'

interface FormFieldProps {
  label: string
  error?: FieldError
  required?: boolean
  className?: string
  children: (props: { className: string; id: string; errorId: string }) => React.ReactNode
}

export function FormField({ label, error, required, className, children }: FormFieldProps) {
  const id = `field-${label.toLowerCase().replace(/\s+/g, '-')}`
  const errorId = `${id}-error`
  return (
    <div className={cn('space-y-1', className)}>
      <label htmlFor={id} className="text-xs font-medium text-foreground">
        {label}
        {required && <span className="text-destructive ml-0.5">*</span>}
      </label>
      {children({
        className: cn(
          'h-8 text-xs',
          error && 'border-destructive ring-destructive/20 focus-visible:ring-destructive/30'
        ),
        id,
        errorId,
      })}
      {error && (
        <p id={errorId} className="text-[10px] text-destructive" role="alert">
          {error.message}
        </p>
      )}
    </div>
  )
}

interface ValidatedInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'name'> {
  name: string
  register: UseFormRegisterReturn
  error?: FieldError
  label: string
  required?: boolean
}

export function ValidatedInput({ name, register, error, label, required, className, ...props }: ValidatedInputProps) {
  return (
    <FormField label={label} error={error} required={required}>
      {({ className: fieldClass, id, errorId }) => (
        <Input
          id={id}
          className={fieldClass}
          aria-invalid={!!error}
          aria-describedby={error ? errorId : undefined}
          {...register}
          {...props}
        />
      )}
    </FormField>
  )
}

interface ValidatedTextareaProps extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, 'name'> {
  name: string
  register: UseFormRegisterReturn
  error?: FieldError
  label: string
  required?: boolean
}

export function ValidatedTextarea({ name, register, error, label, required, className, ...props }: ValidatedTextareaProps) {
  return (
    <FormField label={label} error={error} required={required}>
      {({ className: fieldClass, id, errorId }) => (
        <Textarea
          id={id}
          className={cn(fieldClass, 'resize-none')}
          aria-invalid={!!error}
          aria-describedby={error ? errorId : undefined}
          {...register}
          {...props}
        />
      )}
    </FormField>
  )
}
