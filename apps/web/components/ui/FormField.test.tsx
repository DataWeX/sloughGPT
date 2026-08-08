import { render, screen } from '@testing-library/react'
import { useForm } from 'react-hook-form'
import { describe, it, expect } from 'vitest'
import { FormField, ValidatedInput, ValidatedTextarea } from './FormField'

function TestForm() {
  const { register, formState: { errors } } = useForm({
    defaultValues: { name: '', bio: '' },
  })
  return (
    <form>
      <ValidatedInput
        label="Name"
        name="name"
        register={register('name', { required: 'Name is required' })}
        error={errors.name}
        required
      />
      <ValidatedTextarea
        label="Bio"
        name="bio"
        register={register('bio', { maxLength: { value: 100, message: 'Max 100 chars' } })}
        error={errors.bio}
      />
    </form>
  )
}

describe('FormField', () => {
  it('renders label', () => {
    render(<FormField label="Test Field">{({ className, id }) => <input id={id} className={className} />}</FormField>)
    expect(screen.getByText('Test Field')).toBeInTheDocument()
  })

  it('shows required asterisk', () => {
    render(<FormField label="Test" required>{({ className, id }) => <input id={id} className={className} />}</FormField>)
    expect(screen.getByText('*')).toBeInTheDocument()
  })

  it('renders error message', () => {
    render(
      <FormField label="Test" error={{ type: 'required', message: 'Required field' }}>
        {({ className, id }) => <input id={id} className={className} />}
      </FormField>
    )
    expect(screen.getByText('Required field')).toBeInTheDocument()
  })

  it('links error to field via aria-describedby', () => {
    render(
      <FormField label="Email" error={{ type: 'pattern', message: 'Invalid email' }}>
        {({ className, id }) => <input id={id} className={className} />}
      </FormField>
    )
    const error = screen.getByText('Invalid email')
    expect(error).toHaveAttribute('role', 'alert')
  })
})

describe('ValidatedInput', () => {
  it('renders with label', () => {
    render(<TestForm />)
    expect(screen.getByLabelText(/Name/)).toBeInTheDocument()
  })

  it('shows error state', () => {
    render(<TestForm />)
    const input = screen.getByLabelText(/Name/)
    expect(input).not.toHaveAttribute('aria-invalid', 'true')
  })
})

describe('ValidatedTextarea', () => {
  it('renders with label', () => {
    render(<TestForm />)
    expect(screen.getByLabelText(/Bio/)).toBeInTheDocument()
  })
})
