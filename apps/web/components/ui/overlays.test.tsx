/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription, DialogClose } from './dialog'
import { AlertDialog, AlertDialogTrigger, AlertDialogContent, AlertDialogHeader, AlertDialogFooter, AlertDialogTitle, AlertDialogDescription, AlertDialogAction, AlertDialogCancel } from './alert-dialog'
import { Sheet, SheetTrigger, SheetContent, SheetClose } from './sheet'
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuCheckboxItem } from './dropdown-menu'

afterEach(cleanup)

describe('Dialog', () => {
  it('renders trigger', () => {
    render(
      <Dialog>
        <DialogTrigger>Open</DialogTrigger>
      </Dialog>
    )
    expect(screen.getByText('Open')).toBeInTheDocument()
  })

  it('renders DialogTitle inside Dialog', () => {
    render(
      <Dialog open>
        <DialogTitle>Title</DialogTitle>
      </Dialog>
    )
    expect(screen.getByText('Title')).toBeInTheDocument()
  })

  it('renders DialogDescription inside Dialog', () => {
    render(
      <Dialog open>
        <DialogDescription>Desc</DialogDescription>
      </Dialog>
    )
    expect(screen.getByText('Desc')).toBeInTheDocument()
  })

  it('renders DialogHeader', () => {
    const { container } = render(<DialogHeader>Header</DialogHeader>)
    expect(container.textContent).toBe('Header')
  })

  it('renders DialogFooter', () => {
    const { container } = render(<DialogFooter>Footer</DialogFooter>)
    expect(container.textContent).toBe('Footer')
  })
})

describe('Sheet', () => {
  it('renders trigger', () => {
    render(
      <Sheet>
        <SheetTrigger>Open Sheet</SheetTrigger>
      </Sheet>
    )
    expect(screen.getByText('Open Sheet')).toBeInTheDocument()
  })
})

describe('AlertDialog', () => {
  it('renders trigger', () => {
    render(
      <AlertDialog>
        <AlertDialogTrigger>Delete</AlertDialogTrigger>
      </AlertDialog>
    )
    expect(screen.getByText('Delete')).toBeInTheDocument()
  })

  it('renders AlertDialogTitle inside AlertDialog', () => {
    render(
      <AlertDialog open>
        <AlertDialogTitle>Confirm</AlertDialogTitle>
      </AlertDialog>
    )
    expect(screen.getByText('Confirm')).toBeInTheDocument()
  })

  it('renders AlertDialogDescription inside AlertDialog', () => {
    render(
      <AlertDialog open>
        <AlertDialogDescription>Are you sure?</AlertDialogDescription>
      </AlertDialog>
    )
    expect(screen.getByText('Are you sure?')).toBeInTheDocument()
  })

  it('renders AlertDialogHeader', () => {
    const { container } = render(<AlertDialogHeader>Header</AlertDialogHeader>)
    expect(container.textContent).toBe('Header')
  })

  it('renders AlertDialogFooter', () => {
    const { container } = render(<AlertDialogFooter>Footer</AlertDialogFooter>)
    expect(container.textContent).toBe('Footer')
  })
})

describe('DropdownMenu', () => {
  it('renders trigger', () => {
    render(
      <DropdownMenu>
        <DropdownMenuTrigger>Menu</DropdownMenuTrigger>
      </DropdownMenu>
    )
    expect(screen.getByText('Menu')).toBeInTheDocument()
  })
})
