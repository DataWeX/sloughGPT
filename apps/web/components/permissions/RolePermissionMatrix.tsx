import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

const ROLE_COLORS: Record<string, string> = {
  owner: 'bg-warning/15 text-warning dark:bg-warning/10 dark:text-warning',
  admin: 'bg-info/15 text-info dark:bg-info/10 dark:text-info',
  user: 'bg-success/15 text-success dark:bg-success/10 dark:text-success',
  viewer: 'bg-muted text-muted-foreground',
}

const CATEGORY_LABELS: Record<string, string> = {
  model: 'Models',
  train: 'Training',
  chat: 'Chat',
  dataset: 'Datasets',
  knowledge: 'Knowledge',
  user: 'Users',
  tenant: 'Tenants',
  workspace: 'Workspaces',
  system: 'System',
}

export interface RolePermissionMatrixProps {
  roles: Record<string, { name: string; permissions: string[] }>
  allPermissions: Record<string, string[]>
}

export function RolePermissionMatrix({ roles, allPermissions }: RolePermissionMatrixProps) {
  const roleKeys = Object.keys(roles)

  return (
    <Card className="mb-6">
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Role Permissions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 pr-4 font-medium">Permission</th>
                {roleKeys.map(role => (
                  <th key={role} className="text-center py-2 px-2 font-medium">
                    <span className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${ROLE_COLORS[role] || 'bg-muted'}`}>
                      {role}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(allPermissions).map(([category, perms]) => (
                <>
                  <tr key={`cat-${category}`} className="border-b bg-muted/30">
                    <td colSpan={roleKeys.length + 1} className="py-1.5 font-medium text-muted-foreground">
                      {CATEGORY_LABELS[category] || category}
                    </td>
                  </tr>
                  {perms.map(perm => (
                    <tr key={perm} className="border-b border-border/50">
                      <td className="py-1.5 pr-4 text-muted-foreground">{perm}</td>
                      {roleKeys.map(role => (
                        <td key={role} className="text-center py-1.5 px-2">
                          {roles[role].permissions.includes(perm) ? (
                            <span className="text-success">✓</span>
                          ) : (
                            <span className="text-muted-foreground/40">—</span>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
