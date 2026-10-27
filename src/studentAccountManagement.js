export function accountStatusLabel(isActive) {
  if (isActive === null || isActive === undefined) return 'No account'
  return isActive ? 'Active' : 'Inactive'
}

export async function requestStudentAccountStatusChange(student, isActive, {
  confirmAction,
  request,
}) {
  if (!isActive) {
    const confirmed = confirmAction(
      `Deactivate ${student.name}'s student account? They will be signed out and unable to sign in until reactivated. Their academic records and history will be preserved.`,
    )
    if (!confirmed) return null
  }

  return request(`/students/${student.id}/account-status`, {
    method: 'PUT',
    body: { is_active: isActive },
  })
}
