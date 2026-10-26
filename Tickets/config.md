# Agentic Kanban Configuration

## Team Members

| Name | Role | Type | Primary Responsibilities |
|---|---|---|---|
| **Björn** | Project Lead | Human | Direction, priorities, approvals |
| **Claude Code** | Coding Agent | Agent | Implementation, playtest runs, tickets |
| **Antigravity** | Coding Agent (IDE) | Agent | Implementation, playtest runs |
| **Devin** | Coding Agent (IDE) | Agent | Implementation |

## Runner Templates

```yaml
runners:
  backend:
    command: "python -m uvicorn main:app --host 127.0.0.1 --port 8000"
    working_directory: "backend"
  frontend:
    command: "npm run dev"
    working_directory: "frontend"
```

## Board Settings

- **Default Column**: `Backlog`
- **Completed Column**: `Completed`
- **Subfolder Filtering**: Enabled
- **Auto-Update Work Log**: Enabled
