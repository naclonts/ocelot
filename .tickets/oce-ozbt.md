---
id: oce-ozbt
status: open
deps: []
links: []
created: 2026-04-04T02:36:24Z
type: chore
priority: 3
assignee: Nathan Clonts
tags: [mlflow, infra]
---
# Migrate MLflow from filesystem backend to SQLite

MLflow warns that filesystem tracking backend ('./mlruns') is deprecated as of Feb 2026. Migrate to 'sqlite:///mlflow.db' per https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store

