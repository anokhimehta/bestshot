# CI/CD Notes

## Workflow behavior

On push to `main`, GitHub Actions in `.github/workflows/ci.yml`:

- builds and pushes service images to GHCR,
- optionally applies Kubernetes manifests to the cluster (requires secrets).

The deploy job currently uses `continue-on-error: true` temporarily, so tunnel/secret issues do not fail the whole workflow.

## Required deploy secrets

| Secret | Purpose |
|--------|---------|
| `KUBECONFIG_DATA` | Full kubeconfig from the node (same content as `~/.kube/config` after K3s install) |
| `K3S_SSH_HOST` | Floating IP or DNS of the node (for example `129.114.x.x`) |
| `K3S_SSH_USER` | SSH login (usually `cc`) |
| `K3S_SSH_KEY` | Private key PEM used to SSH into the node |

The deploy job SSH tunnels `127.0.0.1:6443` on the runner to `127.0.0.1:6443` on the Chameleon node.

## Faster runs and optional skips

- Parallel matrix builds for images
- Docker layer cache via `type=gha`
- PR builds do not push images
- Manual workflow supports skip build / skip deploy
- Concurrency cancels outdated runs on the same branch
