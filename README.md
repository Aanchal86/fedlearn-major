
# FedLearn

Privacy-Preserving Distributed Federated Learning Platform

## Overview

FedLearn is a final-year B.Tech IT major project focused on
collaborative machine learning without centrally collecting
raw client training data.

## Technology Stack

- Python 3.11
- PyTorch
- FastAPI
- React + Vite
- SQLAlchemy
- Docker Compose

## Project Structure

- backend/ - FastAPI application
- frontend/ - React dashboard
- federated/ - Federated learning engine
- tests/ - Automated tests
- docs/ - Architecture and documentation

## Local Setup

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Docker Setup

From the project root:

```powershell
docker compose up --build
```

Backend: http://localhost:8000

Frontend: http://localhost:5173

API Docs: http://localhost:8000/docs

## Planned Features

- Custom PyTorch federated learning
- Federated Averaging (FedAvg)
- Adaptive aggregation
- IID and non-IID experiments
- Independent network-connected clients
- React experiment monitoring
- Reproducible model evaluation

## Project Status

Day 1 - Repository and environment setup.
```

### 4. Verify the repository

From the root directory, run:

```powershell
git status
```

Make sure `.venv/`, `node_modules/`, and dataset files are not being tracked.

Then check Docker configuration:

```powershell
docker compose config --quiet
```

If it returns without errors, your Compose configuration is valid.

### 5. Stage your Day 1 changes

Run:

```powershell
git add .
git status
```

Review the staged files. You should see your backend, frontend, federated package files, documentation, Dockerfiles, and configuration files.

### 6. Commit your work

Run:

```powershell
git commit -m "chore: complete Day 1 project foundation and Docker setup"
```

Then push your feature branch:

```powershell
git push -u origin setup/day1-foundation
```

### 7. Merge into `main`

Open your GitHub repository. You should see a **Compare & pull request** option for `setup/day1-foundation`.

Click it, create the pull request, review the changed files, and merge into `main` once the changes are verified.

Then update your local repository:

```powershell
git switch main
git pull origin main
```

## Day 1 completion checklist

{@body const items=["New repository created and cloned","Folder structure organized","Python 3.11 environment configured","Dependencies installed","FastAPI backend verified","React + Vite frontend verified","Docker Compose verified","Git ignore and README configured","Day 1 changes committed and pushed","Pull request merged into main"]}
{@body const [checked,setChecked]=DIL.useState(items.slice(0,7))}
<box border radius="xl" padding={3} gap={2}>
  <row align="center" justify="between">
    **Day 1 progress**
    <text color="secondary" weight="medium" tabularNums>{checked.length}/10</text>
  </row>
  <box background="surface-tertiary" height="6px" radius="full" clip>
    <box background="rgba(34,139,94,0.85)" width={`${checked.length*10}%`} height="100%" radius="full"/>
  </box>
  {#each items as item}
    <checkbox key={item} checked={checked.includes(item)} onChange={v=>setChecked(p=>v?[...p,item]:p.filter(x=>x!==item))}>{item}</checkbox>
  {/each}
</box>

Once you've pushed and merged everything, send me **"Day 1 completed"**.

Then we'll begin **Day 2 — Architecture and Interfaces**, where we'll design the actual federated learning workflow, define `ExperimentConfig`, `ClientUpdate`, and `RoundResult`, select the dataset and model, and establish how the PyTorch engine, FastAPI server, independent clients, and React dashboard will communicate.