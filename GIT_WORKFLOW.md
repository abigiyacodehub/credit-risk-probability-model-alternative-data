# Git and GitHub Best Practices for This Project

Use task-specific branches and descriptive commits so each rubric task is reviewable.

Suggested branch flow:

```bash
git checkout -b task-1-business-understanding
git add README.md
git commit -m "Document credit scoring business understanding"

git checkout -b task-2-eda main
git add notebooks/eda.ipynb
git commit -m "Add executed Xente EDA notebook"

git checkout -b task-3-feature-engineering main
git add src/data_processing.py tests/test_data_processing.py
git commit -m "Add transaction feature engineering pipeline"

git checkout -b task-4-proxy-target main
git add src/data_processing.py
git commit -m "Create RFM proxy high risk target"

git checkout -b task-5-model-training main
git add src/train.py models/metrics.json
git commit -m "Train and track credit risk PD models"

git checkout -b task-6-deployment-ci main
git add src/api Dockerfile docker-compose.yml .github/workflows/ci.yml
git commit -m "Add FastAPI deployment and CI workflow"
```

Open a pull request for each branch, wait for the CI workflow to pass, then merge into `main`.
