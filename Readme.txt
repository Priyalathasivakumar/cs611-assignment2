https://github.com/Priyalathasivakumar/cs611-assignment2

Run:  docker-compose build && docker-compose up
UI:   http://localhost:8080  (admin / admin)
DAG:  loan_default_pipeline starts automatically and backfills Jan 2023 - Dec 2024 (about 40 min).
      Outputs: model_bank/, datamart/gold/model_predictions/, datamart/gold/model_monitoring/ (charts in charts/)
