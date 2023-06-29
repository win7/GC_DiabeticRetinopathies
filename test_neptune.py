import neptune
import time

run = neptune.init_run(
    project="edwin.alvarez/Kaggle-GC",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjM0NzgxNy1lMDI1LTQxMzgtYTE2NS03OTIwOWY5NzgwNTkifQ==",
)  # your credentials

run["sys/tags"].add("best 🏆")

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(30):
    time.sleep(1)
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()