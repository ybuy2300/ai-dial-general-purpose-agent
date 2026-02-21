### Step 1.

Clean old data

```bash
$ rm -rf core-data 
```

```bash
rm -rf core-logs
```

Build docker
```
$ docker compose stop && docker compose up -d --build
```

Run uvicorn
```
$ uvicorn task.app:app --host 0.0.0.0 --port 5030 
```

Restarting

1st terminal
```
$ docker compose down  
$ rm -rf core-data/ core-logs/
$ docker compose stop && docker compose up -d --build
```

2nd terminal
```
(.venv) ~/IdeaProjects/DAIL/ai-dial-general-purpose-agent [main] $ uvicorn task.app:app --host 0.0.0.0 --port 5030
```