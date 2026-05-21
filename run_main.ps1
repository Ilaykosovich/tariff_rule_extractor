if (Test-Path ".\.venv\Scripts\python.exe") {
  & ".\.venv\Scripts\python.exe" @args
} else {
  python @args
}
