@echo First Light

start cmd /k python mRun1.py C1 10000 1000

GOTO EndComment1
start cmd /k python mRun1.py C4 1000 100
start cmd /k python mRun1.py C1 500 200
start cmd /k python mRun1.py C1 1000 500
start cmd /k python mRun1.py C1 10000 1000
:EndComment1

@echo Process running in background

