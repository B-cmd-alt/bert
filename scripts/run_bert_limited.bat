@echo off
REM Run BERT training with low priority and resource limits
echo Starting BERT training with resource limits...

REM Set low priority and limit to 2 CPU cores
start /LOW /AFFINITY 3 /B python resource_limited_trainer.py

echo BERT training started with limited resources
echo Other processes will have priority over this training
pause