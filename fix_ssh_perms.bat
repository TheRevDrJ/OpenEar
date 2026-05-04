@echo off
:: Fix permissions on administrators_authorized_keys so Windows SSH accepts it
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant "SYSTEM:(F)" /grant "Administrators:(F)"
echo Done. Try SSH from Enterprise now.
pause
