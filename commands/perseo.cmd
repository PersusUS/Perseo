@echo off
REM Atajo para `perseo` desde cualquier terminal. Lo de verdad esta en
REM perseo.py, al lado: esto solo encuentra el interprete y le pasa los
REM argumentos tal cual.
REM
REM Para tenerlo a mano, esta carpeta va en el PATH del usuario.
setlocal
set "PERSEO_DIR=%~dp0"
python "%PERSEO_DIR%perseo.py" %*
