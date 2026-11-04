@echo off
set "PATH=%PATH:C:\ProgramData\anaconda3\Scripts\conda.exe;=%"
set "PATH=%PATH:;C:\ProgramData\anaconda3\Scripts\conda.exe=%"
set "PATH=%PATH:C:\ProgramData\anaconda3\Scripts\conda.exe=%"
cd /d "c:\RTM\IIT KGP\sem 7\BTP1_repo\thesis"
"C:\Users\DELL\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex
"C:\Users\DELL\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe" main
"C:\Users\DELL\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex
"C:\Users\DELL\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex
echo COMPILE_FINISHED_EXIT_CODE=%ERRORLEVEL%
