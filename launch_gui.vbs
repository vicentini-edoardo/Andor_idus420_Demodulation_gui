' Path to pythonw.exe in the conda environment — edit if the env moves.
Dim PYTHON
PYTHON = "C:\Users\neaspe\.conda\envs\py38\pythonw.exe"

Dim sh, fso, repoDir
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
repoDir = fso.GetParentFolderName(WScript.ScriptFullName)

sh.CurrentDirectory = repoDir & "\src"
sh.Run """" & PYTHON & """ -m idus420_gui", 0, False
