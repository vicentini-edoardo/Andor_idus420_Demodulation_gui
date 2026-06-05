' Conda environment name — edit if you use a different env.
Dim CONDA_ENV
CONDA_ENV = "py38"

Dim sh, fso, repoDir, python
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
python  = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\.conda\envs\" & CONDA_ENV & "\pythonw.exe"

sh.CurrentDirectory = repoDir & "\src"
sh.Run """" & python & """ -m idus420_gui", 0, False
