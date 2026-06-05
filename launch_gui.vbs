Dim CONDA_ENV
CONDA_ENV = "py38"

Dim sh, fso, repoDir, srcDir, python
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
srcDir  = repoDir & "\src"
python  = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\.conda\envs\" & CONDA_ENV & "\pythonw.exe"

sh.CurrentDirectory = srcDir
sh.Run """" & python & """ -m idus420_gui", 0, False
