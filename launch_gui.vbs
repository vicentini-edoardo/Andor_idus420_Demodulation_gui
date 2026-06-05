' DEBUG version — uses python.exe with a visible console so errors are readable.
' Replace pythonw with pythonw and set window style back to 0 once working.

Dim CONDA_ENV
CONDA_ENV = "py38"

Dim sh, fso, repoDir, srcDir, python
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
srcDir  = repoDir & "\src"
python  = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\.conda\envs\" & CONDA_ENV & "\python.exe"

sh.CurrentDirectory = srcDir
' window style 1 = normal visible, True = wait for exit
sh.Run """" & python & """ -m idus420_gui", 1, True
