Dim CONDA_ENV
CONDA_ENV = "py38"

Dim sh, fso, repoDir, srcDir, python
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
srcDir  = repoDir & "\src"

Dim base
base = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\.conda\envs\" & CONDA_ENV & "\"

If fso.FileExists(base & "pythonw.exe") Then
    python = base & "pythonw.exe"
Else
    python = base & "python.exe"
End If

sh.CurrentDirectory = srcDir
sh.Run """" & python & """ -m idus420_gui", 0, False
