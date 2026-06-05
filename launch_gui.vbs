' Conda environment name — edit if you use a different env.
Dim CONDA_ENV
CONDA_ENV = "py38"

Dim sh, fso, repoDir, srcDir, python
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

repoDir = fso.GetParentFolderName(WScript.ScriptFullName)
srcDir  = repoDir & "\src"
python  = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\.conda\envs\" & CONDA_ENV & "\pythonw.exe"

' --- diagnostics: show resolved paths and bail if something is missing ---
If Not fso.FileExists(python) Then
    MsgBox "pythonw.exe not found:" & vbCrLf & python, 16, "launch_gui error"
    WScript.Quit 1
End If

If Not fso.FolderExists(srcDir) Then
    MsgBox "src folder not found:" & vbCrLf & srcDir, 16, "launch_gui error"
    WScript.Quit 1
End If

sh.CurrentDirectory = srcDir
sh.Run """" & python & """ -m idus420_gui", 0, False
