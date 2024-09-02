This highly robust VIO solution is really not quite ready for public viewing, 
but I'm tired of perfecting it and honestly... it provides really good odometry already anyway. 
So screw it. Here you go. Post any issues you have and I'll try to get around to answering/fixing them. 
I'm using an OAK-D DepthAI camera (OpenCV AI Kit - Depth) but you can write your own driver also. 

I can view the pose in Unity using this script (be sure to set the exe path to where you built the VIO): 

```csharp
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

public class VRCam : MonoBehaviour
{
    public string PathToExe;
    private Process VRCamProcess;

    private void Start()
    {
        Application.targetFrameRate = 120;

        VRCamProcess = new Process();
        ProcessStartInfo startInfo = new ProcessStartInfo();

        startInfo.UseShellExecute = false;
        startInfo.CreateNoWindow = false;
        startInfo.RedirectStandardOutput = true;

        startInfo.FileName = PathToExe;
        VRCamProcess.StartInfo = startInfo;

        VRCamProcess.EnableRaisingEvents = true;
        VRCamProcess.OutputDataReceived += VRCamOutputHandler;

        VRCamProcess.Start();
        VRCamProcess.BeginOutputReadLine();
    }

    private List<float> Pose = new List<float>();

    private void VRCamOutputHandler(object sender, DataReceivedEventArgs e)
    {
        string Msg = e.Data.ToString();
        if (!Msg.ToUpper().Contains("WAIT"))
        {
            Pose = Msg.Split(new string[] { "[", "]", ",", " " }, StringSplitOptions.RemoveEmptyEntries).Select
                (s => float.Parse(s)).ToList();
        }
    }

    private void LateUpdate()
    {
        if (Pose.Count == 7)
        {
            transform.localPosition = new Vector3(Pose[0], Pose[1], Pose[2]) / 100.0f;
            transform.localRotation = new Quaternion(Pose[3], Pose[4], Pose[5], Pose[6]);
        }
    }
}
```

I plan to add more docs and current known issues, but that is really, really boring. :P
