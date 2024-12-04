using System;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace FingerTracker
{
    internal class Program
    {
        static void Main(string[] args)
        {
            VideoCapture capture = new VideoCapture(1);
            if (!capture.IsOpened)
            {
                Console.WriteLine("Camera not found.");
                return;
            }

            int thresholdValue = 150;

            Mat frame = new Mat();
            Mat gray = new Mat();
            Mat thresholded = new Mat();
            Mat morphOutput = new Mat();

            Console.WriteLine("Press ESC to exit. Use UP/DOWN arrows to adjust threshold.");
            while (true)
            {
                capture.Read(frame);
                if (frame.IsEmpty)
                    continue;

                CvInvoke.CvtColor(frame, gray, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(gray, thresholded, thresholdValue, 255, ThresholdType.Binary);

                Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(10, 10), new Point(-1, -1));
                CvInvoke.MorphologyEx(thresholded, morphOutput, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(morphOutput, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                    for (int i = 0; i < contours.Size; i++)
                    {
                        if (contours[i].Size < 5) continue;
                        RotatedRect ellipse = CvInvoke.FitEllipse(contours[i]);
                        CvInvoke.Ellipse(frame, ellipse, new MCvScalar(0, 255, 0), 2);

                        Point center = new Point((int)ellipse.Center.X, (int)ellipse.Center.Y);
                        CvInvoke.PutText(frame, $"{center.X},{center.Y}", center, FontFace.HersheySimplex, 0.5, new MCvScalar(255, 0, 0), 1);
                    }
                }

                CvInvoke.Imshow("Processed Frame", frame);
                CvInvoke.Imshow("Thresholded", thresholded);

                int key = CvInvoke.WaitKey(1);
                if (key == 27) break;
                if (key == 122)
                {
                    thresholdValue = Math.Min(thresholdValue + 5, 255);
                    Console.WriteLine($"Threshold Value: {thresholdValue}");
                }
                if (key == 115)
                {
                    thresholdValue = Math.Max(thresholdValue - 5, 0);
                    Console.WriteLine($"Threshold Value: {thresholdValue}");
                }

            }

            capture.Dispose();
        }
    }
}
