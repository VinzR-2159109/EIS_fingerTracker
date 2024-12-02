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
            // Initialize the video capture (use webcam or video source)
            VideoCapture capture = new VideoCapture(0);
            if (!capture.IsOpened)
            {
                Console.WriteLine("Camera not found.");
                return;
            }

            // Define threshold value (adjustable via user input)
            int thresholdValue = 127;

            Mat frame = new Mat();
            Mat gray = new Mat();
            Mat thresholded = new Mat();
            Mat morphOutput = new Mat();

            Console.WriteLine("Press ESC to exit.");
            while (true)
            {
                // Capture frame from the camera
                capture.Read(frame);
                if (frame.IsEmpty)
                    continue;

                // Convert frame to grayscale
                CvInvoke.CvtColor(frame, gray, ColorConversion.Bgr2Gray);

                // Apply thresholding
                CvInvoke.Threshold(gray, thresholded, thresholdValue, 255, ThresholdType.Binary);

                // Perform morphological "open" operation to reduce noise
                Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(10, 10), new Point(-1, -1));
                CvInvoke.MorphologyEx(thresholded, morphOutput, MorphOp.Open, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

                // Find contours
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(morphOutput, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                    for (int i = 0; i < contours.Size; i++)
                    {
                        // Fit ellipse to the contour
                        if (contours[i].Size < 5) continue; // At least 5 points are needed to fit an ellipse
                        RotatedRect ellipse = CvInvoke.FitEllipse(contours[i]);

                        // Draw the ellipse
                        CvInvoke.Ellipse(frame, ellipse, new MCvScalar(0, 255, 0), 2);

                        // Draw (x, y) coordinates
                        Point center = new Point((int)ellipse.Center.X, (int)ellipse.Center.Y);
                        CvInvoke.PutText(frame, $"{center.X},{center.Y}", center, FontFace.HersheySimplex, 0.5, new MCvScalar(255, 0, 0), 1);
                    }
                }

                // Display the processed frame
                CvInvoke.Imshow("Processed Frame", frame);
                CvInvoke.Imshow("Thresholded", thresholded);

                // Exit on key press
                if (CvInvoke.WaitKey(1) == 27) break; // Esc key
            }

            capture.Dispose();
        }
    }
}
