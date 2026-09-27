package com.example.faceliveness.detection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;

import com.kbyai.facesdk.FaceBox;
import com.kbyai.facesdk.FaceSDK;

import java.io.ByteArrayOutputStream;
import java.util.List;

/**
 * Wraps the FaceSDK anti-spoofing model:
 * <p>
 * This is a SEPARATE layer of security from the heuristic checks elsewhere in
 * this package (FrameConsistencyChecker, PassiveAntiSpoofAnalyzer,
 * ScreenReplayDetector). Those are hand-tuned
 * signal-processing heuristics running continuously per-frame; this is a
 * trained CNN classifier run ONCE, on a single high-quality snapshot, after
 * a deliberate "stay still" moment at the end of a passed challenge sequence
 * (see LivenessViewModel's StillnessCheck state). Different mechanism,
 * different failure modes, genuinely additive as a second opinion.
 */
public class AISpoofDetector {

    private static final String TAG = "AISpoofDetector";

    private static final float LIVENESS_THRESHOLD = 0.7f;

    public static final class SpoofModelResult {
        public final boolean isSuspected;
        public final float confidence; // 0.0 = definitely real, 1.0 = definitely spoof
        public final String reason;

        public final Bitmap faceImageBitmap;

        SpoofModelResult(boolean isSuspected, float confidence, String reason, Bitmap faceImageBitmap) {
            this.isSuspected = isSuspected;
            this.confidence = confidence;
            this.reason = reason;
            this.faceImageBitmap = faceImageBitmap;
        }
    }

    public AISpoofDetector.SpoofModelResult classify(byte[] nv21, int width, int height, int lastRotationDegrees) {
        // 1. Initial basic validation
        if (nv21 == null || width <= 0 || height <= 0) {
            return new SpoofModelResult(true, 0f, "Invalid frame input: dimensions or buffer is null", null);
        }

        Bitmap bitmap = FaceSDK.yuv2Bitmap(nv21, width, height, 7);
        List<FaceBox> faceBoxes = FaceSDK.faceDetection(bitmap);

        float livenessSum = 0;
        int faceBoxesSize = faceBoxes.size();

        for (int i = 0; i < faceBoxesSize; i++) {
            livenessSum += faceBoxes.get(i).liveness;
        }

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        // Use lower quality for performance — we're doing pixel stats, not display
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
        byte[] imageBytes = out.toByteArray();
        Bitmap faceImageBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

        Matrix matrix = new Matrix();
        matrix.postRotate(lastRotationDegrees);
        Bitmap transformedBitmapImage = Bitmap.createBitmap(
                faceImageBitmap,
                0,
                0,
                faceImageBitmap.getWidth(),
                faceImageBitmap.getHeight(),
                matrix,
                true
        );

        if (bitmap != null) bitmap.recycle();
        faceImageBitmap.recycle();

        float averageLivenessConfidence = livenessSum / faceBoxesSize;
        if (averageLivenessConfidence < LIVENESS_THRESHOLD) {
            return new AISpoofDetector.SpoofModelResult(true, averageLivenessConfidence, " frames showed spoof attempts", transformedBitmapImage)
            ;
        } else {
            return new AISpoofDetector.SpoofModelResult(false, averageLivenessConfidence, "frames showed live user", transformedBitmapImage);
        }
    }
}
