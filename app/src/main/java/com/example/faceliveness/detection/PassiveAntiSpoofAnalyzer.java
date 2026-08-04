package com.example.faceliveness.detection;

import android.graphics.Bitmap;
import android.graphics.Rect;
import android.util.Log;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Passive anti-spoof analyzer that runs frame-level pixel analysis
 * to detect screen replay and photo attacks without requiring any
 * user interaction.
 *
 * Analyzes two signals:
 *
 * 1. BRIGHTNESS UNIFORMITY — Real skin has natural local brightness
 *    variation (pores, texture, shadows). Screens and printed photos
 *    tend to produce unnaturally uniform brightness in the face region.
 *
 * 2. COLOR CHANNEL BALANCE — Real skin has a characteristic ratio
 *    between red, green, and blue channels. Screens and photos often
 *    skew these ratios due to display color profiles and ink rendering.
 *
 * Note: This is a heuristic layer, not a trained ML model. It raises
 * suspicion scores — the final decision combines this with
 * FrameConsistencyChecker and challenge-response results.
 */
public class PassiveAntiSpoofAnalyzer {

    private static final String TAG = "PassiveAntiSpoof";

    // Minimum brightness variance for real skin texture
    // Below this = suspiciously uniform = possible screen/photo
    private static final float MIN_BRIGHTNESS_VARIANCE = 180f;

    // Maximum brightness variance — very high variance can indicate
    // extreme lighting or glare artifacts from a screen
    private static final float MAX_BRIGHTNESS_VARIANCE = 3500f;

    // Real skin red channel should dominate over blue
    // Screens often show higher blue balance
    private static final float MIN_RED_BLUE_RATIO = 1.05f;

    // How many pixel samples to take from the face region
    // (sampling avoids processing every pixel for performance)
    private static final int SAMPLE_STEP = 4;

    // Number of frames to accumulate before making a verdict
    private static final int FRAMES_FOR_VERDICT = 15;

    // How many of the last N frames must be suspicious to flag
    private static final int SUSPICIOUS_FRAME_THRESHOLD = 10;

    public static final class FrameAnalysis {
        public final float brightnessVariance;
        public final float redBlueRatio;
        public final boolean isSuspicious;
        public final String reason;

        FrameAnalysis(float brightnessVariance, float redBlueRatio, boolean isSuspicious, String reason) {
            this.brightnessVariance = brightnessVariance;
            this.redBlueRatio = redBlueRatio;
            this.isSuspicious = isSuspicious;
            this.reason = reason;
        }
    }

    public static final class SpoofSignal {
        public final boolean isSuspected;
        public final float confidence; // 0.0 = definitely real, 1.0 = definitely spoof
        public final String reason;

        SpoofSignal(boolean isSuspected, float confidence, String reason) {
            this.isSuspected = isSuspected;
            this.confidence = confidence;
            this.reason = reason;
        }
    }

    private final ArrayDeque<Boolean> recentFrameResults = new ArrayDeque<>(FRAMES_FOR_VERDICT);

    /**
     * Analyze a single camera frame for screen/photo artifacts.
     * Pass the face bounding box from ML Kit to focus analysis on the face region.
     *
     * @param bitmap     Bitmap decoded from the raw CameraX frame
     * @param faceBounds Bounding box of the detected face in bitmap coordinates
     * @return FrameAnalysis result, or null if the frame can't be analyzed
     */
    public FrameAnalysis analyzeFrame(Bitmap bitmap, Rect faceBounds) {
        try {
            return analyzeFaceRegion(bitmap, faceBounds);
        } catch (Exception e) {
            Log.e(TAG, "Frame analysis failed", e);
            return null;
        }
    }

    /**
     * Returns accumulated spoof signal based on recent frame history.
     * Call after analyzeFrame to get a session-level verdict.
     */
    public SpoofSignal getSpoofSignal() {
        if (recentFrameResults.size() < FRAMES_FOR_VERDICT) {
            return new SpoofSignal(false, 0f,
                    "Collecting frames (" + recentFrameResults.size() + "/" + FRAMES_FOR_VERDICT + ")");
        }

        int suspiciousCount = 0;
        for (boolean b : recentFrameResults) {
            if (b) suspiciousCount++;
        }
        float confidence = (float) suspiciousCount / recentFrameResults.size();

        if (suspiciousCount >= SUSPICIOUS_FRAME_THRESHOLD) {
            return new SpoofSignal(true, confidence,
                    suspiciousCount + "/" + FRAMES_FOR_VERDICT + " frames showed screen/photo artifacts");
        } else {
            return new SpoofSignal(false, confidence, "Natural skin texture detected");
        }
    }

    public void reset() {
        recentFrameResults.clear();
    }

    // ─── Internal Analysis ───────────────────────────────────────────────────────

    private FrameAnalysis analyzeFaceRegion(Bitmap bitmap, Rect faceBounds) {
        // Clamp bounds to bitmap dimensions
        int left = Math.max(faceBounds.left, 0);
        int top = Math.max(faceBounds.top, 0);
        int right = Math.min(faceBounds.right, bitmap.getWidth());
        int bottom = Math.min(faceBounds.bottom, bitmap.getHeight());

        if (right <= left || bottom <= top) {
            return new FrameAnalysis(0f, 1f, false, "Invalid face bounds");
        }

        long totalBrightness = 0L;
        long totalR = 0L;
        long totalB = 0L;
        int pixelCount = 0;
        List<Float> brightnessValues = new ArrayList<>();

        // Sample pixels across the face region
        for (int y = top; y < bottom; y += SAMPLE_STEP) {
            for (int x = left; x < right; x += SAMPLE_STEP) {
                int pixel = bitmap.getPixel(x, y);
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                // Perceived brightness (luminance formula)
                float brightness = 0.299f * r + 0.587f * g + 0.114f * b;

                totalBrightness += (long) brightness;
                totalR += r;
                totalB += b;
                brightnessValues.add(brightness);
                pixelCount++;
            }
        }

        if (pixelCount == 0) return new FrameAnalysis(0f, 1f, false, "No pixels sampled");

        float meanBrightness = (float) totalBrightness / pixelCount;
        float meanR = (float) totalR / pixelCount;
        float meanB = (float) totalB / pixelCount;

        // Variance of brightness — real skin texture = higher variance
        float sqSum = 0f;
        for (float v : brightnessValues) {
            float diff = v - meanBrightness;
            sqSum += diff * diff;
        }
        float variance = sqSum / brightnessValues.size();

        // Red/blue channel ratio — real skin skews red
        float redBlueRatio = meanB > 0 ? meanR / meanB : 1f;

        // Determine if frame looks suspicious
        boolean isUniformBrightness = variance < MIN_BRIGHTNESS_VARIANCE;
        boolean isExtremelyHighVariance = variance > MAX_BRIGHTNESS_VARIANCE;
        boolean isScreenColorBalance = redBlueRatio < MIN_RED_BLUE_RATIO;

        boolean isSuspicious = isUniformBrightness || isScreenColorBalance;

        String reason;
        if (isUniformBrightness) {
            reason = String.format(Locale.US, "Brightness too uniform (var=%.1f)", variance);
        } else if (isScreenColorBalance) {
            reason = String.format(Locale.US, "Screen-like color balance (R/B=%.2f)", redBlueRatio);
        } else if (isExtremelyHighVariance) {
            reason = "Extreme brightness variance — possible glare";
        } else {
            reason = "Normal";
        }

        Log.d(TAG, String.format(Locale.US, "Frame — var:%.1f R/B:%.2f suspicious:%b (%s)",
                variance, redBlueRatio, isSuspicious, reason));

        // Record result in history
        recentFrameResults.addLast(isSuspicious);
        if (recentFrameResults.size() > FRAMES_FOR_VERDICT) {
            recentFrameResults.removeFirst();
        }

        return new FrameAnalysis(variance, redBlueRatio, isSuspicious, reason);
    }
}
