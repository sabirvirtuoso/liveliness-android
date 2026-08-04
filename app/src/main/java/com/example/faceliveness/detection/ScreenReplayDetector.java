package com.example.faceliveness.detection;

import android.graphics.Bitmap;
import android.util.Log;

import java.util.ArrayDeque;
import java.util.Locale;

/**
 * Detects screen/monitor replay attacks — i.e. the "face" being captured is
 * actually a video of a face playing on a phone or monitor held up to the
 * camera, not a real 3D face in front of it.
 *
 * PassiveAntiSpoofAnalyzer inspects whether the FACE CONTENT looks like real
 * skin (brightness texture, color balance). That check is structurally blind
 * to video replay: a video of a genuine face already contains genuine skin
 * texture and color, because the footage itself is real — only the CAPTURE
 * MEDIUM (a screen, recaptured by our camera) is fake.
 *
 * This class instead looks for artifacts of the capture medium itself:
 *
 * 1. MOIRE / HIGH-FREQUENCY ALIASING — recapturing one pixel grid (a display)
 *    with another pixel grid (the camera sensor) produces fine repeating
 *    interference patterns that don't occur when photographing real skin
 *    under normal lighting. Approximated here via discrete-Laplacian
 *    high-frequency energy over the FULL frame — not just the face crop,
 *    since this artifact is often clearer on the screen's bezel/background
 *    than on the recaptured face itself.
 *
 * 2. TEMPORAL BRIGHTNESS FLICKER — LCD/OLED panels refresh at a fixed rate
 *    and, combined with rolling-shutter capture, tend to imprint a periodic
 *    frame-to-frame brightness oscillation that stable ambient lighting on
 *    real skin does not produce. Approximated here via the variance of
 *    frame-to-frame brightness deltas over a rolling window.
 *
 * Note: this is a heuristic layer, not a trained ML model, same as
 * PassiveAntiSpoofAnalyzer — it raises suspicion scores, and should feed
 * into a combined verdict alongside challenge-response and the other
 * passive checks, not replace them. It also cannot substitute for hardware
 * depth sensing against a determined, real-time attacker (see caveats
 * discussed with the caller).
 */
public class ScreenReplayDetector {

    private static final String TAG = "ScreenReplayDetector";

    // Sampling stride across the full frame — full per-pixel processing is
    // too slow to run inside a live camera pipeline.
    private static final int SAMPLE_STEP = 6;

    // Laplacian energy above this = suspiciously fine repeating detail,
    // consistent with a recaptured pixel grid (moire/aliasing).
    private static final float MOIRE_ENERGY_THRESHOLD = 900f;

    // How many recent frames' mean brightness we track for flicker analysis.
    private static final int FLICKER_WINDOW = 12;

    // Variance of frame-to-frame brightness deltas above this = suspicious
    // oscillation, consistent with a display's refresh cycle beating against
    // the camera's rolling shutter.
    private static final float FLICKER_VARIANCE_THRESHOLD = 6.5f;

    // Number of frames to accumulate before making a session-level verdict.
    private static final int FRAMES_FOR_VERDICT = 15;

    // How many of the last N frames must be suspicious to flag.
    private static final int SUSPICIOUS_FRAME_THRESHOLD = 9;

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
    private final ArrayDeque<Float> recentBrightness = new ArrayDeque<>(FLICKER_WINDOW);

    /**
     * Analyze one full camera frame for screen-replay artifacts.
     * Unlike PassiveAntiSpoofAnalyzer, this intentionally looks at the WHOLE
     * frame rather than just the face crop.
     */
    public void analyzeFrame(Bitmap bitmap) {
        try {
            float moireEnergy = computeMoireEnergy(bitmap);
            float flickerVariance = updateAndComputeFlicker(bitmap);

            boolean isMoireSuspicious = moireEnergy > MOIRE_ENERGY_THRESHOLD;
            boolean isFlickerSuspicious = flickerVariance > FLICKER_VARIANCE_THRESHOLD;
            boolean isSuspicious = isMoireSuspicious || isFlickerSuspicious;

            String reason;
            if (isMoireSuspicious) {
                reason = String.format(Locale.US, "Moire/aliasing pattern detected (energy=%.1f)", moireEnergy);
            } else if (isFlickerSuspicious) {
                reason = String.format(Locale.US, "Screen-like brightness flicker (var=%.2f)", flickerVariance);
            } else {
                reason = "No screen-replay artifacts";
            }

            Log.d(TAG, String.format(Locale.US, "Frame - moire:%.1f flicker:%.2f suspicious:%b (%s)",
                    moireEnergy, flickerVariance, isSuspicious, reason));

            recentFrameResults.addLast(isSuspicious);
            if (recentFrameResults.size() > FRAMES_FOR_VERDICT) {
                recentFrameResults.removeFirst();
            }
        } catch (Exception e) {
            Log.e(TAG, "Screen-replay frame analysis failed", e);
        }
    }

    /**
     * Returns accumulated screen-replay signal based on recent frame history.
     * Call after analyzeFrame() to get a session-level verdict.
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
                    suspiciousCount + "/" + FRAMES_FOR_VERDICT + " frames showed screen-replay artifacts");
        } else {
            return new SpoofSignal(false, confidence, "No consistent screen-replay artifacts");
        }
    }

    public void reset() {
        recentFrameResults.clear();
        recentBrightness.clear();
    }

    // --- Internal Analysis --------------------------------------------------------

    /**
     * Approximates high-frequency energy across the full frame using a 3x3
     * discrete Laplacian kernel on sampled luminance values. Real skin under
     * normal lighting has smooth local gradients; a recaptured pixel grid
     * introduces fine repeating high-frequency structure that spikes this value.
     */
    private float computeMoireEnergy(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        long sumSq = 0L;
        int count = 0;

        // Walk a sampled grid, staying SAMPLE_STEP away from the border so the
        // 4-neighbor kernel never reads out of bounds.
        for (int y = SAMPLE_STEP; y < height - SAMPLE_STEP; y += SAMPLE_STEP) {
            for (int x = SAMPLE_STEP; x < width - SAMPLE_STEP; x += SAMPLE_STEP) {
                float center = luminance(bitmap.getPixel(x, y));
                float left = luminance(bitmap.getPixel(x - SAMPLE_STEP, y));
                float right = luminance(bitmap.getPixel(x + SAMPLE_STEP, y));
                float up = luminance(bitmap.getPixel(x, y - SAMPLE_STEP));
                float down = luminance(bitmap.getPixel(x, y + SAMPLE_STEP));

                // Discrete Laplacian: 4*center - sum(neighbors)
                float laplacian = 4 * center - (left + right + up + down);
                sumSq += (long) (laplacian * laplacian);
                count++;
            }
        }

        return count == 0 ? 0f : (float) sumSq / count;
    }

    /**
     * Tracks mean full-frame brightness across a short rolling window and
     * returns the variance of frame-to-frame brightness deltas — a proxy for
     * flicker/oscillation that stable ambient lighting on real skin does not produce.
     */
    private float updateAndComputeFlicker(Bitmap bitmap) {
        float meanBrightness = computeMeanBrightness(bitmap);

        recentBrightness.addLast(meanBrightness);
        if (recentBrightness.size() > FLICKER_WINDOW) {
            recentBrightness.removeFirst();
        }

        if (recentBrightness.size() < 3) return 0f;

        Float[] values = recentBrightness.toArray(new Float[0]);
        float[] deltas = new float[values.length - 1];
        float deltaSum = 0f;
        for (int i = 1; i < values.length; i++) {
            deltas[i - 1] = values[i] - values[i - 1];
            deltaSum += deltas[i - 1];
        }
        float deltaMean = deltaSum / deltas.length;

        float sqSum = 0f;
        for (float d : deltas) {
            float diff = d - deltaMean;
            sqSum += diff * diff;
        }
        return sqSum / deltas.length;
    }

    private float computeMeanBrightness(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        long total = 0L;
        int count = 0;

        for (int y = 0; y < height; y += SAMPLE_STEP) {
            for (int x = 0; x < width; x += SAMPLE_STEP) {
                total += (long) luminance(bitmap.getPixel(x, y));
                count++;
            }
        }
        return count == 0 ? 0f : (float) total / count;
    }

    private float luminance(int pixel) {
        int r = (pixel >> 16) & 0xFF;
        int g = (pixel >> 8) & 0xFF;
        int b = pixel & 0xFF;
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
