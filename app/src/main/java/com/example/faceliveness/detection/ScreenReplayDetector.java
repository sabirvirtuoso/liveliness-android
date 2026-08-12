package com.example.faceliveness.detection;

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
 * This class instead looks for artifacts of the capture medium itself, using
 * the RAW Y-plane (luma) bytes directly from CameraX — not a JPEG-recompressed
 * Bitmap. JPEG's 8x8 DCT blocking introduces its own structured high-frequency
 * energy into every frame regardless of content, which swamps the actual
 * moire signal if you analyze a JPEG-derived bitmap instead of the sensor data.
 *
 * 1. PERIODICITY (primary signal) — recapturing one pixel grid (a display)
 *    with another pixel grid (the camera sensor) produces a genuinely
 *    REPEATING interference pattern (moire), not just generic high-frequency
 *    detail. Raw high-frequency ENERGY alone (an earlier version of this
 *    class) can't separate that from ordinary skin texture/sensor noise,
 *    which also carries plenty of broadband high-frequency energy — in
 *    practice their energy ranges overlap heavily and shift with distance,
 *    since moire strength itself is non-monotonic with distance (it only
 *    appears within a "beat frequency" window between the two grids).
 *    Periodicity is the actual distinguishing property: a horizontal luma
 *    profile is detrended and autocorrelated: a strong secondary peak in the
 *    autocorrelation means the signal repeats every N pixels, which is what
 *    a pixel-grid interference pattern does and ordinary texture does not.
 *    Being a normalized correlation coefficient, this is far less sensitive
 *    to absolute brightness/gain/distance than a raw energy threshold.
 *
 * 2. FRAME-TO-FRAME BRIGHTNESS VOLATILITY — tracked via variance of brightness
 *    deltas over a rolling window. Honest caveat: at the sampling rate this
 *    runs at (roughly every 5th camera frame, ~6Hz effective), this CANNOT
 *    resolve true display refresh rates (50-120Hz) by Nyquist — a claim of
 *    "refresh-rate flicker detection" would be overstating what this measures.
 *    What it actually captures is generic brightness volatility: AE/AWB
 *    convergence, hand shake, mains-frequency lighting beat, and — usefully —
 *    scene-content brightness changes from a video actually playing (a real
 *    face under stable room light won't swing much frame to frame; a video's
 *    content can). Treat this as a weak supplementary signal, not a reliable
 *    standalone one.
 *
 * Raw Laplacian energy is still computed and logged for comparison, but is
 * NOT used in the suspicion decision — see the class discussion above for why.
 *
 * All thresholds below are placeholders and MUST be calibrated per target
 * device from real logged data (see analyzeFrame's Log.d output) — comparing
 * genuine-face sessions against actual replay-attack sessions on the same
 * hardware, since absolute values vary substantially by sensor, ISP, and
 * lighting and cannot be guessed correctly in the abstract.
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

    // PLACEHOLDER — calibrate from real Log.d output on target device(s).
    // Kept for logging/comparison only — see class doc for why raw energy
    // alone doesn't reliably separate real faces from screen replay.
    private static final float MOIRE_ENERGY_THRESHOLD = 900f;

    // --- Periodicity detection (primary signal) ---
    // Finer x-sampling than SAMPLE_STEP: moire periods can be as small as a
    // handful of pixels, so this needs denser sampling than the coarse energy scan.
    private static final int PROFILE_STEP = 2;
    // Box-filter window used to detrend the profile (remove slow lighting
    // gradients) before autocorrelation. Must be odd.
    private static final int DETREND_WINDOW = 21;
    // Lag range (in profile samples) to search for a periodic peak.
    private static final int MIN_LAG = 3;
    private static final int MAX_LAG = 60;
    // PLACEHOLDER — calibrate from real Log.d output. Normalized autocorrelation
    // (0..1) above this = a repeating pattern was found, consistent with
    // pixel-grid interference rather than ordinary texture/noise.
    private static final float PERIODICITY_THRESHOLD = 0.35f;

    // How many recent frames' mean brightness we track for flicker analysis.
    private static final int FLICKER_WINDOW = 12;

    // PLACEHOLDER — calibrate from real Log.d output on target device(s).
    // Variance of frame-to-frame brightness deltas above this = suspicious
    // volatility (see class doc caveat — this is not true refresh-rate detection).
    private static final float FLICKER_VARIANCE_THRESHOLD = 6.5f;

    // Number of frames to accumulate before making a session-level verdict.
    private static final int FRAMES_FOR_VERDICT = 15;

    // How many of the last N frames must be suspicious to flag.
    private static final int SUSPICIOUS_FRAME_THRESHOLD = 9;

    // Harsh/bright lighting (direct sunlight, strong backlight, a desk lamp
    // or monitor angled at the face) can sharpen ordinary face/hair texture
    // into something that autocorrelates more strongly than it should —
    // confirmed empirically, not just a theoretical risk. Frames this
    // overexposed are excluded from the periodicity vote entirely rather
    // than trusted either way — see analyzeFrame(). PLACEHOLDER — calibrate
    // against real bright-light sessions on target device(s).
    private static final int CLIP_LUMA_THRESHOLD = 250; // luma value considered "clipped"
    private static final float CLIPPED_FRACTION_THRESHOLD = 0.15f; // fraction of sampled pixels

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
     * Analyze one full camera frame for screen-replay artifacts, reading
     * directly from the raw Y-plane (luma) bytes CameraX hands us — no JPEG
     * round-trip, so no compression-block artifacts inflating the signal.
     *
     * @param yPlane    raw luma bytes from ImageProxy's plane[0]
     * @param width     image width in pixels
     * @param height    image height in pixels
     * @param rowStride bytes per row in yPlane (may exceed width due to padding)
     * @param pixelStride bytes per pixel in yPlane (usually 1 for Y planes)
     */
    public void analyzeFrame(byte[] yPlane, int width, int height, int rowStride, int pixelStride) {
        try {
            float clippedFraction = computeClippedFraction(yPlane, width, height, rowStride, pixelStride);

            if (clippedFraction > CLIPPED_FRACTION_THRESHOLD) {
                // Excluded entirely — not counted as suspicious OR as clean.
                // Same principle as FaceAnalyzer's isTooFar exclusion: a
                // frame we can't trust shouldn't get a vote either way,
                // rather than being trusted by default (which is what
                // happens if you only suppress the SUSPICIOUS verdict but
                // still let the frame count toward the total window size).
                Log.d(TAG, String.format(Locale.US,
                        "Frame skipped — overexposed (clipped=%.2f > %.2f), excluded from periodicity vote",
                        clippedFraction, CLIPPED_FRACTION_THRESHOLD));
                return;
            }

            float moireEnergy = computeMoireEnergy(yPlane, width, height, rowStride, pixelStride);
            float periodicityH = computePeriodicity(buildHorizontalProfile(yPlane, width, height, rowStride, pixelStride, PROFILE_STEP));
            float periodicityV = computePeriodicity(buildVerticalProfile(yPlane, width, height, rowStride, pixelStride, PROFILE_STEP));
            float periodicity = Math.max(periodicityH, periodicityV);
            float flickerVariance = updateAndComputeFlicker(yPlane, width, height, rowStride, pixelStride);

            boolean isPeriodicitySuspicious = periodicity > PERIODICITY_THRESHOLD;
            boolean isFlickerSuspicious = flickerVariance > FLICKER_VARIANCE_THRESHOLD;
            //boolean isSuspicious = isPeriodicitySuspicious || isFlickerSuspicious;
            boolean isSuspicious = isPeriodicitySuspicious;

            String reason;
            if (isPeriodicitySuspicious) {
                reason = String.format(Locale.US, "Repeating pixel-grid pattern detected (periodicity=%.2f, h=%.2f v=%.2f)",
                        periodicity, periodicityH, periodicityV);
            } else if (isFlickerSuspicious) {
                reason = String.format(Locale.US, "Screen-like brightness volatility (var=%.2f)", flickerVariance);
            } else {
                reason = "No screen-replay artifacts";
            }

            Log.d(TAG, String.format(Locale.US,
                    "Frame - periodicity:%.2f(h=%.2f,v=%.2f) energy:%.1f flicker:%.2f suspicious:%b (%s)",
                    periodicity, periodicityH, periodicityV, moireEnergy, flickerVariance, isSuspicious, reason));

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

    private int yAt(byte[] data, int x, int y, int rowStride, int pixelStride) {
        return data[y * rowStride + x * pixelStride] & 0xFF;
    }

    /**
     * Fraction of sampled pixels at or above CLIP_LUMA_THRESHOLD — a proxy
     * for overexposure (direct sunlight, strong backlight, harsh directional
     * light). Reuses the same coarse SAMPLE_STEP grid as computeMoireEnergy/
     * computeMeanBrightness rather than a full-resolution scan, since this
     * only needs to be a reasonable estimate, not an exact count, and is
     * checked before the more expensive periodicity computation runs.
     */
    private float computeClippedFraction(byte[] yPlane, int width, int height, int rowStride, int pixelStride) {
        int clipped = 0;
        int count = 0;
        for (int y = 0; y < height; y += SAMPLE_STEP) {
            for (int x = 0; x < width; x += SAMPLE_STEP) {
                if (yAt(yPlane, x, y, rowStride, pixelStride) >= CLIP_LUMA_THRESHOLD) {
                    clipped++;
                }
                count++;
            }
        }
        return count == 0 ? 0f : (float) clipped / count;
    }

    /**
     * Approximates high-frequency energy across the full frame using a 3x3
     * discrete Laplacian kernel on sampled raw luma values (no JPEG in the
     * path). Real skin under normal lighting has smooth local gradients; a
     * recaptured pixel grid introduces fine repeating high-frequency
     * structure that spikes this value.
     */
    private float computeMoireEnergy(byte[] yPlane, int width, int height, int rowStride, int pixelStride) {
        long sumSq = 0L;
        int count = 0;

        // Walk a sampled grid, staying SAMPLE_STEP away from the border so the
        // 4-neighbor kernel never reads out of bounds.
        for (int y = SAMPLE_STEP; y < height - SAMPLE_STEP; y += SAMPLE_STEP) {
            for (int x = SAMPLE_STEP; x < width - SAMPLE_STEP; x += SAMPLE_STEP) {
                float center = yAt(yPlane, x, y, rowStride, pixelStride);
                float left = yAt(yPlane, x - SAMPLE_STEP, y, rowStride, pixelStride);
                float right = yAt(yPlane, x + SAMPLE_STEP, y, rowStride, pixelStride);
                float up = yAt(yPlane, x, y - SAMPLE_STEP, rowStride, pixelStride);
                float down = yAt(yPlane, x, y + SAMPLE_STEP, rowStride, pixelStride);

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
     * returns the variance of frame-to-frame brightness deltas. See class
     * doc for why this is a weak, supplementary signal rather than true
     * refresh-rate detection at this sampling rate.
     */
    private float updateAndComputeFlicker(byte[] yPlane, int width, int height, int rowStride, int pixelStride) {
        float meanBrightness = computeMeanBrightness(yPlane, width, height, rowStride, pixelStride);

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

    private float computeMeanBrightness(byte[] yPlane, int width, int height, int rowStride, int pixelStride) {
        long total = 0L;
        int count = 0;

        for (int y = 0; y < height; y += SAMPLE_STEP) {
            for (int x = 0; x < width; x += SAMPLE_STEP) {
                total += yAt(yPlane, x, y, rowStride, pixelStride);
                count++;
            }
        }
        return count == 0 ? 0f : (float) total / count;
    }

    /**
     * Builds a 1D horizontal luma profile by averaging a handful of evenly
     * spaced rows column-by-column. Averaging several rows suppresses
     * per-row sensor noise while still preserving any horizontal periodicity
     * that's consistent across those rows (as pixel-grid interference would be).
     *
     * @param profileStep pixel spacing between profile samples — PROFILE_STEP_FINE
     *                     or PROFILE_STEP_COARSE, see class doc for why both exist.
     */
    private float[] buildHorizontalProfile(byte[] yPlane, int width, int height, int rowStride, int pixelStride, int profileStep) {
        int numSampleRows = 5;
        int profileLen = width / profileStep;
        float[] profile = new float[profileLen];

        for (int r = 0; r < numSampleRows; r++) {
            int y = (height * (r + 1)) / (numSampleRows + 1); // evenly spaced, avoiding edges
            for (int i = 0; i < profileLen; i++) {
                profile[i] += yAt(yPlane, i * profileStep, y, rowStride, pixelStride);
            }
        }
        for (int i = 0; i < profileLen; i++) {
            profile[i] /= numSampleRows;
        }
        return profile;
    }

    /** Same idea as buildHorizontalProfile, but sampling columns down the image instead. */
    private float[] buildVerticalProfile(byte[] yPlane, int width, int height, int rowStride, int pixelStride, int profileStep) {
        int numSampleCols = 5;
        int profileLen = height / profileStep;
        float[] profile = new float[profileLen];

        for (int c = 0; c < numSampleCols; c++) {
            int x = (width * (c + 1)) / (numSampleCols + 1);
            for (int i = 0; i < profileLen; i++) {
                profile[i] += yAt(yPlane, x, i * profileStep, rowStride, pixelStride);
            }
        }
        for (int i = 0; i < profileLen; i++) {
            profile[i] /= numSampleCols;
        }
        return profile;
    }
        /**
         * Detrends a 1D profile (removes the slow-varying lighting gradient via a
         * box-filter moving average) and returns the strongest normalized
         * autocorrelation coefficient found across MIN_LAG..MAX_LAG.
         *
         * A real, non-periodic signal (skin texture, sensor noise) decays toward
         * zero correlation as lag increases, with no standout peak. A genuinely
         * periodic signal (pixel-grid interference) produces a distinct peak at
         * the lag matching its period — that peak is what we're detecting, not
         * the signal's raw amplitude, which is why this is far less sensitive to
         * distance/brightness than the energy measure above.
         */
    private float computePeriodicity(float[] profile) {
        int n = profile.length;
        if (n < DETREND_WINDOW * 2) return 0f;

        // Detrend: subtract a centered moving average from each sample.
        float[] detrended = new float[n];
        int half = DETREND_WINDOW / 2;
        for (int i = 0; i < n; i++) {
            int lo = Math.max(0, i - half);
            int hi = Math.min(n - 1, i + half);
            float sum = 0f;
            for (int j = lo; j <= hi; j++) sum += profile[j];
            float localMean = sum / (hi - lo + 1);
            detrended[i] = profile[i] - localMean;
        }

        // Zero-lag autocorrelation (= variance of the detrended signal),
        // used to normalize every other lag into a -1..1 correlation coefficient.
        double zeroLag = 0.0;
        for (float v : detrended) zeroLag += (double) v * v;
        if (zeroLag < 1e-6) return 0f; // flat signal, nothing to correlate

        float maxCorr = 0f;
        int maxLag = Math.min(MAX_LAG, n / 2);
        for (int lag = MIN_LAG; lag < maxLag; lag++) {
            double sum = 0.0;
            for (int i = 0; i < n - lag; i++) {
                sum += (double) detrended[i] * detrended[i + lag];
            }
            float corr = (float) (sum / zeroLag);
            if (corr > maxCorr) maxCorr = corr;
        }
        return maxCorr;
    }

}
