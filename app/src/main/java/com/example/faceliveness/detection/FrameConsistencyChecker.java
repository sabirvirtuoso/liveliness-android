package com.example.faceliveness.detection;

import android.util.Log;

import com.google.mlkit.vision.face.Face;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Detects static photo spoofing by verifying natural micro-movement is present.
 *
 * A real human face held in front of a camera always shows tiny natural
 * variation in head pose across frames — from breathing, micro-expressions,
 * and involuntary hand/body tremor. A printed photo held perfectly still
 * produces near-zero variance across all pose values.
 *
 * This check runs passively throughout the session, independent of challenges.
 */
public class FrameConsistencyChecker {

    private static final String TAG = "FrameConsistency";

    // Number of frames to collect before making a decision
    private static final int MIN_FRAMES_REQUIRED = 20;

    // Maximum frames to keep in the rolling window
    private static final int FRAME_WINDOW_SIZE = 40;

    // Minimum variance thresholds — below these across all axes = suspiciously still
    // Tune these per device; values are in degrees squared
    private static final float MIN_YAW_VARIANCE = 0.04f;
    private static final float MIN_PITCH_VARIANCE = 0.04f;
    private static final float MIN_EYE_VARIANCE = 0.0008f;

    // How many consecutive "still" verdicts before flagging as spoof
    private static final int STILL_VERDICT_THRESHOLD = 3;

    private static final class FrameSnapshot {
        final float eulerY; // yaw (left-right)
        final float eulerX; // pitch (up-down)
        final float leftEyeOpen;
        final float rightEyeOpen;

        FrameSnapshot(float eulerY, float eulerX, float leftEyeOpen, float rightEyeOpen) {
            this.eulerY = eulerY;
            this.eulerX = eulerX;
            this.leftEyeOpen = leftEyeOpen;
            this.rightEyeOpen = rightEyeOpen;
        }
    }

    public static final class ConsistencyResult {
        public final boolean hasNaturalMovement;
        public final boolean isSuspiciouslyStill;
        public final int framesAnalyzed;
        public final float yawVariance;
        public final float pitchVariance;
        public final float eyeVariance;

        ConsistencyResult(boolean hasNaturalMovement, boolean isSuspiciouslyStill, int framesAnalyzed,
                           float yawVariance, float pitchVariance, float eyeVariance) {
            this.hasNaturalMovement = hasNaturalMovement;
            this.isSuspiciouslyStill = isSuspiciouslyStill;
            this.framesAnalyzed = framesAnalyzed;
            this.yawVariance = yawVariance;
            this.pitchVariance = pitchVariance;
            this.eyeVariance = eyeVariance;
        }
    }

    private final ArrayDeque<FrameSnapshot> frameHistory = new ArrayDeque<>(FRAME_WINDOW_SIZE);
    private int stillVerdictCount = 0;

    /**
     * Add a new face detection result to the rolling window.
     * Call this on every frame where a face is detected.
     */
    public void addFrame(Face face) {
        Float leftEyeOpenProb = face.getLeftEyeOpenProbability();
        Float rightEyeOpenProb = face.getRightEyeOpenProbability();

        FrameSnapshot snapshot = new FrameSnapshot(
                face.getHeadEulerAngleY(),
                face.getHeadEulerAngleX(),
                leftEyeOpenProb != null ? leftEyeOpenProb : 1f,
                rightEyeOpenProb != null ? rightEyeOpenProb : 1f
        );

        frameHistory.addLast(snapshot);
        if (frameHistory.size() > FRAME_WINDOW_SIZE) {
            frameHistory.removeFirst();
        }
    }

    /**
     * Analyze the collected frames for natural movement.
     * Returns null if not enough frames have been collected yet.
     */
    public ConsistencyResult analyze() {
        if (frameHistory.size() < MIN_FRAMES_REQUIRED) return null;

        List<FrameSnapshot> frames = new ArrayList<>(frameHistory);

        float yawVariance = variance(frames, FrameSelector.YAW);
        float pitchVariance = variance(frames, FrameSelector.PITCH);
        float leftEyeVariance = variance(frames, FrameSelector.LEFT_EYE);
        float rightEyeVariance = variance(frames, FrameSelector.RIGHT_EYE);
        float eyeVariance = (leftEyeVariance + rightEyeVariance) / 2f;

        // Natural movement requires at least one axis to show sufficient variance.
        // A photo is still on ALL axes simultaneously — that's the key signal.
        boolean hasNaturalMovement = yawVariance > MIN_YAW_VARIANCE
                || pitchVariance > MIN_PITCH_VARIANCE
                || eyeVariance > MIN_EYE_VARIANCE;

        boolean isSuspiciouslyStill = !hasNaturalMovement;

        if (isSuspiciouslyStill) {
            stillVerdictCount++;
        } else {
            stillVerdictCount = 0;
        }

        Log.d(TAG, String.format(Locale.US,
                "Consistency — yawVar:%.4f pitchVar:%.4f eyeVar:%.4f naturalMovement:%b",
                yawVariance, pitchVariance, eyeVariance, hasNaturalMovement));

        return new ConsistencyResult(hasNaturalMovement, isSuspiciouslyStill, frames.size(),
                yawVariance, pitchVariance, eyeVariance);
    }

    /**
     * Returns true if the face has been suspiciously still for long enough
     * to confidently flag as a photo spoof attempt.
     */
    public boolean isSpoofDetected() {
        return stillVerdictCount >= STILL_VERDICT_THRESHOLD;
    }

    public void reset() {
        frameHistory.clear();
        stillVerdictCount = 0;
    }

    // ─── Math helpers ────────────────────────────────────────────────────────────

    private enum FrameSelector {
        YAW, PITCH, LEFT_EYE, RIGHT_EYE;

        float select(FrameSnapshot f) {
            switch (this) {
                case YAW:
                    return f.eulerY;
                case PITCH:
                    return f.eulerX;
                case LEFT_EYE:
                    return f.leftEyeOpen;
                case RIGHT_EYE:
                    return f.rightEyeOpen;
                default:
                    return 0f;
            }
        }
    }

    private static float variance(List<FrameSnapshot> frames, FrameSelector selector) {
        if (frames.size() < 2) return 0f;
        float sum = 0f;
        for (FrameSnapshot f : frames) sum += selector.select(f);
        float mean = sum / frames.size();
        float sqSum = 0f;
        for (FrameSnapshot f : frames) {
            float diff = selector.select(f) - mean;
            sqSum += diff * diff;
        }
        return sqSum / frames.size();
    }
}
