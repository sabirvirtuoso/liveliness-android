package com.example.faceliveness.detection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.example.faceliveness.model.ChallengeType;
import com.example.faceliveness.model.LivenessChallenge;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * CameraX ImageAnalysis.Analyzer that runs ML Kit face detection on every frame,
 * evaluates the active liveness challenge, and feeds passive anti-spoof analyzers.
 *
 * Passive checks run on every frame silently:
 *  - FrameConsistencyChecker  → detects static photo (no natural movement)
 *  - PassiveAntiSpoofAnalyzer → detects screen/photo texture artifacts
 */
public class FaceAnalyzer implements ImageAnalysis.Analyzer {

    private static final String TAG = "FaceAnalyzer";

    // Challenge detection thresholds
    private static final float BLINK_THRESHOLD = 0.2f;
    private static final float SMILE_THRESHOLD = 0.7f;
    private static final float TURN_LEFT_THRESHOLD = -25f;
    private static final float TURN_RIGHT_THRESHOLD = 25f;
    private static final float NOD_DOWN_THRESHOLD = 15f;
    private static final float NOD_UP_THRESHOLD = -5f;
    private static final int FRAMES_REQUIRED = 3;

    // Run pixel analysis every N frames (not every frame — saves CPU)
    private static final int PIXEL_ANALYSIS_INTERVAL = 5;

    private final Consumer<Face> onFaceDetected;
    private final BiConsumer<ChallengeType, Boolean> onChallengeValidated;
    private final Consumer<String> onSpoofDetected;

    private final FaceDetector detector;

    // Passive anti-spoof modules
    public final FrameConsistencyChecker consistencyChecker = new FrameConsistencyChecker();
    public final PassiveAntiSpoofAnalyzer antiSpoofAnalyzer = new PassiveAntiSpoofAnalyzer();
    public final ScreenReplayDetector screenReplayDetector = new ScreenReplayDetector();

    // Current active challenge being evaluated
    private volatile LivenessChallenge activeChallenge;

    private int consecutiveFrames = 0;
    private NodPhase nodPhase = NodPhase.WAITING_FOR_DOWN;
    private int frameCount = 0;
    private boolean spoofAlreadyReported = false;

    public FaceAnalyzer(Consumer<Face> onFaceDetected,
                         BiConsumer<ChallengeType, Boolean> onChallengeValidated,
                         Consumer<String> onSpoofDetected) {
        this.onFaceDetected = onFaceDetected;
        this.onChallengeValidated = onChallengeValidated;
        this.onSpoofDetected = onSpoofDetected;

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE) // face landmark like 2d position of right eye, left eye, nose etc
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL) // smile, blink etc
                .setMinFaceSize(0.2f)
                .enableTracking() // track face and assign ID to each face over consecutive frames
                .build();

        detector = FaceDetection.getClient(options);
    }

    public LivenessChallenge getActiveChallenge() {
        return activeChallenge;
    }

    public void setActiveChallenge(LivenessChallenge activeChallenge) {
        this.activeChallenge = activeChallenge;
    }

    @OptIn(markerClass = ExperimentalGetImage.class) @Override
    public void analyze(@NonNull ImageProxy imageProxy) {
        if (imageProxy.getImage() == null) {
            imageProxy.close();
            return;
        }

        frameCount++;
        InputImage image = InputImage.fromMediaImage(imageProxy.getImage(), imageProxy.getImageInfo().getRotationDegrees());

        // Convert to bitmap for pixel analysis (only every N frames)
        final Bitmap bitmap = (frameCount % PIXEL_ANALYSIS_INTERVAL == 0) ? imageProxyToBitmap(imageProxy) : null;

        detector.process(image)
                .addOnSuccessListener(faces -> {
                    Face face = faces.isEmpty() ? null : faces.get(0);
                    onFaceDetected.accept(face);

                    if (face != null) {
                        // ── Passive checks (run every frame with face) ──────────────
                        consistencyChecker.addFrame(face);

                        // Pixel analysis on sampled frames only
                        if (bitmap != null) {
                            Rect scaledBounds = scaleBoundsToRotatedBitmap(face.getBoundingBox(), imageProxy, bitmap);
                            antiSpoofAnalyzer.analyzeFrame(bitmap, scaledBounds);
                            screenReplayDetector.analyzeFrame(bitmap);
                        }

                        // Check if passive checks have flagged a spoof
                        if (!spoofAlreadyReported) {
                            checkPassiveSpoofSignals();
                        }

                        // ── Challenge evaluation ────────────────────────────────────
                        LivenessChallenge challenge = activeChallenge;
                        if (challenge != null) {
                            evaluateChallenge(face, challenge);
                        } else {
                            consecutiveFrames = 0;
                        }
                    } else {
                        consecutiveFrames = 0;
                    }
                })
                .addOnFailureListener(e -> Log.e(TAG, "Face detection failed", e))
                .addOnCompleteListener(task -> {
                    if (bitmap != null) bitmap.recycle();
                    imageProxy.close();
                });
    }

    // ─── Passive Spoof Signal Aggregation ────────────────────────────────────────

    /**
     * Checks all passive signals and fires onSpoofDetected if any trigger.
     * Uses a two-signal approach: either static movement OR texture alone
     * can flag, but confidence is higher when both trigger together.
     */
    private void checkPassiveSpoofSignals() {
        // analyze() computes variance across the rolling frame window and updates
        // the internal still-verdict streak. It returns null until MIN_FRAMES_REQUIRED
        // frames have been collected — isSpoofDetected() naturally stays false until then.
        consistencyChecker.analyze();

        boolean staticDetected = consistencyChecker.isSpoofDetected();
        PassiveAntiSpoofAnalyzer.SpoofSignal textureSignal = antiSpoofAnalyzer.getSpoofSignal();
        ScreenReplayDetector.SpoofSignal replaySignal = screenReplayDetector.getSpoofSignal();

        if(replaySignal.isSuspected) {
            // Moire/flicker artifacts are a direct signal that the capture medium
            // itself is a screen — this catches video replay, which staticDetected
            // (movement-based) and textureSignal (skin-texture-based) both miss,
            // since a video of a real face has real movement and real skin texture.
            spoofAlreadyReported = true;
            onSpoofDetected.accept("Spoof detected: screen replay artifacts detected. Reason: " + replaySignal.reason);
        } else if (staticDetected && textureSignal.isSuspected) {
            spoofAlreadyReported = true;
            onSpoofDetected.accept(String.format(Locale.US,
                    "Spoof detected: face shows no natural movement AND screen/photo texture. Confidence: %.0f%%",
                    textureSignal.confidence * 100));
        } else if (staticDetected) {
            spoofAlreadyReported = true;
            onSpoofDetected.accept("Spoof detected: face is completely still — possible photo attack.");
        } else if (textureSignal.isSuspected && textureSignal.confidence > 0.8f) {
            // Only flag on texture alone at high confidence to reduce false positives
            spoofAlreadyReported = true;
            onSpoofDetected.accept("Spoof detected: screen or photo texture detected. Reason: " + textureSignal.reason);
        }
    }

    // ─── Challenge Evaluators ────────────────────────────────────────────────────

    private void evaluateChallenge(Face face, LivenessChallenge challenge) {
        boolean conditionMet;
        switch (challenge.getType()) {
            case BLINK:
                conditionMet = evaluateBlink(face);
                break;
            case TURN_LEFT:
                conditionMet = evaluateTurnLeft(face);
                break;
            case TURN_RIGHT:
                conditionMet = evaluateTurnRight(face);
                break;
            case SMILE:
                conditionMet = evaluateSmile(face);
                break;
            case NOD:
                conditionMet = evaluateNod(face);
                break;
            default:
                conditionMet = false;
        }

        if (conditionMet) {
            consecutiveFrames++;
            if (consecutiveFrames >= FRAMES_REQUIRED) {
                consecutiveFrames = 0;
                nodPhase = NodPhase.WAITING_FOR_DOWN;
                onChallengeValidated.accept(challenge.getType(), true);
            }
        } else {
            if (challenge.getType() != ChallengeType.NOD) {
                consecutiveFrames = 0;
            }
        }
    }

    private boolean evaluateBlink(Face face) {
        Float leftEyeOpen = face.getLeftEyeOpenProbability();
        Float rightEyeOpen = face.getRightEyeOpenProbability();
        if (leftEyeOpen == null || rightEyeOpen == null) return false;
        boolean isBlink = leftEyeOpen < BLINK_THRESHOLD && rightEyeOpen < BLINK_THRESHOLD;
        Log.d(TAG, "Blink — L:" + leftEyeOpen + " R:" + rightEyeOpen + " = " + isBlink);
        return isBlink;
    }

    private boolean evaluateTurnLeft(Face face) {
        float yaw = face.getHeadEulerAngleY();
        boolean result = yaw < TURN_LEFT_THRESHOLD;
        Log.d(TAG, "TurnLeft — yaw:" + yaw + " = " + result);
        return result;
    }

    private boolean evaluateTurnRight(Face face) {
        float yaw = face.getHeadEulerAngleY();
        boolean result = yaw > TURN_RIGHT_THRESHOLD;
        Log.d(TAG, "TurnRight — yaw:" + yaw + " = " + result);
        return result;
    }

    private boolean evaluateSmile(Face face) {
        Float smiling = face.getSmilingProbability();
        if (smiling == null) return false;
        boolean result = smiling > SMILE_THRESHOLD;
        Log.d(TAG, "Smile — prob:" + smiling);
        return result;
    }

    private boolean evaluateNod(Face face) {
        float pitch = face.getHeadEulerAngleX();
        Log.d(TAG, "Nod — pitch:" + pitch + " phase:" + nodPhase);
        switch (nodPhase) {
            case WAITING_FOR_DOWN:
                if (pitch > NOD_DOWN_THRESHOLD) nodPhase = NodPhase.WAITING_FOR_UP;
                return false;
            case WAITING_FOR_UP:
                return pitch < NOD_UP_THRESHOLD;
            default:
                return false;
        }
    }

    // ─── Bitmap Conversion ───────────────────────────────────────────────────────

    /**
     * Converts a YUV_420_888 ImageProxy to a Bitmap for pixel analysis.
     * Returns null silently if conversion fails — pixel analysis is optional.
     */
    @OptIn(markerClass = ExperimentalGetImage.class) private Bitmap imageProxyToBitmap(ImageProxy imageProxy) {
        try {
            Image image = imageProxy.getImage();
            if (image == null) return null;

            ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
            ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
            ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();

            byte[] nv21 = new byte[ySize + uSize + vSize];
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);

            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            // Use lower quality for performance — we're doing pixel stats, not display
            yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 60, out);
            byte[] imageBytes = out.toByteArray();
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        } catch (Exception e) {
            Log.w(TAG, "Bitmap conversion failed: " + e.getMessage());
            return null;
        }
    }

    /**
     * Scales ML Kit face bounds (which are in the original image space) to
     * match the rotated/scaled bitmap dimensions used for pixel analysis.
     */
    private Rect scaleBoundsToRotatedBitmap(Rect bounds, ImageProxy imageProxy, Bitmap bitmap) {
        float scaleX = (float) bitmap.getWidth() / imageProxy.getWidth();
        float scaleY = (float) bitmap.getHeight() / imageProxy.getHeight();
        return new Rect(
                (int) (bounds.left * scaleX),
                (int) (bounds.top * scaleY),
                (int) (bounds.right * scaleX),
                (int) (bounds.bottom * scaleY)
        );
    }

    // ─── Lifecycle ───────────────────────────────────────────────────────────────

    public void resetChallengeState() {
        consecutiveFrames = 0;
        nodPhase = NodPhase.WAITING_FOR_DOWN;
        activeChallenge = null;
    }

    public void resetAntiSpoof() {
        consistencyChecker.reset();
        antiSpoofAnalyzer.reset();
        spoofAlreadyReported = false;
        frameCount = 0;
    }

    public void shutdown() {
        detector.close();
    }

    private enum NodPhase {
        WAITING_FOR_DOWN,
        WAITING_FOR_UP
    }
}
