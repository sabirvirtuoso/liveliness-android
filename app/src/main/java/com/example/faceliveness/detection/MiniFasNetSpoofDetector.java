package com.example.faceliveness.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;

/**
 * Wraps the MiniFASNet-V2 ONNX anti-spoofing model:
 * https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx
 *
 * This is a SEPARATE layer of security from the heuristic checks elsewhere in
 * this package (FrameConsistencyChecker, PassiveAntiSpoofAnalyzer,
 * ScreenReplayDetector, ExpressionDynamicsAnalyzer). Those are hand-tuned
 * signal-processing heuristics running continuously per-frame; this is a
 * trained CNN classifier run ONCE, on a single high-quality snapshot, after
 * a deliberate "stay still" moment at the end of a passed challenge sequence
 * (see LivenessViewModel's StillnessCheck state). Different mechanism,
 * different failure modes, genuinely additive as a second opinion.
 *
 * Model contract (from the model card, with one correction — see note below):
 *  - Input:  (1, 3, 80, 80) float32, BGR, range [0, 1] (pixel / 255)
 *  - Crop:   2.7x scale margin around the face bounding box CENTER (not just
 *            the box itself), resized to 80x80, no alignment warp
 *  - Output: 3 raw class scores. Card claims order [live, print, replay]
 *            (index 0 = live) — real-device testing shows index 2 is
 *            actually live for this export. See note below.
 *  - Liveness score = 1 - (p[print] + p[replay])
 *
 * NOTE ON THE OUTPUT: the model card describes the output as "already
 * softmax," but real-device runs of this ONNX export produced negative
 * values for live/print alongside a large positive replay value — which is
 * impossible for genuine softmax output (probabilities can't be negative,
 * and must sum to 1). That's raw logits, not softmax. classify() applies
 * softmax itself below rather than trusting the card, and logs loudly if a
 * future model update makes that unnecessary.
 *
 * NOTE ON CLASS INDEX ORDERING — index 2 is LIVE, confirmed empirically:
 * the model card states output order [live, print-attack, replay-attack]
 * (index 0 = live). A real-device test on a genuine face returned a
 * confident (0.99+) value at index 2 under BOTH an index-0-is-live and an
 * index-1-is-live assumption — i.e. the raw output itself consistently
 * concentrates at index 2 for real faces regardless of which label we
 * attach to it. IDX_LIVE=2 below reflects that. (An independent
 * reimplementation using the same upstream .pth weights, yakhyo/face-anti-spoofing,
 * treats index 1 as "Real" — that turned out not to hold for this specific
 * ONNX export, most likely because garciafido's conversion script reordered
 * the output layer relative to whatever convention that reimplementation
 * assumed. Real-device evidence from this actual pipeline overrides that.)
 *
 * IDX_PRINT/IDX_REPLAY (the remaining two indices) are still UNVERIFIED — a
 * real-face test can only confirm which index is live, since neither spoof
 * index is expected to fire on genuine input either way. If you need the
 * print-vs-replay distinction to be correct (e.g. for user-facing copy
 * naming the specific attack type), test with an actual printed photo and
 * an actual screen replay and check which index fires for each — nothing
 * currently depends on this sub-ordering (isLive only depends on IDX_LIVE).
 */
public class MiniFasNetSpoofDetector {

    private static final String TAG = "MiniFasNetSpoofDetector";

    private static final String MODEL_ASSET_PATH = "minifasnet_v2.onnx";
    private static final int INPUT_SIZE = 80;
    // Crop margin factor around the face bbox CENTER — matches the "2.7_80x80"
    // upstream filename convention documented on the model card. This is NOT
    // the same as just padding the bbox; the crop is recentered on the bbox
    // center and sized to CROP_SCALE times the bbox's own width/height.
    private static final float CROP_SCALE = 2.7f;

    // Class index ordering — see NOTE ON CLASS INDEX ORDERING in the class
    // doc above. IDX_LIVE=2 is confirmed by real-device testing.
    // IDX_PRINT/IDX_REPLAY's relative order is still an unverified
    // placeholder — it doesn't affect isLive, only which spoof TYPE gets
    // reported (predictedClass, printProb vs replayProb).
    private static final int IDX_PRINT = 0;
    private static final int IDX_REPLAY = 1;
    private static final int IDX_LIVE = 2;

    // Decision rule for isLive: argmax must be LIVE, AND liveProb must clear
    // this bar. Requiring a bar (not just argmax) avoids calling a genuinely
    // uncertain/near-even 3-way split "live" just because it edged out the
    // other two by a hair. PLACEHOLDER — calibrate against real paired
    // live/print/replay sessions on target device(s), same as every other
    // threshold in this codebase.
    private static final float LIVE_CONFIDENCE_THRESHOLD = 0.5f;

    // DEBUGGING ONLY — saves the exact crop and exact 80x80 model input to
    // app-private internal storage on every classify() call, so you can
    // visually confirm whether the crop region and/or the resize distortion
    // is the actual problem rather than guessing from preprocessing code
    // alone. Set to false once you're done debugging — this is a real disk
    // write on every inference and isn't meant to run in production.
    // Retrieve saved files via Android Studio's Device File Explorer at
    // /data/data/<applicationId>/files/spoof_debug_snapshots/, or:
    //   adb shell run-as <applicationId> ls files/spoof_debug_snapshots
    //   adb exec-out run-as <applicationId> cat files/spoof_debug_snapshots/<name> > local.png
    private static final boolean SAVE_DEBUG_SNAPSHOTS = true;
    private static final String DEBUG_SNAPSHOT_DIR_NAME = "spoof_debug_snapshots";
    public enum PredictedClass { LIVE, PRINT_ATTACK, REPLAY_ATTACK }

    /** Result of one inference. On failure, isAvailable() is false and every score is 0. */
    public static final class SpoofModelResult {
        public final boolean isAvailable;
        public final float liveProb;
        public final float printProb;
        public final float replayProb;
        public final float livenessScore; // 1 - (printProb + replayProb)
        public final PredictedClass predictedClass; // argmax of the three probabilities
        public final boolean isLive; // predictedClass == LIVE AND liveProb >= LIVE_CONFIDENCE_THRESHOLD
        public final String errorMessage; // non-null only when isAvailable == false

        private SpoofModelResult(boolean isAvailable, float liveProb, float printProb, float replayProb,
                                 float livenessScore, PredictedClass predictedClass, boolean isLive,
                                 String errorMessage) {
            this.isAvailable = isAvailable;
            this.liveProb = liveProb;
            this.printProb = printProb;
            this.replayProb = replayProb;
            this.livenessScore = livenessScore;
            this.predictedClass = predictedClass;
            this.isLive = isLive;
            this.errorMessage = errorMessage;
        }

        static SpoofModelResult success(float liveProb, float printProb, float replayProb) {
            float livenessScore = 1f - (printProb + replayProb);

            PredictedClass predicted;
            if (liveProb >= printProb && liveProb >= replayProb) {
                predicted = PredictedClass.LIVE;
            } else if (printProb >= replayProb) {
                predicted = PredictedClass.PRINT_ATTACK;
            } else {
                predicted = PredictedClass.REPLAY_ATTACK;
            }

            boolean isLive = predicted == PredictedClass.LIVE && liveProb >= LIVE_CONFIDENCE_THRESHOLD;

            return new SpoofModelResult(true, liveProb, printProb, replayProb, livenessScore,
                    predicted, isLive, null);
        }

        static SpoofModelResult failure(String errorMessage) {
            return new SpoofModelResult(false, 0f, 0f, 0f, 0f, null, false, errorMessage);
        }
    }

    private final OrtEnvironment ortEnvironment;
    private OrtSession session; // null if load failed — every public method degrades gracefully
    private final File debugSnapshotDir; // resolved once in the constructor; only this File is retained, not the Context itself

    /**
     * Loads the model from assets. Does NOT retain the Context afterward —
     * only the bytes it needs are read during construction. Never throws:
     * a load failure leaves this detector permanently in a "model
     * unavailable" state, and every call to classify() will return a
     * failure SpoofModelResult rather than crash the caller. This is a
     * deliberate choice — a missing/corrupt model file shouldn't take down
     * a liveness flow that already has heuristic checks providing coverage.
     */
    public MiniFasNetSpoofDetector(Context context) {
        OrtEnvironment env = null;
        OrtSession loadedSession = null;
        try {
            env = OrtEnvironment.getEnvironment();
            byte[] modelBytes = readAsset(context, MODEL_ASSET_PATH);
            loadedSession = env.createSession(modelBytes, new OrtSession.SessionOptions());
            Log.i(TAG, "MiniFASNet-V2 model loaded successfully from assets/" + MODEL_ASSET_PATH);
        } catch (IOException e) {
            Log.e(TAG, "Failed to read model asset '" + MODEL_ASSET_PATH + "' — " +
                    "was it added to app/src/main/assets/? See model card: " +
                    "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx", e);
        } catch (OrtException e) {
            Log.e(TAG, "ONNX Runtime failed to create a session from the model bytes " +
                    "(corrupt file, incompatible opset, or unsupported ABI?)", e);
        } catch (Exception e) {
            // Deliberately broad — this constructor must never throw and take
            // the whole liveness flow down with it; log whatever it is and
            // continue in "unavailable" state.
            Log.e(TAG, "Unexpected error loading MiniFASNet-V2 model", e);
        }
        this.ortEnvironment = env;
        this.session = loadedSession;

        // App-private internal storage — no runtime permission needed, works
        // on every API level. Only the resolved File is kept; context itself
        // is not retained past this constructor.
        File dir = new File(context.getFilesDir(), DEBUG_SNAPSHOT_DIR_NAME);
        if (SAVE_DEBUG_SNAPSHOTS && !dir.exists() && !dir.mkdirs()) {
            Log.w(TAG, "Could not create debug snapshot directory: " + dir.getAbsolutePath());
        }
        this.debugSnapshotDir = dir;
    }

    public boolean isModelLoaded() {
        return session != null;
    }

    /**
     * Runs one inference against a face snapshot.
     *
     * @param frame            the full captured frame — in the RAW SENSOR
     *                         coordinate space/orientation, not yet rotated
     *                         upright (see imageProxyToBitmap in FaceAnalyzer,
     *                         which does no rotation of its own)
     * @param faceBounds       ML Kit's face bounding box, in frame's own
     *                         (raw, unrotated) coordinate space
     * @param rotationDegrees  imageProxy.getImageInfo().getRotationDegrees() —
     *                         the clockwise rotation needed to make frame
     *                         upright. Applied to the CROP (after locating it
     *                         in frame's raw space, before resizing to 80x80),
     *                         not to the full frame — cheaper, and avoids
     *                         having to re-map faceBounds into rotated space.
     */
    public SpoofModelResult classify(Bitmap frame, Rect faceBounds, int rotationDegrees) {
        if (session == null) {
            return SpoofModelResult.failure("Model not loaded — see earlier log for why");
        }
        if (frame == null || faceBounds == null) {
            return SpoofModelResult.failure("Null frame or face bounds passed to classify()");
        }

        OnnxTensor inputTensor = null;
        try {
            Bitmap cropped = cropWithMargin(frame, faceBounds);

            // Rotate to upright BEFORE resizing. imageProxyToBitmap() (in
            // FaceAnalyzer) never rotates the raw sensor buffer — this is
            // the actual root cause of the "model doesn't discriminate at
            // all" result: a 90°-rotated face is wildly out-of-distribution
            // for a network trained on upright crops, real or spoofed alike.
            Bitmap upright = rotateBitmap(cropped, rotationDegrees);
            saveDebugBitmap(upright, "crop"); // post-rotation — this is what should look correctly oriented now

            Bitmap resized = Bitmap.createScaledBitmap(upright, INPUT_SIZE, INPUT_SIZE, true);
            saveDebugBitmap(resized, "input80"); // the exact bytes fed to the model
            if (resized != upright) upright.recycle();

            float[] chwData = bitmapToBgrChw(resized);
            resized.recycle();

            inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(chwData),
                    new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});

            String inputName = session.getInputNames().iterator().next();
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

            try (OrtSession.Result output = session.run(inputs)) {
                float[][] rawOutput = (float[][]) output.get(0).getValue();
                float[] logits = rawOutput[0];

                if (logits.length != 3) {
                    String msg = "Model output has " + logits.length + " classes, expected 3 " +
                            "[live, print, replay] — model card mismatch?";
                    Log.e(TAG, msg);
                    return SpoofModelResult.failure(msg);
                }

                // See class doc: real-device runs showed negative values here,
                // which is impossible for genuine softmax output — this is raw
                // logits despite what the model card claims. Softmax ourselves.
                float rawSum = logits[IDX_LIVE] + logits[IDX_PRINT] + logits[IDX_REPLAY];
                boolean looksAlreadySoftmaxed = Math.abs(rawSum - 1f) < 0.05f
                        && logits[IDX_LIVE] >= 0f && logits[IDX_PRINT] >= 0f && logits[IDX_REPLAY] >= 0f;
                if (looksAlreadySoftmaxed) {
                    Log.w(TAG, String.format(Locale.US,
                            "Raw model output already looks like valid softmax probabilities " +
                                    "(sum=%.4f, all non-negative) — applying softmax again anyway per " +
                                    "current code path. If this log appears consistently, the model/export " +
                                    "may have changed and this softmax step should be removed.",
                            rawSum));
                }

                float[] probs = softmax(logits);

                SpoofModelResult result = SpoofModelResult.success(
                        probs[IDX_LIVE], probs[IDX_PRINT], probs[IDX_REPLAY]);

                Log.i(TAG, String.format(
                        "Classification — live:%.4f print:%.4f replay:%.4f livenessScore:%.4f",
                        result.liveProb, result.printProb, result.replayProb, result.livenessScore));

                return result;
            }
        } catch (OrtException e) {
            Log.e(TAG, "ONNX Runtime inference failed", e);
            return SpoofModelResult.failure("Inference failed: " + e.getMessage());
        } catch (Exception e) {
            // Broad on purpose, same reasoning as the constructor — a bad
            // frame or an unexpected bitmap state must not crash the caller.
            Log.e(TAG, "Unexpected error during classify()", e);
            return SpoofModelResult.failure("Unexpected error: " + e.getMessage());
        } finally {
            if (inputTensor != null) inputTensor.close();
        }
    }

    /**
     * Crops a SQUARE region with a CROP_SCALE margin around the face bbox's
     * CENTER, using the LARGER of the bbox's width/height for both crop
     * dimensions. A face bbox is rarely square itself (commonly taller than
     * wide); cropping independently by width and height and then forcing
     * the result into a square 80x80 via createScaledBitmap non-uniformly
     * stretches the face — square-then-resize instead only ever scales
     * uniformly, so proportions are preserved.
     * Clamped to the source bitmap's bounds; if the face is near an edge,
     * the crop is naturally smaller than the ideal margin rather than
     * reading out of bounds.
     */
    private Bitmap cropWithMargin(Bitmap frame, Rect faceBounds) {
        float centerX = faceBounds.exactCenterX();
        float centerY = faceBounds.exactCenterY();
        float half = (Math.max(faceBounds.width(), faceBounds.height()) / 2f) * CROP_SCALE;

        int left = Math.max(0, Math.round(centerX - half));
        int top = Math.max(0, Math.round(centerY - half));
        int right = Math.min(frame.getWidth(), Math.round(centerX + half));
        int bottom = Math.min(frame.getHeight(), Math.round(centerY + half));

        int width = Math.max(1, right - left);
        int height = Math.max(1, bottom - top);

        return Bitmap.createBitmap(frame, left, top, width, height);
    }

    /**
     * Rotates a bitmap clockwise by rotationDegrees (as reported by
     * imageProxy.getImageInfo().getRotationDegrees()) to make it upright.
     * A no-op (returns the same instance) when rotationDegrees is 0.
     */
    private Bitmap rotateBitmap(Bitmap bitmap, int rotationDegrees) {
        if (rotationDegrees == 0) return bitmap;
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationDegrees);
        Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (rotated != bitmap) bitmap.recycle();
        return rotated;
    }

    /**
     * Debug-only: writes a bitmap to internal storage as PNG for visual
     * inspection. Best-effort — logs and swallows any failure rather than
     * letting a disk-write problem break classify() itself. Filename
     * includes a timestamp so repeated test captures don't overwrite each
     * other; "crop" vs "input80" in the name tells the two saved stages apart.
     */
    private void saveDebugBitmap(Bitmap bitmap, String label) {
        if (!SAVE_DEBUG_SNAPSHOTS) return;
        String filename = String.format(Locale.US, "%d_%s.png", System.currentTimeMillis(), label);
        File outFile = new File(debugSnapshotDir, filename);
        try (OutputStream out = new FileOutputStream(outFile)) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            Log.i(TAG, "Saved debug snapshot: " + outFile.getAbsolutePath()
                    + " (" + bitmap.getWidth() + "x" + bitmap.getHeight() + ")");
        } catch (IOException e) {
            Log.w(TAG, "Failed to save debug snapshot '" + filename + "': " + e.getMessage());
        }
    }

    /**
     * Converts a Bitmap (Android's ARGB_8888) to a [0,1]-normalized,
     * NCHW-ordered float array. Channel order (BGR vs RGB) is controlled by
     * USE_BGR_CHANNEL_ORDER — see that constant's doc if real-vs-fake still
     * doesn't separate after the crop/orientation fixes.
     */
    private float[] bitmapToBgrChw(Bitmap bitmap) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        int[] pixels = new int[w * h];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);

        float[] chw = new float[3 * w * h];
        int channelSize = w * h;

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            float r = ((pixel >> 16) & 0xFF) / 255f;
            float g = ((pixel >> 8) & 0xFF) / 255f;
            float b = (pixel & 0xFF) / 255f;

            chw[i] = b;                     // channel 0 = B
            chw[channelSize + i] = g;       // channel 1 = G
            chw[2 * channelSize + i] = r;   // channel 2 = R
        }
        return chw;
    }

    /**
     * Numerically-stable softmax: subtracts the max logit before exponentiating
     * to avoid overflow (standard trick — exp() of a large raw logit can
     * overflow float range; subtracting the max keeps every exponent <= 0
     * without changing the resulting ratios).
     */
    private float[] softmax(float[] logits) {
        float max = logits[0];
        for (float v : logits) {
            if (v > max) max = v;
        }
        float[] exp = new float[logits.length];
        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    private byte[] readAsset(Context context, String assetPath) throws IOException {
        try (InputStream is = context.getAssets().open(assetPath)) {
            byte[] buffer = new byte[is.available()];
            int totalRead = 0;
            int read;
            while (totalRead < buffer.length && (read = is.read(buffer, totalRead, buffer.length - totalRead)) != -1) {
                totalRead += read;
            }
            return buffer;
        }
    }

    /** Releases the ONNX session. Safe to call even if the model never loaded. */
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.w(TAG, "Error closing ONNX session", e);
            }
            session = null;
        }
    }
}
