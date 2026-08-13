package com.example.faceliveness.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

/**
 * Wraps the MiniFASNet-V2 ONNX anti-spoofing model:
 * https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx
 *
 * This is a SEPARATE layer of security from the heuristic checks elsewhere in
 * this package (FrameConsistencyChecker, PassiveAntiSpoofAnalyzer,
 * ScreenReplayDetector). Those are hand-tuned
 * signal-processing heuristics running continuously per-frame; this is a
 * trained CNN classifier run ONCE, on a single high-quality snapshot, after
 * a deliberate "stay still" moment at the end of a passed challenge sequence
 * (see LivenessViewModel's StillnessCheck state). Different mechanism,
 * different failure modes, genuinely additive as a second opinion.
 *
 * Model contract (from the model card):
 *  - Input:  (1, 3, 80, 80) float32, BGR, range [0, 1] (pixel / 255)
 *  - Crop:   2.7x scale margin around the face bounding box CENTER (not just
 *            the box itself), resized to 80x80, no alignment warp
 *  - Output: 3-class softmax [live, print-attack, replay-attack]
 *  - Liveness score = 1 - (p[print] + p[replay])
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

    // Softmax output indices, per the model card.
    private static final int IDX_LIVE = 0;
    private static final int IDX_PRINT = 1;
    private static final int IDX_REPLAY = 2;

    /** Result of one inference. On failure, isAvailable() is false and every score is 0. */
    public static final class SpoofModelResult {
        public final boolean isAvailable;
        public final float liveProb;
        public final float printProb;
        public final float replayProb;
        public final float livenessScore; // 1 - (printProb + replayProb), per model card
        public final String errorMessage; // non-null only when isAvailable == false

        private SpoofModelResult(boolean isAvailable, float liveProb, float printProb,
                                 float replayProb, float livenessScore, String errorMessage) {
            this.isAvailable = isAvailable;
            this.liveProb = liveProb;
            this.printProb = printProb;
            this.replayProb = replayProb;
            this.livenessScore = livenessScore;
            this.errorMessage = errorMessage;
        }

        static SpoofModelResult success(float liveProb, float printProb, float replayProb) {
            float livenessScore = 1f - (printProb + replayProb);
            return new SpoofModelResult(true, liveProb, printProb, replayProb, livenessScore, null);
        }

        static SpoofModelResult failure(String errorMessage) {
            return new SpoofModelResult(false, 0f, 0f, 0f, 0f, errorMessage);
        }
    }

    private final OrtEnvironment ortEnvironment;
    private OrtSession session; // null if load failed — every public method degrades gracefully

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
    }

    public boolean isModelLoaded() {
        return session != null;
    }

    /**
     * Runs one inference against a face snapshot.
     *
     * @param frame       the full captured frame
     * @param faceBounds  ML Kit's face bounding box, in frame's own coordinate space
     */
    public SpoofModelResult classify(Bitmap frame, Rect faceBounds) {
        if (session == null) {
            return SpoofModelResult.failure("Model not loaded — see earlier log for why");
        }
        if (frame == null || faceBounds == null) {
            return SpoofModelResult.failure("Null frame or face bounds passed to classify()");
        }

        OnnxTensor inputTensor = null;
        try {
            Bitmap cropped = cropWithMargin(frame, faceBounds);
            Bitmap resized = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true);
            if (resized != cropped) cropped.recycle();

            float[] chwData = bitmapToBgrChw(resized);
            resized.recycle();

            inputTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(chwData),
                    new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});

            String inputName = session.getInputNames().iterator().next();
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);

            try (OrtSession.Result output = session.run(inputs)) {
                float[][] rawOutput = (float[][]) output.get(0).getValue();
                float[] probs = rawOutput[0];

                if (probs.length != 3) {
                    String msg = "Model output has " + probs.length + " classes, expected 3 " +
                            "[live, print, replay] — model card mismatch?";
                    Log.e(TAG, msg);
                    return SpoofModelResult.failure(msg);
                }

                // Model card states the output IS already softmax. Sanity-check
                // that assumption rather than silently trusting it — if it's off,
                // logging this is far more useful than a confidently wrong score.
                float sum = probs[IDX_LIVE] + probs[IDX_PRINT] + probs[IDX_REPLAY];
                if (Math.abs(sum - 1f) > 0.05f) {
                    Log.w(TAG, String.format(
                            "Model output doesn't sum to ~1.0 (sum=%.4f) — expected pre-softmaxed " +
                                    "output per model card; scores below may not be reliable probabilities.",
                            sum));
                }

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
     * Crops with a CROP_SCALE margin around the face bbox's CENTER — per the
     * model card, this is centered on the bbox, not just the bbox padded
     * outward asymmetrically. Clamped to the source bitmap's bounds; if the
     * face is near an edge, the crop is naturally smaller than the ideal
     * margin rather than reading out of bounds.
     */
    private Bitmap cropWithMargin(Bitmap frame, Rect faceBounds) {
        float centerX = faceBounds.exactCenterX();
        float centerY = faceBounds.exactCenterY();
        float halfW = (faceBounds.width() / 2f) * CROP_SCALE;
        float halfH = (faceBounds.height() / 2f) * CROP_SCALE;

        int left = Math.max(0, Math.round(centerX - halfW));
        int top = Math.max(0, Math.round(centerY - halfH));
        int right = Math.min(frame.getWidth(), Math.round(centerX + halfW));
        int bottom = Math.min(frame.getHeight(), Math.round(centerY + halfH));

        int width = Math.max(1, right - left);
        int height = Math.max(1, bottom - top);

        return Bitmap.createBitmap(frame, left, top, width, height);
    }

    /**
     * Converts a Bitmap (Android's ARGB_8888) to a BGR, [0,1]-normalized,
     * NCHW-ordered float array — exactly what the model expects per its card.
     * Note the deliberate B/G/R reorder: Android gives pixels as ARGB: this
     * writes them out as B, G, R channel PLANES (not interleaved), since NCHW
     * means all of channel 0, then all of channel 1, then all of channel 2.
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
