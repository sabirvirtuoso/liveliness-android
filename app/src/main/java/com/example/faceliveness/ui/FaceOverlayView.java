package com.example.faceliveness.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.mlkit.vision.face.Face;

/**
 * Transparent overlay drawn on top of the camera preview.
 * Draws the detected face bounding box and a colored status ring.
 */
public class FaceOverlayView extends View {

    private static final String TAG = "FaceOverlayView";
    // Ratio of the 9 sampled face-box points (see isFaceWithinOval) required
    // to fall inside the guide oval. Requested as "3/9 or 4/9" — 4/9 chosen
    // as the default (slightly stricter); change to 3f/9f if 4/9 proves too
    // strict in practice. PLACEHOLDER — calibrate against real sessions.
    private static final float OVAL_OVERLAP_THRESHOLD = 3f / 9f;

    // This app always uses the front camera (see LivenessActivity.startCamera(),
    // CameraSelector.DEFAULT_FRONT_CAMERA). CameraX's Preview use case
    // auto-mirrors the ON-SCREEN display for a front camera (so the user sees
    // a normal "mirror" selfie view), but the raw ImageAnalysis buffer ML Kit
    // reads — and therefore face.getBoundingBox() — is NOT mirrored. Drawing
    // an unmirrored box over a mirrored preview puts it on the wrong side;
    // this corrects for that explicitly.
    private static final boolean MIRROR_FOR_FRONT_CAMERA = true;

    private Face face;
    private int previewWidth = 0;
    private int previewHeight = 0;

    // Named frameWidth/frameHeight (not previewWidth/previewHeight) deliberately —
    // these are the RAW SENSOR frame's dimensions, not any UI view's own pixel
    // size. See FaceAnalyzer.getLastFrameWidth()/getLastFrameHeight() doc.
    private int frameWidth = 0;
    private int frameHeight = 0;
    private int rotationDegrees = 0;

    private boolean isFaceDetected = false;
    private boolean isChallengePassed = false;
    private boolean isFaceTooFar = false;
    private String spoofWarningMessage = null;

    private final Paint boxPaint = new Paint();
    private final Paint ovalPaint = new Paint();
    private final Paint fillPaint = new Paint();
    private final Paint warningBoxPaint = new Paint();
    private final Paint warningTextPaint = new Paint();


    // Reused across onDraw() calls to avoid per-frame allocation (see onDraw / onSizeChanged)
    private final RectF ovalRect = new RectF();

    public FaceOverlayView(Context context) {
        this(context, null);
    }

    public FaceOverlayView(Context context, @Nullable AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public FaceOverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        boxPaint.setColor(Color.WHITE);

        ovalPaint.setStyle(Paint.Style.STROKE);
        ovalPaint.setStrokeWidth(6f);
        ovalPaint.setColor(Color.WHITE);

        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(Color.argb(30, 255, 255, 255));

        warningBoxPaint.setStyle(Paint.Style.STROKE);
        warningBoxPaint.setStrokeWidth(4f);
        warningBoxPaint.setColor(Color.parseColor("#F44336")); // red — overrides normal box color

        warningTextPaint.setStyle(Paint.Style.FILL);
        warningTextPaint.setColor(Color.parseColor("#F44336"));
        warningTextPaint.setTextSize(42f);
        warningTextPaint.setTextAlign(Paint.Align.CENTER);
        warningTextPaint.setFakeBoldText(true);
    }

    /**
     * Sets whether the face is too far away for reliable detection, as
     * determined authoritatively by FaceAnalyzer (raw sensor coordinate
     * space) — not computed locally here, so the UI and the actual
     * challenge/spoof enforcement can never disagree.
     */
    public void setTooFar(boolean tooFar) {
        isFaceTooFar = tooFar;
        postInvalidate();
    }

    public boolean isFaceTooFar() {
        return isFaceTooFar;
    }

    /**
     * Sets (or clears, if null) the spoof warning shown on the overlay.
     * Drawn in the same red warning style as the "too far" prompt, and takes
     * visual priority over it — see onDraw(). The session is NOT navigated
     * away on this; see LivenessViewModel.onSpoofDetected() for why.
     */
    public void setSpoofWarning(@Nullable String message) {
        spoofWarningMessage = message;
        postInvalidate();
    }

    /**
     * @param width           RAW SENSOR frame width (e.g. FaceAnalyzer.getLastFrameWidth()) —
     *                        NOT any UI view's pixel width.
     * @param height          RAW SENSOR frame height — same caveat.
     * @param rotationDegrees imageProxy.getImageInfo().getRotationDegrees() —
     *                        needed to correctly rotate face bounds into
     *                        view-space; see mapFrameRectToView().
     */
    public void updateFace(@Nullable Face detectedFace, int width, int height, int rotationDegrees) {
        face = detectedFace;
        frameWidth = width;
        frameHeight = height;
        this.rotationDegrees = rotationDegrees;
        isFaceDetected = detectedFace != null;
        postInvalidate();
    }

    public void setStatus(boolean detected, boolean passed) {
        isFaceDetected = detected;
        isChallengePassed = passed;

        int color;
        if (isChallengePassed) {
            color = Color.GREEN;
        } else if (isFaceDetected) {
            color = Color.parseColor("#4CAF50");
        } else {
            color = Color.parseColor("#F44336");
        }
        ovalPaint.setColor(color);
        boxPaint.setColor(color);
        postInvalidate();
    }

    public void setStatus(boolean detected) {
        setStatus(detected, false);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);

        // ovalRect only depends on view size, not on per-frame face data, so it's
        // recomputed here (rare: once at layout, or on rotation) instead of in onDraw.
        float centerX = w / 2f;
        float centerY = h / 2f;
        float ovalWidth = w * 0.6f;
        float ovalHeight = h * 0.45f;
        ovalRect.set(
                centerX - ovalWidth / 2,
                centerY - ovalHeight / 2,
                centerX + ovalWidth / 2,
                centerY + ovalHeight / 2
        );
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        // Draw guide oval — shows user where to position face
        canvas.drawOval(ovalRect, fillPaint);
        canvas.drawOval(ovalRect, ovalPaint);

        boolean showWarningStyle = spoofWarningMessage != null || isFaceTooFar;

        // Draw face bounding box if face is detected
        if (face != null && frameWidth > 0 && frameHeight > 0) {
            // Spoof warning takes priority over the "too far" prompt, and is
            // drawn regardless of whether a face box is currently present —
            // a spoof verdict can outlast a momentary detection drop.
            if (spoofWarningMessage != null) {
                canvas.drawText("Spoof Detected", getWidth() / 2f, ovalRect.top - 24f, warningTextPaint);
            } else if (isFaceTooFar) {
                canvas.drawText("Move closer", getWidth() / 2f, ovalRect.top - 24f, warningTextPaint);
            }
        }
    }

    /**
     * True when at least OVAL_OVERLAP_THRESHOLD (3/9 or 4/9, see that
     * constant) of the face's bounding box lies within the guide oval.
     * Approximated by sampling 9 points of the mapped face box (4 corners,
     * 4 edge midpoints, center) against a point-in-ellipse test, rather than
     * computing exact ellipse-rectangle overlap area — that has no simple
     * closed form, and this is a UX guide check, not a security boundary, so
     * a lightweight approximation is a reasonable tradeoff. Uses
     * mapFrameRectToView() — the SAME transform onDraw() uses — so this
     * stays consistent with what's actually drawn on screen.
     *
     * @param frameWidth      RAW SENSOR frame width — NOT any UI view's pixel width.
     * @param frameHeight     RAW SENSOR frame height — same caveat.
     * @param rotationDegrees imageProxy.getImageInfo().getRotationDegrees().
     */
    public boolean isFaceWithinOval(Face face, int frameWidth, int frameHeight, int rotationDegrees) {
        if (face == null || frameWidth <= 0 || frameHeight <= 0 || getWidth() <= 0 || getHeight() <= 0) {
            return false;
        }

        RectF r = mapFrameRectToView(face.getBoundingBox(), frameWidth, frameHeight, rotationDegrees);
        float midX = r.centerX();
        float midY = r.centerY();

        float[][] samplePoints = {
                {r.left, r.top}, {r.right, r.top}, {r.left, r.bottom}, {r.right, r.bottom},
                {midX, r.top}, {midX, r.bottom}, {r.left, midY}, {r.right, midY},
                {midX, midY}
        };

        int insideCount = 0;
        for (float[] p : samplePoints) {
            if (isPointInOval(p[0], p[1])) insideCount++;
        }
        boolean result = (float) insideCount / samplePoints.length >= OVAL_OVERLAP_THRESHOLD;

        Log.d(TAG, String.format(java.util.Locale.US,
                "isFaceWithinOval — faceRectView=[%.0f,%.0f,%.0f,%.0f] ovalRect=[%.0f,%.0f,%.0f,%.0f] " +
                        "insideCount=%d/9 threshold=%.2f result=%b",
                r.left, r.top, r.right, r.bottom, ovalRect.left, ovalRect.top, ovalRect.right, ovalRect.bottom,
                insideCount, OVAL_OVERLAP_THRESHOLD, result));

        return result;
    }

    /**
     * Maps a rect in RAW SENSOR coordinate space into view-pixel space,
     * accounting for:
     *  1. Sensor rotation (rotationDegrees) — a genuine geometric rotation of
     *     each corner (0/90/180/270), not an axis-swapped-scale approximation.
     *  2. Front-camera horizontal mirroring (MIRROR_FOR_FRONT_CAMERA).
     *  3. Uniform scaling from the now-correctly-oriented rotated space into
     *     actual view pixel dimensions.
     * All 4 corners are transformed individually and then bounded (min/max),
     * rather than transforming just two opposite corners — an axis-aligned
     * rect rotated by a multiple of 90° is still axis-aligned in the rotated
     * space either way, and this avoids sign mistakes in any one case.
     */
    private RectF mapFrameRectToView(Rect raw, int frameWidth, int frameHeight, int rotationDegrees) {
        float[][] corners = {
                {raw.left, raw.top}, {raw.right, raw.top}, {raw.left, raw.bottom}, {raw.right, raw.bottom}
        };

        int rotatedW, rotatedH;
        float[][] rotated = new float[4][2];
        int normalizedDegrees = ((rotationDegrees % 360) + 360) % 360;

        switch (normalizedDegrees) {
            case 90:
                rotatedW = frameHeight;
                rotatedH = frameWidth;
                for (int i = 0; i < 4; i++) {
                    rotated[i][0] = frameHeight - corners[i][1];
                    rotated[i][1] = corners[i][0];
                }
                break;
            case 180:
                rotatedW = frameWidth;
                rotatedH = frameHeight;
                for (int i = 0; i < 4; i++) {
                    rotated[i][0] = frameWidth - corners[i][0];
                    rotated[i][1] = frameHeight - corners[i][1];
                }
                break;
            case 270:
                rotatedW = frameHeight;
                rotatedH = frameWidth;
                for (int i = 0; i < 4; i++) {
                    rotated[i][0] = corners[i][1];
                    rotated[i][1] = frameWidth - corners[i][0];
                }
                break;
            default: // 0
                rotatedW = frameWidth;
                rotatedH = frameHeight;
                for (int i = 0; i < 4; i++) {
                    rotated[i][0] = corners[i][0];
                    rotated[i][1] = corners[i][1];
                }
        }

        if (MIRROR_FOR_FRONT_CAMERA) {
            for (int i = 0; i < 4; i++) {
                rotated[i][0] = rotatedW - rotated[i][0];
            }
        }

        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (float[] p : rotated) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }

        float scaleX = getWidth() / (float) rotatedW;
        float scaleY = getHeight() / (float) rotatedH;

        return new RectF(minX * scaleX, minY * scaleY, maxX * scaleX, maxY * scaleY);
    }

    private boolean isPointInOval(float x, float y) {
        float cx = ovalRect.centerX();
        float cy = ovalRect.centerY();
        float rx = ovalRect.width() / 2f;
        float ry = ovalRect.height() / 2f;
        if (rx <= 0 || ry <= 0) return false;
        float dx = (x - cx) / rx;
        float dy = (y - cy) / ry;
        return (dx * dx + dy * dy) <= 1f;
    }
}
