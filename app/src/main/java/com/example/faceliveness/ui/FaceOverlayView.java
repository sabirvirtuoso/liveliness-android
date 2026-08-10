package com.example.faceliveness.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.mlkit.vision.face.Face;

/**
 * Transparent overlay drawn on top of the camera preview.
 * Draws the detected face bounding box and a colored status ring.
 */
public class FaceOverlayView extends View {

    private Face face;
    private int previewWidth = 0;
    private int previewHeight = 0;

    private boolean isFaceDetected = false;
    private boolean isChallengePassed = false;
    private boolean isFaceTooFar = false;

    private final Paint boxPaint = new Paint();
    private final Paint ovalPaint = new Paint();
    private final Paint fillPaint = new Paint();
    private final Paint warningBoxPaint = new Paint();
    private final Paint warningTextPaint = new Paint();


    // Reused across onDraw() calls to avoid per-frame allocation (see onDraw / onSizeChanged)
    private final RectF ovalRect = new RectF();
    private final RectF scaledRect = new RectF();

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

    public void updateFace(@Nullable Face detectedFace, int width, int height) {
        face = detectedFace;
        previewWidth = width;
        previewHeight = height;
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

        // Draw face bounding box if face is detected
        if (face != null && previewWidth > 0 && previewHeight > 0) {
            float scaleX = getWidth() / (float) previewHeight; // rotated
            float scaleY = getHeight() / (float) previewWidth;

            Rect bounds = face.getBoundingBox();
            scaledRect.set(bounds.left * scaleX,
                    bounds.top * scaleY,
                    bounds.right * scaleX,
                    bounds.bottom * scaleY);

            canvas.drawRect(scaledRect, isFaceTooFar ? warningBoxPaint : boxPaint);

            if (isFaceTooFar) {
                canvas.drawText("Move closer", getWidth() / 2f, ovalRect.top - 24f, warningTextPaint);
            }
        }
    }
}
