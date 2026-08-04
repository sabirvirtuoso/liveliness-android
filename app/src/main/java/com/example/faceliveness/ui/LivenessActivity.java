package com.example.faceliveness.ui;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.ViewModelProvider;

import com.example.faceliveness.databinding.ActivityLivenessBinding;
import com.example.faceliveness.detection.FaceAnalyzer;
import com.example.faceliveness.model.ChallengeState;
import com.example.faceliveness.model.LivenessResult;
import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LivenessActivity extends AppCompatActivity {

    private static final String TAG = "LivenessActivity";
    public static final String EXTRA_RESULT = "liveness_result";

    private ActivityLivenessBinding binding;
    private LivenessViewModel viewModel;

    private ExecutorService cameraExecutor;
    private FaceAnalyzer faceAnalyzer;
    private ImageAnalysis imageAnalysis;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityLivenessBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        viewModel = new ViewModelProvider(this).get(LivenessViewModel.class);

        cameraExecutor = Executors.newSingleThreadExecutor();

        setupFaceAnalyzer();
        startCamera();
        observeViewModel();

        // Small delay to let camera warm up before starting challenges
        binding.getRoot().postDelayed(() -> viewModel.startSession(3), 1500);

        binding.btnRetry.setOnClickListener(v -> {
            binding.layoutResult.setVisibility(View.GONE);
            binding.layoutChallenge.setVisibility(View.VISIBLE);
            faceAnalyzer.resetAntiSpoof();
            viewModel.startSession(3);
        });
    }

    private void setupFaceAnalyzer() {
        faceAnalyzer = new FaceAnalyzer(
                face -> runOnUiThread(() -> {
                    // In-flight ML Kit detections can complete and post here after
                    // onDestroy() has already run (e.g. rotation, back press mid-frame).
                    // binding/viewModel are still valid Java references at that point —
                    // they won't null out or throw on their own — so we must check the
                    // Activity's own destruction state explicitly before touching either.
                    if (isDestroyed() || isFinishing()) return;

                    PreviewView previewView = binding.previewView;
                    binding.faceOverlay.updateFace(face, previewView.getWidth(), previewView.getHeight());
                    binding.faceOverlay.setStatus(face != null);
                    viewModel.onFaceVisibilityChanged(face != null);
                    faceAnalyzer.setActiveChallenge(viewModel.currentChallenge());
                }),
                (type, success) -> {
                    if (isDestroyed() || isFinishing()) return;
                    
                    viewModel.onChallengeValidated(type, success);
                },
                reason -> viewModel.onSpoofDetected(reason)
        );
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());

                imageAnalysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                imageAnalysis.setAnalyzer(cameraExecutor, faceAnalyzer);

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_FRONT_CAMERA, // front camera for selfie liveness
                        preview,
                        imageAnalysis
                );
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Camera binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void observeViewModel() {
        viewModel.getChallengeState().observe(this, this::onChallengeStateChanged);
        viewModel.getTimeRemaining().observe(this, this::onTimeRemainingChanged);
        viewModel.getFaceVisible().observe(this, this::onFaceVisibleChanged);
        viewModel.getSpoofWarning().observe(this, this::onSpoofWarningChanged);
    }

    private void onChallengeStateChanged(ChallengeState state) {
        if (state instanceof ChallengeState.Idle) {
            binding.tvInstruction.setText("Preparing...");
            binding.tvProgress.setText("");

        } else if (state instanceof ChallengeState.InProgress) {
            ChallengeState.InProgress inProgress = (ChallengeState.InProgress) state;
            binding.layoutChallenge.setVisibility(View.VISIBLE);
            binding.layoutResult.setVisibility(View.GONE);
            binding.tvInstruction.setText(inProgress.getChallenge().getInstruction());
            binding.tvProgress.setText("Challenge " + (inProgress.getIndex() + 1) + " of " + inProgress.getTotal());
            binding.progressChallenge.setProgress(0);
            // Sync analyzer with current challenge
            faceAnalyzer.setActiveChallenge(inProgress.getChallenge());
            faceAnalyzer.resetChallengeState();

        } else if (state instanceof ChallengeState.ChallengeCompleted) {
            binding.tvInstruction.setText("✓ Done!");
            binding.faceOverlay.setStatus(true, true);
            faceAnalyzer.setActiveChallenge(null);

        } else if (state instanceof ChallengeState.SessionPassed) {
            showResult(((ChallengeState.SessionPassed) state).getResult());

        } else if (state instanceof ChallengeState.SessionFailed) {
            showResult(((ChallengeState.SessionFailed) state).getResult());
        }
    }

    private void onTimeRemainingChanged(int seconds) {
        binding.tvTimer.setText(seconds > 0 ? seconds + "s" : "");
        // Update circular progress
        ChallengeState state = viewModel.getChallengeState().getValue();
        if (state instanceof ChallengeState.InProgress) {
            ChallengeState.InProgress inProgress = (ChallengeState.InProgress) state;
            int total = (int) (inProgress.getChallenge().getTimeoutMillis() / 1000);
            int progress = total > 0 ? (int) (((float) (total - seconds) / total) * 100) : 0;
            binding.progressChallenge.setProgress(progress);
        }
    }

    private void onFaceVisibleChanged(boolean visible) {
        binding.tvFaceStatus.setText(visible ? "Face detected ✓" : "No face detected — look at camera");
        binding.tvFaceStatus.setTextColor(
                ContextCompat.getColor(this, visible ? android.R.color.holo_green_light : android.R.color.holo_red_light)
        );
    }

    private void onSpoofWarningChanged(String warning) {
        if (warning != null) {
            binding.tvFaceStatus.setText("⚠ " + warning);
            binding.tvFaceStatus.setTextColor(ContextCompat.getColor(this, android.R.color.holo_orange_dark));
        }
    }

    private void showResult(LivenessResult result) {
        faceAnalyzer.setActiveChallenge(null);
        binding.layoutChallenge.setVisibility(View.GONE);
        binding.layoutResult.setVisibility(View.VISIBLE);

        if (result.isPassed()) {
            binding.tvResultTitle.setText("Liveness Verified ✓");
            binding.tvResultMessage.setText("All " + result.getCompletedChallenges().size()
                    + " challenges passed.\nYou are verified as a live person.");
            binding.btnRetry.setVisibility(View.GONE);
            binding.btnContinue.setVisibility(View.VISIBLE);

            binding.btnContinue.setOnClickListener(v -> {
                Intent intent = new Intent(this, ResultActivity.class);
                intent.putExtra(EXTRA_RESULT, result.isPassed());
                startActivity(intent);
                finish();
            });
        } else {
            binding.tvResultTitle.setText("Verification Failed");
            binding.tvResultMessage.setText(
                    result.getFailureReason() != null ? result.getFailureReason() : "Liveness check failed. Please try again."
            );
            binding.btnRetry.setVisibility(View.VISIBLE);
            binding.btnContinue.setVisibility(View.GONE);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        faceAnalyzer.shutdown();
    }
}
