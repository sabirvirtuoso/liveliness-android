package com.example.faceliveness.ui;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.example.faceliveness.databinding.ActivityResultBinding;

import java.util.Locale;

public class ResultActivity extends AppCompatActivity {

    private ActivityResultBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityResultBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        boolean passed = getIntent().getBooleanExtra(LivenessActivity.EXTRA_RESULT, false);

        if (passed) {
            binding.tvTitle.setText("Identity Verified");
            binding.tvSubtitle.setText("Liveness detection passed successfully.\nYou may now proceed.");
            // Here you would typically:
            // 1. Send the verified flag + session token to your backend
            // 2. Navigate to the protected profile/content area

            if (getIntent().hasExtra(LivenessActivity.EXTRA_MODEL_CONFIDENCE)) {
                float confidence = getIntent().getFloatExtra(LivenessActivity.EXTRA_MODEL_CONFIDENCE, 0f);
                binding.tvConfidenceScore.setText(String.format(Locale.US,
                        "AI model confidence: %.1f%%", confidence * 100));
                binding.tvConfidenceScore.setVisibility(android.view.View.VISIBLE);
            }
            // If the extra is absent, the model didn't return a usable score
            // (unavailable/failed — see MiniFasNetSpoofDetector's logs) and
            // tvConfidenceScore stays hidden, as set in the layout's default.
        }

        binding.btnBack.setOnClickListener(v -> finish());
    }
}
