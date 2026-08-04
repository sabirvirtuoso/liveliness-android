package com.example.faceliveness.ui;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.example.faceliveness.databinding.ActivityResultBinding;

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
        }

        binding.btnBack.setOnClickListener(v -> finish());
    }
}
