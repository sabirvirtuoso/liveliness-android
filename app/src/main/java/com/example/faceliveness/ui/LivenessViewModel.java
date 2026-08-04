package com.example.faceliveness.ui;

import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;

import androidx.annotation.Nullable;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.faceliveness.detection.ChallengeGenerator;
import com.example.faceliveness.model.ChallengeState;
import com.example.faceliveness.model.ChallengeType;
import com.example.faceliveness.model.LivenessChallenge;
import com.example.faceliveness.model.LivenessResult;

import java.util.ArrayList;
import java.util.List;

public class LivenessViewModel extends ViewModel {

    // Emits current challenge state to the UI
    private final MutableLiveData<ChallengeState> challengeState =
            new MutableLiveData<>(ChallengeState.Idle.INSTANCE);

    public LiveData<ChallengeState> getChallengeState() {
        return challengeState;
    }

    // Countdown remaining for current challenge (seconds)
    private final MutableLiveData<Integer> timeRemaining = new MutableLiveData<>(0);

    public LiveData<Integer> getTimeRemaining() {
        return timeRemaining;
    }

    // Whether a face is currently visible in frame
    private final MutableLiveData<Boolean> faceVisible = new MutableLiveData<>(false);

    public LiveData<Boolean> getFaceVisible() {
        return faceVisible;
    }

    // Anti-spoof warning message (null = no warning)
    private final MutableLiveData<String> spoofWarning = new MutableLiveData<>(null);

    public LiveData<String> getSpoofWarning() {
        return spoofWarning;
    }

    private List<LivenessChallenge> challenges = new ArrayList<>();
    private int currentIndex = 0;
    private final List<ChallengeType> completedChallenges = new ArrayList<>();
    private CountDownTimer timer;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    /**
     * Starts a new liveness session with a fresh randomized challenge set.
     */
    public void startSession(int challengeCount) {
        challenges = ChallengeGenerator.generateChallenges(challengeCount);
        currentIndex = 0;
        completedChallenges.clear();
        spoofWarning.setValue(null);
        startNextChallenge();
    }

    public void startSession() {
        startSession(3);
    }

    /**
     * Called by FaceAnalyzer when passive anti-spoof checks detect a spoof attempt.
     * Immediately fails the session — no challenge result can override this.
     */
    public void onSpoofDetected(String reason) {
        cancelTimer();
        spoofWarning.setValue(reason);
        challengeState.setValue(new ChallengeState.SessionFailed(
                new LivenessResult(false, new ArrayList<>(completedChallenges), null,
                        "Spoof attempt detected: " + reason, null)
        ));
    }

    /**
     * Called by FaceAnalyzer when it detects face presence changes.
     */
    public void onFaceVisibilityChanged(boolean visible) {
        faceVisible.setValue(visible);
    }

    /**
     * Called by FaceAnalyzer when a challenge condition is validated.
     */
    public void onChallengeValidated(ChallengeType type, boolean success) {
        ChallengeState state = challengeState.getValue();
        if (!(state instanceof ChallengeState.InProgress)) return;
        ChallengeState.InProgress inProgress = (ChallengeState.InProgress) state;
        if (inProgress.getChallenge().getType() != type) return;

        if (success) {
            cancelTimer();
            completedChallenges.add(type);

            challengeState.setValue(new ChallengeState.ChallengeCompleted(inProgress.getChallenge()));

            // Brief pause before next challenge
            mainHandler.postDelayed(() -> {
                currentIndex++;
                if (currentIndex < challenges.size()) {
                    startNextChallenge();
                } else {
                    // All challenges completed
                    challengeState.setValue(new ChallengeState.SessionPassed(
                            new LivenessResult(true, new ArrayList<>(completedChallenges))
                    ));
                }
            }, 800);
        }
    }

    /**
     * Returns the currently active challenge (for FaceAnalyzer to read).
     */
    @Nullable
    public LivenessChallenge currentChallenge() {
        ChallengeState state = challengeState.getValue();
        if (state instanceof ChallengeState.InProgress) {
            return ((ChallengeState.InProgress) state).getChallenge();
        }
        return null;
    }

    private void startNextChallenge() {
        LivenessChallenge challenge = challenges.get(currentIndex);
        challengeState.setValue(new ChallengeState.InProgress(challenge, currentIndex, challenges.size()));
        startChallengeTimer(challenge);
    }

    private void startChallengeTimer(LivenessChallenge challenge) {
        cancelTimer();
        final int totalSeconds = (int) (challenge.getTimeoutMillis() / 1000);
        timeRemaining.setValue(totalSeconds);

        timer = new CountDownTimer(challenge.getTimeoutMillis(), 1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                int remaining = (int) Math.round(millisUntilFinished / 1000.0);
                timeRemaining.setValue(remaining);
            }

            @Override
            public void onFinish() {
                timeRemaining.setValue(0);
                // Timeout — session failed
                challengeState.setValue(new ChallengeState.SessionFailed(
                        new LivenessResult(false, new ArrayList<>(completedChallenges), challenge.getType(),
                                "Time expired for: " + challenge.getInstruction(), null)
                ));
            }
        }.start();
    }

    private void cancelTimer() {
        if (timer != null) {
            timer.cancel();
            timer = null;
        }
        mainHandler.removeCallbacksAndMessages(null);
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        cancelTimer();
    }
}
