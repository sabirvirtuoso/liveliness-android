package com.example.faceliveness.ui;

import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

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

    private static final String TAG = "LivenessViewModel";
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

    // True once the post-challenge stillness check is satisfied — Activity
    // observes this and calls faceAnalyzer.requestLivenessSnapshot(). See
    // startStillnessCheck() below.
    private final MutableLiveData<Boolean> snapshotRequested = new MutableLiveData<>(false);

    public LiveData<Boolean> getSnapshotRequested() {
        return snapshotRequested;
    }

    private List<LivenessChallenge> challenges = new ArrayList<>();
    private int currentIndex = 0;
    private final List<ChallengeType> completedChallenges = new ArrayList<>();
    private CountDownTimer timer;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    // Post-challenge stillness check, before the model snapshot is captured.
    private static final long STILLNESS_DURATION_MS = 3000L;
    private static final long STILLNESS_TICK_MS = 200L;
    private long stillnessAccumulatedMs = 0L;
    private Runnable stillnessTickRunnable;

    /**
     * Starts a new liveness session with a fresh randomized challenge set.
     */
    public void startSession(int challengeCount) {
        challenges = ChallengeGenerator.generateChallenges(challengeCount);
        currentIndex = 0;
        completedChallenges.clear();
        spoofWarning.setValue(null);
        stillnessAccumulatedMs = 0L;
        snapshotRequested.setValue(false);
        if (stillnessTickRunnable != null) {
            mainHandler.removeCallbacks(stillnessTickRunnable);
        }
        startNextChallenge();
    }

    public void startSession() {
        startSession(3);
    }

    /**
     * Called by FaceAnalyzer when passive anti-spoof checks detect a spoof
     * attempt.
     *
     * During the StillnessCheck phase specifically, this IMMEDIATELY fails
     * the session — same SessionFailed/LivenessResult shape as a challenge
     * timeout (see startChallengeTimer()'s onFinish()), so it flows through
     * the exact same "show result, offer Retry" path with no other changes
     * needed. The stillness snapshot is the last, most security-sensitive
     * step before a pass, so a spoof signal here shouldn't wait for
     * anything else to resolve it.
     *
     * During any other phase (an active challenge), this only surfaces a
     * live warning (see FaceOverlayView.setSpoofWarning()) — the session is
     * NOT failed immediately there; challenge evaluation is independently
     * paused elsewhere (FaceAnalyzer's spoofAlreadyReported gating) while
     * the flag remains set, and an unmet challenge times out normally if it
     * can't clear before its own timer runs out.
     */
    public void onSpoofDetected(String reason) {
        spoofWarning.setValue(reason);

        ChallengeState currentState = challengeState.getValue();
        if (currentState instanceof ChallengeState.StillnessCheck) {
            cancelTimer(); // also stops the stillness tick loop — shares mainHandler
            challengeState.setValue(new ChallengeState.SessionFailed(
                    new LivenessResult(false, new ArrayList<>(completedChallenges), null,
                            "Spoof attempt detected: " + reason, null)
            ));
        }
    }

    /**
     * Called by FaceAnalyzer/LivenessActivity when face presence changes.
     * During the StillnessCheck phase specifically, LivenessActivity passes
     * a combined signal here — present, correctly distanced, AND mostly
     * within the guide oval (see FaceOverlayView.isFaceWithinOval()) — not
     * just raw ML Kit presence, since that combined signal is what
     * scheduleStillnessTick() reads to decide whether to accumulate or
     * reset the stillness timer.
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
                    startStillnessCheck();
                    // All challenges completed
//                    challengeState.setValue(new ChallengeState.SessionPassed(
//                            new LivenessResult(true, new ArrayList<>(completedChallenges))
//                    ));
                }
            }, 800);
        }
    }

    /**
     * Begins the post-challenge stillness phase: the user is asked to hold
     * still while a face stays continuously detected for STILLNESS_DURATION_MS.
     * Uses the same "pause, don't reset" pattern established elsewhere in this
     * app (see FaceAnalyzer's isTooFar/spoof gating) — a momentary detection
     * drop doesn't restart the count from zero, it just stops accumulating
     * until the face reappears. Once satisfied, snapshotRequested flips to
     * true; LivenessActivity observes that and calls
     * faceAnalyzer.requestLivenessSnapshot().
     */
    private void startStillnessCheck() {
        challengeState.setValue(ChallengeState.StillnessCheck.INSTANCE);
        stillnessAccumulatedMs = 0L;
        snapshotRequested.setValue(false);
        scheduleStillnessTick();
    }

    private void scheduleStillnessTick() {
        stillnessTickRunnable = () -> {
            if (Boolean.TRUE.equals(faceVisible.getValue())) {
                stillnessAccumulatedMs += STILLNESS_TICK_MS;
            } else {
                // RESET, not pause — see startStillnessCheck()'s doc for why
                // this phase intentionally differs from the pause pattern
                // used for isTooFar/spoof gating during active challenges.
                stillnessAccumulatedMs = 0L;
            }

            if (stillnessAccumulatedMs >= STILLNESS_DURATION_MS) {
                snapshotRequested.setValue(true);
                // Stop ticking — now waiting for onLivenessModelResult().
            } else {
                mainHandler.postDelayed(stillnessTickRunnable, STILLNESS_TICK_MS);
            }
        };
        mainHandler.postDelayed(stillnessTickRunnable, STILLNESS_TICK_MS);
    }

    /**
     * Called by LivenessActivity once FaceAnalyzer's one-shot model
     * classification completes (success or failure — see
     * MiniFasNetSpoofDetector.SpoofModelResult).
     *
     * DESIGN CHOICE: a model failure (missing/corrupt model file, inference
     * error, unsupported ABI, etc.) does NOT fail an otherwise-passed session
     * — the heuristic checks (FrameConsistencyChecker, PassiveAntiSpoofAnalyzer,
     * ScreenReplayDetector, ExpressionDynamicsAnalyzer) already gated this
     * session before it got here, so treating a model-layer outage as a hard
     * failure would make the whole flow newly fragile to something as simple
     * as forgetting to bundle the .onnx asset. The failure is logged (see
     * MiniFasNetSpoofDetector) and surfaced to the result screen as an absent
     * score rather than blocking the user. If you'd rather this be a hard
     * gate instead, this is the one place to change that.
     */
    public void onLivenessModelResult(boolean isSuspected, float livenessScore, String errorMessage) {
        ChallengeState state = challengeState.getValue();
        if (!(state instanceof ChallengeState.StillnessCheck)) {
            // Stale/duplicate callback (e.g. a race with a retry) — ignore.
            return;
        }

        Log.i(TAG, "The facesdk model result is: " + isSuspected + " with a score of: " + livenessScore);
        if (isSuspected) {
            challengeState.setValue(new ChallengeState.SessionFailed(
                    new LivenessResult(false, new ArrayList<>(), null, null, null, livenessScore)));
        } else {
            challengeState.setValue(new ChallengeState.SessionPassed(
                    new LivenessResult(true, new ArrayList<>(completedChallenges), null, null, null, livenessScore)
            ));
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
