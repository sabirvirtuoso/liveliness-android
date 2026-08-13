package com.example.faceliveness.model;

import java.util.List;

/**
 * Overall liveness session result.
 */
public class LivenessResult {

    private final boolean passed;
    private final List<ChallengeType> completedChallenges;
    private final ChallengeType failedChallenge;
    private final String failureReason;
    private final AntiSpoofResult antiSpoofResult;
    // MiniFASNet-V2 model's liveness score (1 - (p_print + p_replay)), 0..1.
    // Null when the model wasn't run, failed, or hasn't returned a result yet
    // — see MiniFasNetSpoofDetector.SpoofModelResult.
    private final Float livenessModelConfidence;

    public LivenessResult(boolean passed, List<ChallengeType> completedChallenges) {
        this(passed, completedChallenges, null, null, null, null);
    }

    public LivenessResult(boolean passed,
                          List<ChallengeType> completedChallenges,
                          ChallengeType failedChallenge,
                          String failureReason,
                          AntiSpoofResult antiSpoofResult) {
        this(passed, completedChallenges, failedChallenge, failureReason, antiSpoofResult, null);
    }

    public LivenessResult(boolean passed,
                           List<ChallengeType> completedChallenges,
                           ChallengeType failedChallenge,
                           String failureReason,
                           AntiSpoofResult antiSpoofResult,
                           Float livenessModelConfidence) {
        this.passed = passed;
        this.completedChallenges = completedChallenges;
        this.failedChallenge = failedChallenge;
        this.failureReason = failureReason;
        this.antiSpoofResult = antiSpoofResult;
        this.livenessModelConfidence = livenessModelConfidence;
    }

    public boolean isPassed() {
        return passed;
    }

    public List<ChallengeType> getCompletedChallenges() {
        return completedChallenges;
    }

    public ChallengeType getFailedChallenge() {
        return failedChallenge;
    }

    public String getFailureReason() {
        return failureReason;
    }

    public AntiSpoofResult getAntiSpoofResult() {
        return antiSpoofResult;
    }

    public Float getLivenessModelConfidence() {
        return livenessModelConfidence;
    }
}
