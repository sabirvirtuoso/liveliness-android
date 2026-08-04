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

    public LivenessResult(boolean passed, List<ChallengeType> completedChallenges) {
        this(passed, completedChallenges, null, null, null);
    }

    public LivenessResult(boolean passed,
                           List<ChallengeType> completedChallenges,
                           ChallengeType failedChallenge,
                           String failureReason,
                           AntiSpoofResult antiSpoofResult) {
        this.passed = passed;
        this.completedChallenges = completedChallenges;
        this.failedChallenge = failedChallenge;
        this.failureReason = failureReason;
        this.antiSpoofResult = antiSpoofResult;
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
}
