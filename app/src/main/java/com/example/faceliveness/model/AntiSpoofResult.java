package com.example.faceliveness.model;

/**
 * Aggregated result from all passive anti-spoof checks.
 * Combines FrameConsistencyChecker and PassiveAntiSpoofAnalyzer verdicts.
 */
public class AntiSpoofResult {

    private final boolean passed;
    private final AntiSpoofFailureReason failureReason;
    private final String details;

    public AntiSpoofResult(boolean passed) {
        this(passed, null, "");
    }

    public AntiSpoofResult(boolean passed, AntiSpoofFailureReason failureReason) {
        this(passed, failureReason, "");
    }

    public AntiSpoofResult(boolean passed, AntiSpoofFailureReason failureReason, String details) {
        this.passed = passed;
        this.failureReason = failureReason;
        this.details = details;
    }

    public boolean isPassed() {
        return passed;
    }

    public AntiSpoofFailureReason getFailureReason() {
        return failureReason;
    }

    public String getDetails() {
        return details;
    }
}
