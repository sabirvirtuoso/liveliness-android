package com.example.faceliveness.model;

/**
 * Represents a single liveness challenge the user must complete.
 */
public class LivenessChallenge {

    private final ChallengeType type;
    private final String instruction;
    private final long timeoutMillis;

    public LivenessChallenge(ChallengeType type, String instruction) {
        this(type, instruction, 5000L);
    }

    public LivenessChallenge(ChallengeType type, String instruction, long timeoutMillis) {
        this.type = type;
        this.instruction = instruction;
        this.timeoutMillis = timeoutMillis;
    }

    public ChallengeType getType() {
        return type;
    }

    public String getInstruction() {
        return instruction;
    }

    public long getTimeoutMillis() {
        return timeoutMillis;
    }
}
