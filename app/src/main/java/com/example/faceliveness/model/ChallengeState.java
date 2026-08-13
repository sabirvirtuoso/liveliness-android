package com.example.faceliveness.model;

/**
 * State of the current challenge during detection.
 * Java equivalent of a Kotlin sealed class: a closed hierarchy of state subtypes.
 */
public abstract class ChallengeState {

    private ChallengeState() {
    }

    public static final class Idle extends ChallengeState {
        public static final Idle INSTANCE = new Idle();

        private Idle() {
        }
    }

    public static final class InProgress extends ChallengeState {
        private final LivenessChallenge challenge;
        private final int index;
        private final int total;

        public InProgress(LivenessChallenge challenge, int index, int total) {
            this.challenge = challenge;
            this.index = index;
            this.total = total;
        }

        public LivenessChallenge getChallenge() {
            return challenge;
        }

        public int getIndex() {
            return index;
        }

        public int getTotal() {
            return total;
        }
    }

    public static final class ChallengeCompleted extends ChallengeState {
        private final LivenessChallenge challenge;

        public ChallengeCompleted(LivenessChallenge challenge) {
            this.challenge = challenge;
        }

        public LivenessChallenge getChallenge() {
            return challenge;
        }
    }

    /**
     * All gesture challenges passed — the user is now asked to hold still
     * for a moment while the MiniFASNet-V2 model snapshot is captured and
     * classified (see LivenessViewModel's stillness-tracking logic and
     * FaceAnalyzer.requestLivenessSnapshot()). Transitions to SessionPassed
     * once a model result (success or failure) comes back.
     */
    public static final class StillnessCheck extends ChallengeState {
        public static final StillnessCheck INSTANCE = new StillnessCheck();

        private StillnessCheck() {
        }
    }

    public static final class SessionPassed extends ChallengeState {
        private final LivenessResult result;

        public SessionPassed(LivenessResult result) {
            this.result = result;
        }

        public LivenessResult getResult() {
            return result;
        }
    }

    public static final class SessionFailed extends ChallengeState {
        private final LivenessResult result;

        public SessionFailed(LivenessResult result) {
            this.result = result;
        }

        public LivenessResult getResult() {
            return result;
        }
    }
}
