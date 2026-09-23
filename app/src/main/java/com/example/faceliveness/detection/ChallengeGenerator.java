package com.example.faceliveness.detection;

import com.example.faceliveness.model.ChallengeType;
import com.example.faceliveness.model.LivenessChallenge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Generates a randomized sequence of liveness challenges.
 * Randomization prevents pre-recorded video replay attacks.
 */
public final class ChallengeGenerator {

    private ChallengeGenerator() {
    }

    // Base timeouts, tuned per gesture rather than one blanket value:
    //  - BLINK: near-instant once triggered — dominant cost is reaction time, not the blink itself
    //  - TURN_LEFT/RIGHT: needs to rotate, hold briefly, and stay in-range (see FaceAnalyzer's
    //    distance gating) for a few confirming frames
    //  - SMILE: needs a genuine rise (see ExpressionDynamicsAnalyzer.SMILE_ARC_MIN_RANGE) —
    //    a natural smile takes longer to build than a blink or head turn
    //  - NOD: a full down-up (or up-down) cycle is the slowest gesture here
    private static final List<LivenessChallenge> ALL_CHALLENGES = new ArrayList<>();

    static {
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.BLINK, "Please blink your eyes. Open your spectacles", 10000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.TURN_LEFT, "Turn your head to the LEFT", 10000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.TURN_RIGHT, "Turn your head to the RIGHT", 10000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.SMILE, "Please smile", 10000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.NOD, "Nod your head up and down", 10000L));
    }

    // Random jitter applied to each challenge's timeout at generation time —
    // imperceptible to a real person, but stops an attacker from pre-scripting
    // a response precisely timed against a known, fixed window. Kept small
    // (well under 1s) so it never meaningfully shortens the human-factors
    // floor each gesture actually needs.
    private static final long JITTER_MILLIS = 700L;

    private static final Random RANDOM = new Random();

    /**
     * Returns a randomized subset of challenges, each with a fresh, jittered
     * timeout — never returns the shared static template instances directly,
     * so ALL_CHALLENGES itself is never mutated across calls/sessions.
     */
    public static List<LivenessChallenge> generateChallenges(int count) {
        List<LivenessChallenge> shuffled = new ArrayList<>(ALL_CHALLENGES);
        Collections.shuffle(shuffled);
        int take = Math.min(count, shuffled.size());

        List<LivenessChallenge> result = new ArrayList<>(take);
        for (LivenessChallenge template : shuffled.subList(0, take)) {
            long jitter = (long) ((RANDOM.nextDouble() * 2 - 1) * JITTER_MILLIS); // ±JITTER_MILLIS
            long jitteredTimeout = template.getTimeoutMillis() + jitter;
            result.add(new LivenessChallenge(template.getType(), template.getInstruction(), jitteredTimeout));
        }
        return result;
    }

    /** Default: 3 challenges picked randomly from the full set. */
    public static List<LivenessChallenge> generateChallenges() {
        return generateChallenges(3);
    }
}
