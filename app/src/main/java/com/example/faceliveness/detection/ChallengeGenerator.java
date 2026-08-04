package com.example.faceliveness.detection;

import com.example.faceliveness.model.ChallengeType;
import com.example.faceliveness.model.LivenessChallenge;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Generates a randomized sequence of liveness challenges.
 * Randomization prevents pre-recorded video replay attacks.
 */
public final class ChallengeGenerator {

    private ChallengeGenerator() {
    }

    private static final List<LivenessChallenge> ALL_CHALLENGES = new ArrayList<>();

    static {
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.BLINK, "Please blink your eyes", 5000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.TURN_LEFT, "Turn your head to the LEFT", 5000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.TURN_RIGHT, "Turn your head to the RIGHT", 5000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.SMILE, "Please smile", 5000L));
        ALL_CHALLENGES.add(new LivenessChallenge(ChallengeType.NOD, "Nod your head up and down", 6000L));
    }

    /**
     * Returns a randomized subset of challenges.
     * Order is randomized each call — defeating replay attacks.
     */
    public static List<LivenessChallenge> generateChallenges(int count) {
        List<LivenessChallenge> shuffled = new ArrayList<>(ALL_CHALLENGES);
        Collections.shuffle(shuffled);
        int take = Math.min(count, shuffled.size());
        return new ArrayList<>(shuffled.subList(0, take));
    }

    /** Default: 3 challenges picked randomly from the full set. */
    public static List<LivenessChallenge> generateChallenges() {
        return generateChallenges(3);
    }
}
