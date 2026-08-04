package com.example.faceliveness.model;

/**
 * Reasons a passive anti-spoof check may fail.
 */
public enum AntiSpoofFailureReason {
    /** Face showed no natural micro-movement — likely a held photo */
    STATIC_FACE_DETECTED,

    /** Pixel analysis shows screen/photo brightness pattern */
    SCREEN_OR_PHOTO_TEXTURE_DETECTED,

    /** Both static movement and screen texture detected — high confidence spoof */
    COMBINED_SPOOF_SIGNALS
}
