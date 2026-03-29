"""
Autism Sensory Profiles
-----------------------
Based on DSM-5 sensory processing criteria, the Sensory Processing Measure (SPM),
and peer-reviewed research on sensory modulation in autism spectrum disorder.

References:
- Marco et al. (2011). Sensory processing in autism: A review of neurophysiologic findings.
- Baranek et al. (2006). Sensory processing subtypes in autism.
- Green et al. (2016). Sensory over-responsivity in ASD.
"""

from dataclasses import dataclass
from typing import List
import random


@dataclass
class AutismSensoryProfile:
    name: str
    # Preferred content modality based on sensory processing strengths
    preference: str            # 'visual' | 'audio' | 'text' | 'mixed'
    # Baseline arousal/stress level at session start
    baseline_stress: float     # 0.0 - 1.0
    # Learning acquisition speed
    learning_rate: float       # 0.5 - 1.2
    # Global sensory sensitivity multiplier
    sensory_sensitivity: float # 0.5 - 2.0
    # Sustained attention capacity (affects fatigue rate)
    attention_span: float      # 0.2 - 1.0
    # Auditory hypersensitivity
    noise_sensitivity: float   # 0.5 - 2.0
    # Visual hypersensitivity (bright colors, motion, contrast)
    visual_sensitivity: float  # 0.5 - 2.0
    # Cognitive processing speed
    processing_speed: float    # 0.5 - 1.5
    # Optimal task difficulty for zone of proximal development
    optimal_difficulty: int    # 1-5
    # Preferred font size for reduced visual strain
    preferred_font_size: int   # 14-22
    # Preferred display contrast
    preferred_contrast: str    # 'normal' | 'high' | 'low'
    # Maximum tolerated animation speed
    max_animation_speed: str   # 'none' | 'slow' | 'normal'


# Four clinically-inspired profiles covering common ASD sensory subtypes
PROFILES: List[AutismSensoryProfile] = [
    AutismSensoryProfile(
        # Visual hypersensitivity: bright lights, fast motion cause overload
        # Prefers audio/text content with minimal visual stimulation
        name="hypersensitive_visual",
        preference="audio",
        baseline_stress=0.35,
        learning_rate=0.85,
        sensory_sensitivity=1.8,
        attention_span=0.55,
        noise_sensitivity=0.6,
        visual_sensitivity=1.9,
        processing_speed=0.70,
        optimal_difficulty=2,
        preferred_font_size=18,
        preferred_contrast="low",
        max_animation_speed="none"
    ),
    AutismSensoryProfile(
        # Auditory hypersensitivity: sounds cause distress
        # Prefers visual content in quiet/text-forward formats
        name="hypersensitive_auditory",
        preference="visual",
        baseline_stress=0.30,
        learning_rate=0.90,
        sensory_sensitivity=1.5,
        attention_span=0.65,
        noise_sensitivity=1.9,
        visual_sensitivity=0.8,
        processing_speed=0.80,
        optimal_difficulty=3,
        preferred_font_size=16,
        preferred_contrast="high",
        max_animation_speed="slow"
    ),
    AutismSensoryProfile(
        # Sensory seeking / hyposensitive: craves stimulation
        # Can handle more intensity; gets bored easily without variation
        name="hyposensitive_seeking",
        preference="mixed",
        baseline_stress=0.18,
        learning_rate=1.05,
        sensory_sensitivity=0.6,
        attention_span=0.40,
        noise_sensitivity=0.5,
        visual_sensitivity=0.6,
        processing_speed=1.20,
        optimal_difficulty=4,
        preferred_font_size=14,
        preferred_contrast="normal",
        max_animation_speed="normal"
    ),
    AutismSensoryProfile(
        # Mixed pattern: moderate sensitivities across modalities
        # Balanced profile, responds well to gradual adaptation
        name="mixed_pattern",
        preference="visual",
        baseline_stress=0.25,
        learning_rate=0.95,
        sensory_sensitivity=1.2,
        attention_span=0.70,
        noise_sensitivity=1.3,
        visual_sensitivity=1.1,
        processing_speed=0.90,
        optimal_difficulty=3,
        preferred_font_size=16,
        preferred_contrast="normal",
        max_animation_speed="slow"
    ),
]


def get_random_profile() -> AutismSensoryProfile:
    return random.choice(PROFILES)


def get_profile_by_name(name: str) -> AutismSensoryProfile:
    for p in PROFILES:
        if p.name == name:
            return p
    return PROFILES[0]
