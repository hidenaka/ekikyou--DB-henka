# LLM Annotation Prompt Template for Trigram Assignment

## Overview

This document describes the prompt template used by `scripts/annotate_gold_set.py` to assign
trigrams to case data via LLM (Claude). Each case receives 4 trigram assignments:
- `before_lower_trigram` (internal driver of the before-state)
- `before_upper_trigram` (external manifestation of the before-state)
- `after_lower_trigram` (internal driver of the after-state)
- `after_upper_trigram` (external manifestation of the after-state)

## Dual Annotation Design

Two independent passes are made per case:
- **Pass 1**: `temperature=0`, system prompt emphasizes analytical precision
- **Pass 2**: `temperature=0.3`, system prompt emphasizes holistic narrative reading

This produces independent annotations for computing inter-rater agreement (Cohen's kappa).

---

## System Prompt (Pass 1 - Analytical)

```
You are an expert annotator for the I Ching (易経) trigram classification system.
Your task is to read case descriptions of organizational/personal/national transitions
and assign trigrams based SOLELY on the text content.

CRITICAL RULES:
1. Read the story text carefully. Do NOT simply map state labels to trigrams.
2. Each trigram is equally likely a priori. Consider all 8 options for every position.
3. Inner (lower) and outer (upper) trigrams are INDEPENDENT assessments.
4. Base your judgment on textual evidence, not on label-to-trigram shortcuts.

You are precise, analytical, and evidence-based. Cite specific phrases from the text.
```

## System Prompt (Pass 2 - Holistic)

```
You are an expert annotator for the I Ching (易経) trigram classification system.
Your task is to read case descriptions of organizational/personal/national transitions
and assign trigrams based on the overall narrative arc and emotional tenor.

CRITICAL RULES:
1. Read the story text carefully. Do NOT simply map state labels to trigrams.
2. Each trigram is equally likely a priori. Consider all 8 options for every position.
3. Inner (lower) and outer (upper) trigrams are INDEPENDENT assessments.
4. Focus on the underlying dynamics and emotional quality of the transition.

You read narratives holistically, sensing the deeper currents beneath surface descriptions.
```

---

## User Prompt Template

```
Below is a case describing a transition (change event). Read ALL fields carefully, then assign
trigrams for the before-state and after-state.

=== CASE DATA ===
Story Summary: {story_summary}
Before State Label: {before_state}
After State Label: {after_state}
Trigger Type: {trigger_type}
Action Type: {action_type}
Scale: {scale}
Target: {target_name}
Period: {period}
=================

## YOUR TASK

For each of the 4 positions below, select ONE trigram from the 8 options.

### The 8 Trigrams and Their Meanings

IMPORTANT: Each trigram has DIFFERENT meanings depending on whether it is assigned as
an inner (lower) trigram or an outer (upper) trigram. Read both columns carefully.

| Trigram | As INNER (Lower) Trigram — Subject's Internal State | As OUTER (Upper) Trigram — External Environment |
|---------|------------------------------------------------------|------------------------------------------------|
| 乾 (Qian) | The subject has strong will, confidence, and actively drives expansion/leadership. Energy is abundant and pushing forward. | The environment supports growth/expansion. Markets or society are in an upward trend. Strong external authority is at work. |
| 坤 (Kun) | The subject is receptive, passive, building foundations quietly. Following rather than leading. Steady accumulation without assertion. | The environment is stable with no major upheaval. Mature market. Calm, supportive conditions. |
| 震 (Zhen) | The subject made a sudden decision or action. Internal shock or disruption occurred. A new initiative was launched abruptly. | External shock hit (disaster, regulation change, market crash). Sudden environmental disruption. Disruptive technology appeared. |
| 巽 (Xun) | The subject adapts flexibly, changing gradually. Incremental strategy, penetrating like wind. DX, optimization, step-by-step reform. | Environmental change is slow but steady. Social trends gradually penetrating. Regulations evolving incrementally. |
| 坎 (Kan) | The subject faces serious difficulty, structural risk, or crisis. Financial distress, organizational collapse, loss of trust. Trapped in hardship. | The environment is harsh and dangerous. Recession, pandemic, intense competition threatening survival. Regulatory pressure. |
| 離 (Li) | The subject's core driver is clarity, vision, discernment, or passionate illumination. Ideas/technology shine brightly. Analytical insight or creative passion drives action. Transparency is central. | Media attention is focused on the subject. Information is being revealed/publicized. Society's gaze is concentrated. Under evaluation or scrutiny. |
| 艮 (Gen) | The subject intentionally stops, accumulates, or reflects. A deliberate pause. Restructuring preparation. "We will not proceed further" as a strategic choice. | The environment is stuck/stagnant. Market has plateaued. Barriers block progress. Society is in a "wait" mode. |
| 兌 (Dui) | The subject's core driver is joy, openness, communication, or collaborative exchange. Satisfaction with results. Dialogue and connection are central values. Open organizational culture. | Market/society is welcoming and favorable. High praise from customers/stakeholders. Partnership opportunities abound. Celebratory/optimistic mood. |

### Position Definitions

- **Inner (Lower) Trigram**: The subject's INTERNAL state — motivation, psychology, organizational health, foundational driver. What is happening INSIDE, often invisible from outside.
- **Outer (Upper) Trigram**: The EXTERNAL environment — market conditions, social context, visible situation, how things appear from outside.

### Critical Instructions

1. **Do NOT mechanically map labels to trigrams.** "どん底・危機" does NOT automatically mean 坎.
   A company in crisis might have a leader with burning vision (離 as inner) or might be
   deliberately stopping to restructure (艮 as inner). READ THE STORY.

2. **Inner and outer are INDEPENDENT.** Assess them separately. They CAN be the same trigram
   but usually differ. Never default to making them the same.

3. **Consider 離 and 兌 as inner trigrams equally.** They are just as valid as any other:
   - 離 as inner: When the subject's core driver is clarity, vision, discernment, or passionate
     illumination. Example: a company driven by R&D brilliance, a leader whose analytical
     clarity guides decisions, a startup whose core is innovative vision.
   - 兌 as inner: When the subject's core driver is joy, openness, communication, or
     collaborative exchange. Example: a company whose culture centers on dialogue and
     employee satisfaction, a leader driven by building connections, an organization that
     thrives on stakeholder relationships.

4. **State labels are hints, not answers.** They provide context but the story_summary
   is the primary evidence source.

5. **For each position, also provide:**
   - Your confidence level (high/medium/low)
   - One alternative trigram you seriously considered
   - Brief reasoning citing specific text evidence

### Output Format

Return ONLY valid JSON in this exact structure:

```json
{
  "before_lower": {
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  },
  "before_upper": {
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  },
  "after_lower": {
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  },
  "after_upper": {
    "trigram": "X",
    "confidence": "high|medium|low",
    "alternative": "Y",
    "reasoning": "..."
  }
}
```

Trigram values must be one of: 乾, 坤, 震, 巽, 坎, 離, 艮, 兌
```

---

## Design Rationale

### Why Two System Prompts?

The analytical prompt (Pass 1) emphasizes evidence citation and precision, while the
holistic prompt (Pass 2) emphasizes narrative arc and emotional dynamics. This variation
produces meaningfully independent annotations rather than deterministic duplicates.

### Why Temperature Variation?

`temperature=0` for Pass 1 produces the model's most confident single answer.
`temperature=0.3` for Pass 2 introduces slight variation, allowing the model to surface
alternative interpretations that might be equally valid.

### Addressing the 離/兌 Inner Trigram Gap

The existing data has 離 and 兌 appearing 0 times as inner trigrams due to formulaic
label-to-trigram mapping. The prompt explicitly:
1. Defines what each trigram means specifically as an inner trigram
2. Provides concrete examples for 離 and 兌 as inner trigrams
3. Instructs the model that all 8 trigrams are equally valid a priori
4. Warns against mechanical label mapping (the root cause of the gap)

### Confidence + Alternative Design

Requesting confidence levels and alternatives serves two purposes:
1. Identifies cases where the model is uncertain (candidates for adjudication)
2. The alternative trigram data enables confusion-pair analysis (which trigrams
   are most often confused with each other)
