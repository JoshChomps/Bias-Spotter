# Propaganda Technique Taxonomy

Based on the **SemEval-2020 Task 11** classification scheme. These 14 categories form the canonical taxonomy for the Cognitive Bias Spotter.

---

## 1. Loaded Language
**Definition:** Using words/phrases with strong emotional connotations to influence an audience beyond rational argument.

**Examples:**
- "brutal regime" instead of "government"
- "freedom fighters" vs. "terrorists"
- "taxpayer-funded handouts"

**Difficulty for ML:** Low — strong lexical signal. Layer 1 (lexicon) catches many cases.

---

## 2. Name Calling / Labeling
**Definition:** Attaching a negative label to a person, group, or idea to dismiss it without examining the evidence.

**Examples:**
- "snowflake," "extremist," "radical left"
- "anti-science," "conspiracy theorist"

**Difficulty for ML:** Low-Medium — overlaps heavily with Loaded Language.

---

## 3. Repetition
**Definition:** Repeating a message or slogan so frequently that the audience accepts it as truth.

**Examples:**
- Repeating "build the wall" across a speech
- Brand slogans repeated in ad copy

**Difficulty for ML:** Medium — requires tracking across a document, not just individual sentences.

---

## 4. Exaggeration / Minimization
**Definition:** Overstating or understating something to make it seem better or worse than it is.

**Examples:**
- "The greatest threat humanity has ever faced"
- "It's just a minor inconvenience" (for a major issue)

**Difficulty for ML:** Medium — requires understanding scale and context.

---

## 5. Doubt
**Definition:** Questioning the credibility of someone or something without providing evidence.

**Examples:**
- "Some experts disagree" (unnamed experts)
- "Can we really trust these numbers?"
- "Their motives are questionable"

**Difficulty for ML:** Medium — requires detecting hedging and vagueness.

---

## 6. Appeal to Fear / Prejudice
**Definition:** Building support by instilling fear or exploiting existing prejudices.

**Examples:**
- "If we don't act now, our children will pay the price"
- "They're coming for your jobs"

**Difficulty for ML:** Medium — overlaps with Loaded Language and urgency cues.

---

## 7. Flag-Waving
**Definition:** Appealing to patriotism, national identity, or group pride to justify a position.

**Examples:**
- "As true Americans, we must..."
- "Our great nation demands..."
- "For the sake of our country"

**Difficulty for ML:** Medium — requires detecting identity-group appeals.

---

## 8. Causal Oversimplification
**Definition:** Assuming a single cause for a complex outcome, ignoring other contributing factors.

**Examples:**
- "Crime is up because of immigration"
- "The economy crashed because of the president"

**Difficulty for ML:** High — requires understanding causal reasoning.

---

## 9. Slogans
**Definition:** Brief, striking phrases used to provoke an emotional reaction rather than a reasoned response.

**Examples:**
- "Make America Great Again"
- "Yes We Can"
- "Lock her up"

**Difficulty for ML:** Medium — often short, distinctive phrases. Repetition helps detection.

---

## 10. Appeal to Authority
**Definition:** Using an authority figure's opinion as evidence, especially when the authority is not an expert on the topic.

**Examples:**
- "A Nobel Prize winner says..."
- "Doctors recommend..."
- "According to a leading expert..."

**Difficulty for ML:** Medium — requires identifying authority claims and relevance.

---

## 11. Black-and-White Fallacy (False Dichotomy)
**Definition:** Presenting only two options when more exist, forcing a binary choice.

**Examples:**
- "You're either with us or against us"
- "If you're not part of the solution, you're part of the problem"

**Difficulty for ML:** High — requires structural argument analysis. LLM layer often needed.

---

## 12. Thought-Terminating Clichés
**Definition:** Commonly used phrases that discourage critical thinking and end debate.

**Examples:**
- "It is what it is"
- "Everything happens for a reason"
- "That's just how it works"
- "End of story"

**Difficulty for ML:** Low-Medium — strong lexical signal, but context matters.

---

## 13. Bandwagon / Reductio ad Hitlerum
**Definition:** Either appealing to popularity ("everyone does it") or comparing an opponent to Hitler/Nazis to discredit them.

**Examples:**
- Bandwagon: "Millions of people can't be wrong"
- Reductio: "You know who else wanted to control the media? Hitler."

**Difficulty for ML:** Medium — bandwagon has lexical cues; reductio requires entity recognition.

**Note:** This is the rarest category in SemEval data (<100 training examples). Class imbalance is severe.

---

## 14. Whataboutism / Straw Men / Red Herring
**Definition:** A super-category combining three deflection techniques:
- **Whataboutism:** Deflecting criticism by pointing to someone else's wrongdoing
- **Straw Man:** Misrepresenting an opponent's argument to make it easier to attack
- **Red Herring:** Introducing an irrelevant topic to divert attention

**Examples:**
- Whataboutism: "What about when your side did X?"
- Straw Man: "So you're saying we should just let criminals go free?"
- Red Herring: "But what about the economy?" (when discussing healthcare)

**Difficulty for ML:** High — requires understanding argument structure and relevance. These were collapsed into one category by SemEval because even human annotators couldn't reliably distinguish them.

---

## Extended Techniques (Beyond SemEval)

These are additional techniques the Bias Spotter targets via the lexicon and LLM layers:

| Technique | Description |
|-----------|-------------|
| False Scarcity | Manufactured urgency ("limited time," "only 3 left") |
| Identity Bait | "Real Americans," "true believers" — weaponizing group identity |
| FOMO | Fear of missing out, social pressure to act |
| Motte-and-Bailey | Defending a weak claim by retreating to a stronger one |
| Hidden Premises | Assumptions baked into the argument structure |
| Missing Comparisons | "50% better" — better than what? |
| Passive Voice Evasion | "Mistakes were made" — obscuring responsibility |

---

## Category Groupings

For the density score breakdown, techniques are grouped into meta-categories:

| Meta-Category | Techniques |
|---------------|-----------|
| **Emotional** | Loaded Language, Appeal to Fear, Name Calling, Flag-Waving |
| **Social** | Bandwagon, Appeal to Authority, Social Proof, Identity Bait |
| **Logical** | Black-and-White Fallacy, Causal Oversimplification, Whataboutism/Straw Man/Red Herring |
| **Framing** | Exaggeration/Minimization, Doubt, False Scarcity, Missing Comparisons |
| **Rhetorical** | Slogans, Repetition, Thought-Terminating Clichés |

---

## Universal Dataset Mapping (Cross-Domain)

To expand beyond news, we map external datasets into our canonical 14 categories:

### A. Social Media (GoEmotions - Reddit)
Reddit emotions are mapped as signals for specific techniques:
- **Anger, Fear, Disgust, Sadness** → Signal for **Loaded Language** and **Appeal to Fear**.
- **Admiration, Pride** → Signal for **Flag-Waving**.
- **Disapproval** → Signal for **Doubt** and **Name Calling**.

### B. Political Discourse (Vox / Speeches)
Formal rhetoric often uses specific structural techniques:
- **Whataboutism** is significantly more common in political debates.
- **Thought-Terminating Clichés** are used to end difficult lines of questioning.
- **Identity Bait** ("The common people") is the primary driver of political polarization.

### C. Conversational Bias (GUS-Net)
Token-level bias annotations (religion, race, gender) are mapped to:
- **Identity Bait**
- **Name Calling / Labeling**
- **Doubt** (bias via questioning credibility based on identity)
