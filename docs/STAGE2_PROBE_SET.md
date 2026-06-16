# Stage 2 — Held-out Probe Set (20 multi-turn recall probes)

These 20 conversations are the Bar 2 evaluation set. Each is a 3-5 turn dialogue whose final user turn requires recalling a fact stated several turns earlier. `answer_keys` are the substrings (case-insensitive) that count as a correct recall.

Canonical source of truth is `PROBES` in `scripts/eval_stage2_ablation.py`; this file is generated from it for review. If you edit one, regenerate the other.

## p01 — named-entity
*answer keys:* 'Sarah'

1. I'm planning a trip to Lisbon with my friend Sarah.
2. Can you suggest three neighborhoods to stay in?
3. Nice. What about restaurants in the second one you mentioned?
4. What was my friend's name again?  **(recall)**

## p02 — preference
*answer keys:* 'vegetarian', 'nut'

1. Quick note about me: I'm vegetarian and allergic to nuts.
2. Give me a dinner recipe idea.
3. Now suggest a dessert to go with it.
4. Remind me — what are my dietary restrictions?  **(recall)**

## p03 — number
*answer keys:* '17', '42'

1. My two favorite numbers are 17 and 42.
2. Tell me a fact about prime numbers.
3. What's an interesting property of even numbers?
4. What were my two favorite numbers?  **(recall)**

## p04 — named-entity
*answer keys:* 'Mochi'

1. I just adopted a dog named Mochi.
2. What's a good daily walking routine for a puppy?
3. How often should puppies eat?
4. By the way, what's my dog's name?  **(recall)**

## p05 — named-entity
*answer keys:* 'Tucson'

1. I grew up in Tucson before moving away for work.
2. What are some things that make desert cities unique?
3. How do people stay cool in extreme heat?
4. Where did I say I grew up?  **(recall)**

## p06 — occupation
*answer keys:* 'marine biologist', 'marine biology'

1. I work as a marine biologist studying coral reefs.
2. What's causing coral bleaching?
3. Are there reefs that recover well?
4. What's my job again?  **(recall)**

## p07 — named-entity
*answer keys:* 'Subaru'

1. I drive a 2012 Subaru Outback.
2. What maintenance should I do at 150,000 miles?
3. Is it worth replacing the timing belt early?
4. What car do I drive?  **(recall)**

## p08 — named-entity
*answer keys:* 'Salt Road', 'The Salt Road'

1. I'm writing a novel called The Salt Road.
2. How do I keep a reader engaged in chapter one?
3. Any tips for writing believable dialogue?
4. Do you remember the title of my novel?  **(recall)**

## p09 — preference
*answer keys:* 'teal'

1. My favorite color is teal.
2. Suggest a color palette for a living room.
3. What accent colors pair well with grey?
4. What's my favorite color?  **(recall)**

## p10 — schedule
*answer keys:* 'Thursday'

1. I have a dentist appointment on Thursday.
2. How should I prepare for a routine cleaning?
3. Is flossing the night before enough?
4. Which day is my dentist appointment?  **(recall)**

## p11 — named-entity
*answer keys:* 'Priya'

1. My daughter Priya is starting school next month.
2. How can I help a child adjust to a new school?
3. What's a good bedtime routine for a six-year-old?
4. What is my daughter's name?  **(recall)**

## p12 — named-entity
*answer keys:* 'Portuguese'

1. I'm learning Portuguese for an upcoming move.
2. What's the fastest way to build vocabulary?
3. How much daily practice do you recommend?
4. Which language am I learning?  **(recall)**

## p13 — medical
*answer keys:* 'penicillin'

1. Important: I'm allergic to penicillin.
2. What are general signs of an allergic reaction?
3. When should someone use an epinephrine auto-injector?
4. What am I allergic to?  **(recall)**

## p14 — number
*answer keys:* '2000', '2,000', '$2000', '$2,000'

1. My budget for a new laptop is 2000 dollars.
2. What specs matter most for video editing?
3. Is more RAM or a faster CPU more important?
4. What was my budget?  **(recall)**

## p15 — named-entity
*answer keys:* 'Denver'

1. I'm moving to Denver in the spring.
2. What should I know about living at high altitude?
3. How long does it take to acclimate?
4. Which city am I moving to?  **(recall)**

## p16 — preference
*answer keys:* 'cello'

1. I play the cello in a community orchestra.
2. How do I keep my bowing relaxed?
3. Any advice for sight-reading?
4. Which instrument do I play?  **(recall)**

## p17 — schedule
*answer keys:* 'March 14', 'March 14th'

1. My project deadline is March 14.
2. How do I plan backwards from a deadline?
3. What's a good way to handle scope creep?
4. When is my deadline?  **(recall)**

## p18 — preference
*answer keys:* 'Arsenal'

1. I support Arsenal in the Premier League.
2. What makes a strong midfield?
3. How important is squad depth over a season?
4. Which team do I support?  **(recall)**

## p19 — secret-word
*answer keys:* 'Albatross', 'albatross'

1. Let's set a codeword for this session: Albatross.
2. Tell me a short fact about the ocean.
3. Now tell me a fact about birds.
4. What was the codeword I set?  **(recall)**

## p20 — preference
*answer keys:* 'oat milk'

1. My usual coffee order is an oat milk latte, no sugar.
2. How is a latte different from a flat white?
3. Does milk type change the foam?
4. What's my usual coffee order?  **(recall)**

