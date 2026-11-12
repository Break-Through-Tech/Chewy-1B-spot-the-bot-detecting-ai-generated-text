
export type ParaphraseLevel = 0 | 1 | 2;

export type ParaphraseVariant = {
  level: ParaphraseLevel;
  label: string;
  text: string;
  ai_score: number;
};

export type ParaphraseSample = {
  id: string;
  title: string;
  variants: [
    ParaphraseVariant,
    ParaphraseVariant,
    ParaphraseVariant
  ];
};

export const paraphraseSamples: ParaphraseSample[] = [
  {
    id: "pet-care",
    title: "Pet Care Advice",
    variants: [
      {
        level: 0,
        label: "Original",
        text: "If your dog frequently has digestive issues, it may be worth looking into food formulated for sensitive stomachs. Some diets use highly digestible ingredients, adjusted fat levels, or specific fiber blends to support digestive health.",
        ai_score: 0.97,
      },
      {
        level: 1,
        label: "Lightly paraphrased",
        text: "If your dog often experiences digestive problems, you might consider food designed for sensitive stomachs. These diets may include easily digestible ingredients, different fat levels, or fiber blends that support healthy digestion.",
        ai_score: 0.78,
      },
      {
        level: 2,
        label: "Heavily paraphrased",
        text: "Does your dog's stomach get upset a lot? It might be worth trying a food that's easier on their stomach. Some formulas use ingredients that are easier to break down, along with changes to the fat and fiber content.",
        ai_score: 0.24,
      },
    ],
  },
  {
    id: "technology",
    title: "Technology",
    variants: [
      {
        level: 0,
        label: "Original",
        text: "Artificial intelligence is transforming the way organizations process information, enabling businesses to identify patterns, improve operational efficiency, and make more informed decisions.",
        ai_score: 0.89,
      },
      {
        level: 1,
        label: "Lightly paraphrased",
        text: "Artificial intelligence is changing how organizations process information, helping businesses recognize patterns, increase operational efficiency, and make better-informed decisions.",
        ai_score: 0.72,
      },
      {
        level: 2,
        label: "Heavily paraphrased",
        text: "Companies are starting to work with data differently because of AI. It can help them spot things they might have missed, get their work done more efficiently, and make better decisions.",
        ai_score: 0.41,
      },
    ],
  },
  {
    id: "daily-routines",
    title: "Daily Routines",
    variants: [
      {
        level: 0,
        label: "Original",
        text: "Establishing a consistent daily routine can improve productivity, reduce stress, and promote overall well-being by providing structure and encouraging healthy habits.",
        ai_score: 0.94,
      },
      {
        level: 1,
        label: "Lightly paraphrased",
        text: "Following a regular daily routine can help increase productivity, lower stress, and improve well-being by adding structure to your day and supporting healthy habits.",
        ai_score: 0.63,
      },
      {
        level: 2,
        label: "Heavily paraphrased",
        text: "Having a regular routine can make your day feel less chaotic. It helps you get things done, feel less stressed, and stick to healthier habits.",
        ai_score: 0.18,
      },
    ],
  },
];

