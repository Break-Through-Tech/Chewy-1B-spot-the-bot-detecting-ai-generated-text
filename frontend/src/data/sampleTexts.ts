
export type SampleAnswer = {
  text: string;
  label: "human" | "chatgpt";
  target: 0 | 1;
};

export type SampleQuestion = {
  id: number;
  question: string;
  source: string;
  answers: {
    human: SampleAnswer;
    ai: SampleAnswer;
  };
};

export const sampleQuestions: SampleQuestion[] = [
  {
    id: 693,
    question:
      "Why is aluminum foil shiny on one side and dull on the other?",
    source: "reddit_eli5",
    answers: {
      human: {
        text: "I saw this in How It's Made. Basically aluminium foil is made in pairs, because that way the layers can be made thinner, as a single foil could tear before reaching the desired thickness. The two layers go through rollers and rollers during the process, so the outsides come out shiny but the insides, that's been against another layer of aluminium and didn't came into contact with anything else, come out opaque.",
        label: "human",
        target: 0,
      },
      ai: {
        text: "Aluminum foil is made by rolling sheets of aluminum metal until they are very thin. When the sheets are rolled, they go through a machine that polishes the metal and gives it a shiny finish on one side. The other side of the foil is left with a dull finish because it is the back side that was not polished by the machine. So, the shiny side of aluminum foil is the side that has been polished and the dull side is the side that has not been polished.",
        label: "chatgpt",
        target: 1,
      },
    },
  },
  {
    id: 11313,
    question: "Why isn't time travel possible?",
    source: "reddit_eli5",
    answers: {
      human: {
        text: "There are hypotheses that support certain methods of time travel, but they all rely on strongly bending spacetime, using wormholes or going near or over the speed of light, all of which we don't know how to do. The only physically impossible of those is going over the speed of light, as the speed of light is believed to be the absolute maximum speed for anything in the universe.",
        label: "human",
        target: 0,
      },
      ai: {
        text: "Well, time travel is not currently possible according to our current understanding of physics. The reason is that time travel would require some strange things to happen, like being able to travel faster than the speed of light, or being able to bend space and time in strange ways. Our current laws of physics don't allow us to do these things, so we don't have any way to travel through time right now.",
        label: "chatgpt",
        target: 1,
      },
    },
  },
  {
    id: 18028,
    question: "Who played Batman in The Dark Knight?",
    source: "open_qa",
    answers: {
      human: {
        text: "Christian Bale reprises the lead role of Bruce Wayne/Batman, with a returning cast of Michael Caine as Alfred Pennyworth, Gary Oldman as James Gordon and Morgan Freeman as Lucius Fox.",
        label: "human",
        target: 0,
      },
      ai: {
        text: 'Christian Bale played Batman in "The Dark Knight," which was the second film in Christopher Nolan\'s Batman trilogy. Bale\'s performance as Batman in the film was widely praised, and he received a number of awards and nominations for his work on the film. The Dark Knight was released in 2008 and was a critical and commercial success.',
        label: "chatgpt",
        target: 1,
      },
    },
  },
];
