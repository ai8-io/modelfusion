import { createOpenAITextModel } from "ai-utils.js/provider/openai";
import { generateText } from "ai-utils.js/text";

(async () => {
  const model = createOpenAITextModel({
    apiKey: "invalid-api-key",
    model: "text-davinci-003",
  });

  const generateStory = generateText.asSafeFunction({
    model,
    prompt: async ({ character }: { character: string }) =>
      `Write a short story about ${character} learning to love:\n\n`,
  });

  const abortController = new AbortController();
  abortController.abort(); // this would happen in parallel to generateStory

  const response = await generateStory(
    { character: "a robot" },
    { abortSignal: abortController.signal }
  );

  if (!response.ok) {
    if (response.isAborted) {
      console.error("The story generation was aborted.");
      return;
    }

    console.error(response.error);
    return;
  }

  console.log(response.result);
})();
