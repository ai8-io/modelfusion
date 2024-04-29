import { TextGenerationPromptTemplate } from "../TextGenerationPromptTemplate";
import { ChatPrompt } from "./ChatPrompt";
import { validateContentIsString } from "./ContentPart";
import { InstructionPrompt } from "./InstructionPrompt";
import { InvalidPromptError } from "./InvalidPromptError";

// See
// (1) - https://github.com/Azure-Samples/Phi-3MiniSamples/blob/main/onnx/run-onnx.ipynb
// (2) - https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
// (3) - https://ollama.com/library/phi3:latest/blobs/c608dc615584

const BEGIN_SEGMENT = ""; // Doesn't appear to be needed - <s> mentioned in one place but nowhere else

const BEGIN_INSTRUCTION = "<|user|>";
const END_INSTRUCTION = "<|end|>\n";

// This is the marker of an assistant response, or the end of the prompt to indicate it should carry on
const BEGIN_RESPONSE_ASSISTANT = "<|assistant|>";

// Some indication that System prompt on Phi3 might not be reliable - and some docs (2) don't mention it
const BEGIN_SYSTEM = "<|system|>";
const END_SYSTEM = "<|end|>\n";

// <|end|><|endoftext|>is what the assistant sends to indicate it has finished and has no more to say
const STOP_SEQUENCE = "<|end|>";
const END_SEGMENT = "<|endoftext|>";

/**
 * Formats a text prompt as a Llama 3 prompt.
 *
 * Phi 3 prompt template:
 * ```
 * <|user|>{ instruction }<|end|>
 * <|assistant|>
 * ```
 *
 * @see https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
 */
export function text(): TextGenerationPromptTemplate<string, string> {
  return {
    stopSequences: [STOP_SEQUENCE, END_SEGMENT],
    format(prompt) {
      let result = `${BEGIN_SEGMENT}${BEGIN_INSTRUCTION}${prompt}${END_INSTRUCTION}${BEGIN_RESPONSE_ASSISTANT}`;

      return result;
    },
  };
}

/**
 * Formats an instruction prompt as a Llama 3 prompt.
 *
 * Phi 3 prompt template:
 * ```
 * <|system|>{ $system prompt }<|end|>
 *
 * <|user|>{ instruction }<|end|>
 * <|assistant|>
 * ```
 *
 * @see https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
 */
export function instruction(): TextGenerationPromptTemplate<
  InstructionPrompt,
  string
> {
  return {
    stopSequences: [STOP_SEQUENCE, END_SEGMENT],
    format(prompt) {
      const instruction = validateContentIsString(prompt.instruction, prompt);

      let result = `${BEGIN_SEGMENT}`;
      result += `${prompt.system != null ? `${BEGIN_SYSTEM}${prompt.system}${END_SYSTEM}` : ""}`;
      result += `${BEGIN_INSTRUCTION}${instruction}${END_INSTRUCTION}${BEGIN_RESPONSE_ASSISTANT}`;

      return result;
    },
  };
}

/**
 * Formats a chat prompt as a Llama 3 prompt.
 *
 * Phi 3 prompt template:
 *
 * ```
 * <<|system|>{ $system prompt }<|end|>
 *
 * <|user|>${ user msg 1 }<|end|><|assistant|>
 * ${ model response 1 }<|end|><|user|>${ user msg 2 }<|end|><|assistant|>
 * ${ model response 2 }<|end|><|user|>${ user msg 3 }<|end|><|assistant|>
 * ```
 *
 * @see https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
 */
export function chat(): TextGenerationPromptTemplate<ChatPrompt, string> {
  return {
    format(prompt) {
      validatePhi3Prompt(prompt);

      // get content of the first message (validated to be a user message)
      const content = prompt.messages[0].content;

      let text = `${BEGIN_SEGMENT}`;
      text += `${prompt.system != null ? `${BEGIN_SYSTEM}${prompt.system}${END_SYSTEM}` : ""}`;
      text += `${BEGIN_INSTRUCTION}${content}${END_INSTRUCTION}${BEGIN_RESPONSE_ASSISTANT}`;

      // process remaining messages
      for (let i = 1; i < prompt.messages.length; i++) {
        const { role, content } = prompt.messages[i];
        switch (role) {
          case "user": {
            const textContent = validateContentIsString(content, prompt);
            text += `${BEGIN_INSTRUCTION}${textContent}${END_INSTRUCTION}${BEGIN_RESPONSE_ASSISTANT}`;
            break;
          }
          case "assistant": {
            // The assistant will have added \n\n to the start of their response - we don't do that so the tests are slightly different than reality
            text += `${validateContentIsString(content, prompt)}${END_INSTRUCTION}`;
            break;
          }
          case "tool": {
            throw new InvalidPromptError(
              "Tool messages are not supported.",
              prompt
            );
          }
          default: {
            const _exhaustiveCheck: never = role;
            throw new Error(`Unsupported role: ${_exhaustiveCheck}`);
          }
        }
      }

      return text;
    },
    stopSequences: [STOP_SEQUENCE, END_SEGMENT],
  };
}

/**
 * Checks if a Phi3 chat prompt is valid. Throws a {@link ChatPromptValidationError} if it's not.
 *
 * - The first message of the chat must be a user message.
 * - Then it must be alternating between an assistant message and a user message.
 * - The last message must always be a user message (when submitting to a model).
 *
 * The type checking is done at runtime when you submit a chat prompt to a model with a prompt template.
 *
 * @throws {@link ChatPromptValidationError}
 */
export function validatePhi3Prompt(chatPrompt: ChatPrompt) {
  const messages = chatPrompt.messages;

  if (messages.length < 1) {
    throw new InvalidPromptError(
      "ChatPrompt should have at least one message.",
      chatPrompt
    );
  }

  for (let i = 0; i < messages.length; i++) {
    const expectedRole = i % 2 === 0 ? "user" : "assistant";
    const role = messages[i].role;

    if (role !== expectedRole) {
      throw new InvalidPromptError(
        `Message at index ${i} should have role '${expectedRole}', but has role '${role}'.`,
        chatPrompt
      );
    }
  }

  if (messages.length % 2 === 0) {
    throw new InvalidPromptError(
      "The last message must be a user message.",
      chatPrompt
    );
  }
}
