import { FunctionEvent } from "../core/FunctionEvent.js";
import { ModelCallFinishedEvent } from "./ModelCallEvent.js";
import { ModelInformation } from "./ModelInformation.js";

export type SuccessfulModelCall = {
  type:
    | "image-generation"
    | "json-generation"
    | "json-or-text-generation"
    | "speech-synthesis"
    | "text-embedding"
    | "text-generation"
    | "text-streaming"
    | "transcription";
  model: ModelInformation;
  settings: unknown;
  response: unknown;
};

export function extractSuccessfulModelCalls(
  runFunctionEvents: FunctionEvent[]
) {
  return runFunctionEvents
    .filter(
      (event): event is ModelCallFinishedEvent & { status: "success" } =>
        "status" in event && event.status === "success"
    )
    .map(
      (event): SuccessfulModelCall => ({
        model: event.model,
        settings: event.settings,
        response: event.response,
        type: event.functionType,
      })
    );
}
