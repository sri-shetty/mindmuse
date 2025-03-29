# pylint: disable=line-too-long,invalid-name
"""
This module demonstrates the usage of the Vertex AI Gemini 1.5 API within a Streamlit application.
"""

import os
from typing import List, Tuple, Union

import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_REGION")

vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models() -> Tuple[GenerativeModel, GenerativeModel]:
    """Load Gemini 1.5 Flash and Pro models."""
    return GenerativeModel("gemini-1.5-flash-001"), GenerativeModel(
        "gemini-1.5-pro-001"
    )


def get_gemini_response(
    model: GenerativeModel,
    contents: Union[str, List],
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=2048
    ),
    stream: bool = True,
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def get_model_name(model: GenerativeModel) -> str:
    """Get Gemini Model Name"""
    model_name = model._model_name.replace(  # pylint: disable=protected-access
        "publishers/google/models/", ""
    )
    return f"`{model_name}`"


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


st.header("Furallies AI Assist", divider="rainbow")
gemini_15_flash, gemini_15_pro = load_models()

#st.subheader("Generate a story")

#selected_model = st.radio(
#    "Select Gemini Model:",
#    [gemini_15_flash, gemini_15_pro],
#    format_func=get_model_name,
#    key="selected_model_story",
#    horizontal=True,
#)


selected_model = gemini_15_pro

#st.write(selected_model)


# Story premise
#dog_name = st.text_input(
#    "Enter your dog's name: \n\n", key="dog_name", value=""
#)
dog_type = st.text_input(
    "What type of dog is it? \n\n", key="dog_type", value="German Shepherd"
)
dog_persona = st.text_input(
    "What is dog's typical personality like? \n\n",
    key="dog_persona",
    value="A very friendly dog.",
)
dog_location = st.text_input(
    "City or Address of dog's location? \n\n",
    key="dog_location",
    value="Atlanta, GA",
)
dog_premise = st.multiselect(
    "How is your dog feeling now? (can select multiple) \n\n",
    [
        "Happiness",
        "Excitement",
        "Affection",
        "Anxiety/Fear",
        "Anger/Aggression",
        "Sadness/Boredom",
        "Curiosity",
        "Disgust/Guilt",
    ],
    key="story_premise",
    default=["Happiness", "Affection"],
)

ask_question = st.text_input(
    "Ask your question? \n\n",
    key="ask_question",
    value="Ask any question about your dog",
)

creative_control = "Low"

length_of_story = "Short"

if creative_control == "Low":
    temperature = 0.30
else:
    temperature = 0.95

if length_of_story == "Short":
    max_output_tokens = 2048
else:
    max_output_tokens = 8192

prompt = f"""Provide explanation on dog's behavior. The explanation should consider all the following context and provide recommedation at the end. \n
Type of the dog: {dog_type} \n
Dog's personality: {dog_persona} \n
Address dog's live in: {dog_location} \n
Dog's current feeling: {",".join(dog_premise)}, explain the reason for this behavior. \n
Provide answer to question that is submitted which is {ask_question} \n
If {ask_question} is empty, provide general information about wellbeing of the dog. \n
List of veterinary hospital or clinic that is closest to the dog's location or address: \n
"""
config = GenerationConfig(
    temperature=temperature, max_output_tokens=max_output_tokens
)

generate_t2t = st.button("Generate answer", key="generate_t2t")
if generate_t2t and prompt:
    # st.write(prompt)
    with st.spinner(
        f"Generating medical advise on dog's behavior and answer dog owner's question {get_model_name(selected_model)} ..."
    ):
        first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
        with first_tab1:
            response = get_gemini_response(
                selected_model,  # Use the selected model
                prompt,
                generation_config=config,
            )
            if response:
                st.write("The Response:")
                st.write(response)
        with first_tab2:
            st.text(
                f"""Parameters:\n- Temperature: {temperature}\n- Max Output Tokens: {max_output_tokens}\n"""
            )
            st.text(prompt)
