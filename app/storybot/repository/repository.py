import base64
import logging
import os

import replicate
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)


class StoryBotRepository:
    def __init__(self):
        self.vector_store = None

    def generate_story(self, text):
        """Generate a story using the langchain library and OpenAI's GPT-3 model"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
             You are a fun and seasoned storyteller. Maximum length 870 characters. Generate a story for me about {text}.
                     """,
            max_length=500,  # Maximum length of characters
        )
        story = LLMChain(llm=llm, prompt=prompt)
        return story.run(text=text)

    def generate_audio(self, text, voice):
        logging.info("generated audio for text:" + str({"text": text, voice: voice}))
        """Convert the generated story to audio using the Eleven Labs API."""
        audio_bytes = generate(text=text, voice=voice, api_key=eleven_api_key)
        logging.info("generated audio" + str({"len": len(audio_bytes)}))
        return audio_bytes

    def generate_images(self, story_text):
        """Generate images using the story text using the Replicate API."""
        output = replicate.run(
            "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
            input={"prompt": story_text},
        )
        return output
