import openai
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re

SESSIONS = {}


def create_session(user_ip):
    SESSIONS[user_ip] = [
        """Narrator of a text based game where the player has to escape
- give short answers
- give no opinions or suggestions
- give no clues on how to scape 
- take no extra actions for the player but those he explicitly tells
- at the end of each action say what the player sees
- finish each sentence with "What do you do?"
- if the player escapes, show "winner, winner chicken dinner"
- if the player loses, show "game over"
Narrator: You wake up in a dark room. You can't remember how you got here. You can see a door and a window. What do you do?"""
    ]


def execute_completion_model(prompt, model="text-davinci-002", temperature=1, max_tokens=100, many=False, *args, **kwargs):
    """
    Executes the completion model with the given parameters and returns the list of responses.
    """
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        *args, **kwargs
    )
    if many:
        return [x.text.strip() for x in response.choices]
    else:
        return response.choices[0].text.strip()


def open_ai_model_func(model, type='completion'):
    """
    Returns a function that executes the given model with the specified type.
    """
    if type == 'completion':
        def execute(prompt_text, *args, **kwargs):
            return execute_completion_model(prompt_text, model=model, *args, **kwargs)
        return execute


def get_integer(txt):
    try:
        search = re.search("\\d+", txt)
        if search:
            return int(search.group())
    except Exception as e:
        print("err get_integer", txt, e)
    return 0


openai.api_key = os.getenv("OPENAPI_API_KEY")
gpt = open_ai_model_func("text-davinci-002")
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROMPT_FORMATS = dict(
    pros="top investors from y combinator give 5 pros for the idea \"{idea}\", include social impact:",
    cons="top investors from y combinator give 5 cons for the idea \"{idea}\":",
    state_of_the_art="give me companies that are already implementing ideas similar to this one \"{idea}\":",
    pitch="write pitch for this idea \"{idea}\":",

    viability="give a integer score from 1 to 10 to the viability of the idea in real life \"{idea}\":",
    innovation="give a integer score from 1 to 10 to the innovation of the idea \"{idea}\":",
    resources="give a integer score from 1 to 10 to the technological or logistical resources needed for the idea \"{idea}\":",
)


class Prompt(BaseModel):
    prompt: str


class ActionResponse(BaseModel):
    response: str
    game_over: bool


class CriticResponse(BaseModel):
    pros: str
    cons: str
    state_of_the_art: str
    pitch: str
    viability: int
    innovation: int
    resources: int


@app.get("/")
def read_root():
    """
    Test endpoint that returns a simple message.
    """
    return {"Hello": "World"}


@app.post("/start")
def start_game(request: Request):
    user_ip = request.client.host
    create_session(user_ip)
    return ActionResponse(response="You wake up in a dark room. You can't remember how you got here. You can see a door and a window. What do you do?", game_over=False)


@app.post("/action")
def action(request: Request, prompt: Prompt):
    """
    Receives an user action and requests the model with the previous saved session responses. Returns an answer.
    """
    # check if the user has a valid session
    if request.client.host not in SESSIONS:
        raise HTTPException(status_code=404, detail="Item not found")

    # add the user's prompt to the session history
    SESSIONS[request.client.host].extend(["Player: " + prompt.prompt])
    complete_prompt = "\n".join(SESSIONS[request.client.host]) + "\nNarrator: "

    # get the response from the model
    response = gpt(complete_prompt, stop=["Player:"])
    SESSIONS[request.client.host].extend(["Narrator: " + response])

    # check if the game is over
    if "game over" in response.lower():
        del SESSIONS[request.client.host]
        return ActionResponse(response=response, game_over=True)

    return ActionResponse(response=response, game_over=False)


@app.post("/critic")
def critic(request: Request, prompt: Prompt):

    return CriticResponse(
        pros=gpt(PROMPT_FORMATS['pros'].format(
            idea=prompt.prompt), temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
        cons=gpt(PROMPT_FORMATS['cons'].format(
            idea=prompt.prompt), temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
        state_of_the_art=gpt(PROMPT_FORMATS['state_of_the_art'].format(
            idea=prompt.prompt), max_tokens=200),
        pitch=gpt(PROMPT_FORMATS['pitch'].format(
            idea=prompt.prompt), max_tokens=100),
        viability=get_integer(gpt(PROMPT_FORMATS['viability'].format(
            idea=prompt.prompt), max_tokens=100)),
        innovation=get_integer(gpt(PROMPT_FORMATS['innovation'].format(
            idea=prompt.prompt), max_tokens=100)),
        resources=get_integer(gpt(PROMPT_FORMATS['resources'].format(
            idea=prompt.prompt), max_tokens=100)),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
