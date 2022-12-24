import openai
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

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

def execute_completion_model(prompt, model="text-davinci-002", temperature=1, max_tokens=100, *args, **kwargs):
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
    return [x.text for x in response.choices]

def open_ai_model_func(model, type='completion'):
    """
    Returns a function that executes the given model with the specified type.
    """
    if type == 'completion':
        def execute(prompt_text, *args, **kwargs):
            return execute_completion_model(prompt_text, model=model, *args, **kwargs)
        return execute

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

class Prompt(BaseModel):
    prompt: str

class Response(BaseModel):
    response: str
    game_over: bool

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
    return Response(response="You wake up in a dark room. You can't remember how you got here. You can see a door and a window. What do you do?", game_over=False)


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
    response = gpt(complete_prompt, stop=["Player:"])[0]
    SESSIONS[request.client.host].extend(["Narrator: " + response])

    # check if the game is over
    if "game over" in response.lower():
        del SESSIONS[request.client.host]
        return Response(response=response, game_over=True)

    return Response(response=response, game_over=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)