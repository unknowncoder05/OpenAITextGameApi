import openai
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

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
limiter = Limiter(key_func=get_remote_address)

openai.api_key = os.getenv("OPENAPI_API_KEY")
gpt = open_ai_model_func("text-davinci-002")
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    pitch="top investors from y combinator write pitch for the idea \"{idea}\":",
    plan="top investors from y combinator write Business Plan for the idea \"{idea}\":",
    survey="write a survey for the idea \"{idea}\":",

    viability="give a integer score from 1 to 10 to the viability of the idea in real life \"{idea}\":",
    innovation="give a integer score from 1 to 10 to the innovation of the idea \"{idea}\":",
    resources="give a integer score from 1 to 10 to the technological or logistical resources needed for the idea \"{idea}\":",
)

"""
The product or service being marketed: The prompt should clearly describe the product or service being marketed, including its features and benefits. This will help GPT generate marketing copy that accurately reflects the product or service.

The target audience: The prompt should specify the target audience for the marketing campaign, including their demographics, interests, and needs. This will help GPT generate marketing copy that is relevant and appealing to the intended audience.

The key selling points: The prompt should highlight the key selling points of the product or service, such as its unique features, benefits, or competitive advantage. This will help GPT generate marketing copy that emphasizes the most compelling aspects of the product or service.

The desired tone: The prompt should specify the desired tone for the marketing copy, such as professional, casual, or humorous. This will help GPT generate marketing copy that is appropriate for the target audience and aligns with the overall tone of the campaign.
"""
COPY_PROMPT_FORMAT = """Generate general marketing copy for {name}
description: {description}
target audience: {target_audience}
key selling points: {key_selling_points}
desired tone: {desired_tone}
"""

RESUME_PROMPT_FORMATS = dict(
    pros="""make a resume for this person "{description}":""",
)


class ResumeRequest(BaseModel):
    description: str


class ResumeResponse(BaseModel):
    resume: str


class Prompt(BaseModel):
    prompt: str


class CopyRequest(BaseModel):
    name: str
    description: str
    target_audience: str
    key_selling_points: str
    desired_tone: str


class CopyResponse(BaseModel):
    copy_text: str


class ActionResponse(BaseModel):
    response: str
    game_over: bool


class CriticResponse(BaseModel):
    pros: str
    cons: str
    state_of_the_art: str
    pitch: str
    plan: str
    survey: str

    viability: int
    innovation: int
    resources: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/start")
@limiter.limit("5/minute")
def start_game(request: Request):
    user_ip = request.client.host
    create_session(user_ip)
    return ActionResponse(response="You wake up in a dark room. You can't remember how you got here. You can see a door and a window. What do you do?", game_over=False)


@app.post("/action")
@limiter.limit("5/minute")
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
@limiter.limit("5/minute")
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
        plan=gpt(PROMPT_FORMATS['plan'].format(
            idea=prompt.prompt), max_tokens=400),
        survey=gpt(PROMPT_FORMATS['survey'].format(
            idea=prompt.prompt), max_tokens=400),

        viability=get_integer(gpt(PROMPT_FORMATS['viability'].format(
            idea=prompt.prompt), max_tokens=100)),
        innovation=get_integer(gpt(PROMPT_FORMATS['innovation'].format(
            idea=prompt.prompt), max_tokens=100)),
        resources=get_integer(gpt(PROMPT_FORMATS['resources'].format(
            idea=prompt.prompt), max_tokens=100)),
    )


@app.post("/copy")
@limiter.limit("5/minute")
def copy(request: Request, copy_request: CopyRequest):

    return CopyResponse(
        copy_text=gpt(COPY_PROMPT_FORMAT.format(
            name=copy_request.name,
            description=copy_request.description,
            target_audience=copy_request.target_audience,
            key_selling_points=copy_request.key_selling_points,
            desired_tone=copy_request.desired_tone,
        ), temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
    )


@app.post("/resume")
@limiter.limit("5/minute")
def resume(request: Request, resume_request: ResumeRequest):

    return ResumeResponse(
        resume=gpt(RESUME_PROMPT_FORMATS.format(
            description=resume_request.description,
        ), temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
