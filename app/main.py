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

def check_content_filter(prompt):
    completions = openai.Completion.create(
        engine="content-filter-alpha",
        prompt=f'<|endoftext|>[{prompt}]\n--\nLabel:',
        max_tokens=1,
        temperature=0,
        top_p=0
    )
    print('content filter result', completions.choices[0].text)
    if completions.choices[0].text == '2':
        raise HTTPException(status_code=404, detail="Inappropriate input")

def remove_list_numeration(string):
    return re.sub(r'^\d\s*\.*|^-', '', string).strip()

def remove_empty_items(items):
    return [item for item in items if item.strip()]

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


def execute_completion_model(prompt, model="text-babbage-001", temperature=1, max_tokens=100, many=False, *args, **kwargs):# text-davinci-002
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
        print(response.choices[0])
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
    "*"
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

RESUME_PROMPT_FORMATS = """make a well formatted resume for this person:
{description}"""

SCRIPT_PROMPT_FORMAT = """Generate a script for {prompt}:"""


"""
Polarity: This measures the overall sentiment of a piece of text as positive, negative, or neutral.

Subjectivity: This measures how subjective or objective a piece of text is. Text that is more subjective is more likely to express personal opinions or feelings, while text that is more objective is more likely to present facts and information.

Valence: This measures the intensity of the sentiment expressed in a piece of text. A text with a high valence might be very positive or very negative, while a text with a low valence might be more neutral.

Emotion: This measures the specific emotions that are expressed in a piece of text, such as happiness, sadness, anger, fear, or surprise.
"""

SENTIMENT_PROMPT_FORMAT = """sentiment analysis for this messages:
t: text
Polarity from 1 to 10, Subjectivity from 1 to 10, Valence from 1 to 10, Emotion, Emoji
t: I had a terrible experience at that restaurant. The service was slow and the food was overcooked. I would never go back again.
1, 8, 8, Anger, 😠
t: {prompt}
"""
IDEAS_PROMPT_FORMAT = """generate 5 effective and useful ideas for {prompt}
separate each idea with only one line break, don't number them
Ideas:
"""

APP_FEATURES_PROMPT_FORMAT = """generate 5 new features for an app with this features:
{previous_features}
separate each idea with only one line break, don't number them
New unique Features:
"""


class ResumeRequest(BaseModel):
    description: str


class ResumeResponse(BaseModel):
    resume: str


class Prompt(BaseModel):
    prompt: str

class ListRequest(BaseModel):
    prompts: list


class ScriptRequest(BaseModel):
    name: str
    description: str
    target_audience: str
    key_selling_points: str
    desired_tone: str


class CopyRequest(BaseModel):
    name: str
    description: str
    target_audience: str
    key_selling_points: str
    desired_tone: str


class CopyResponse(BaseModel):
    copy_text: str


class TextResponse(BaseModel):
    text: str


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


class SentimentResponse(BaseModel):
    polarity: str
    subjectivity: str
    valence: str
    emotion: str
    emote: str


class ListResponse(BaseModel):
    results: list


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/start")
@limiter.limit("5/minute")
async def start_game(request: Request):
    user_ip = request.client.host
    create_session(user_ip)
    return ActionResponse(response="You wake up in a dark room. You can't remember how you got here. You can see a door and a window. What do you do?", game_over=False)


@app.post("/action")
@limiter.limit("40/minute")
async def action(request: Request, prompt: Prompt):
    """
    Receives an user action and requests the model with the previous saved session responses. Returns an answer.
    """
    check_content_filter(prompt.prompt)
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
async def critic(request: Request, prompt: Prompt):
    check_content_filter(prompt.prompt)

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
async def copy(request: Request, copy_request: CopyRequest):
    rendered_prompt = ' '.join([
        copy_request.name,
        copy_request.description,
        copy_request.target_audience,
        copy_request.key_selling_points,
        copy_request.desired_tone,
    ])
    check_content_filter(rendered_prompt)

    return CopyResponse(
        copy_text=gpt(rendered_prompt, temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
    )


@app.post("/resume")
@limiter.limit("5/minute")
async def resume(request: Request, resume_request: ResumeRequest):
    check_content_filter(resume_request.description)
    return ResumeResponse(
        resume=gpt(RESUME_PROMPT_FORMATS.format(
            description=resume_request.description,
        ), temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
    )


@app.post("/script")
@limiter.limit("5/minute")
async def script(request: Request, prompt: Prompt):

    return TextResponse(
        text=gpt(SCRIPT_PROMPT_FORMAT.format(
            prompt=prompt.prompt,
        ), temperature=0.1, max_tokens=400, presence_penalty=2, frequency_penalty=2),
    )


@app.post("/sentiment")
@limiter.limit("5/minute")
async def sentiment(request: Request, prompt: Prompt):
    check_content_filter(prompt.promp)
    res = gpt(SENTIMENT_PROMPT_FORMAT.format(
        prompt=prompt.prompt,
    ), temperature=0.1, max_tokens=100, presence_penalty=2, frequency_penalty=2)
    results = res.replace(" ", '').split(",")
    return SentimentResponse(
        polarity=results[0],
        subjectivity=results[1],
        valence=results[2],
        emotion=results[3],
        emote=results[4],
    )


@app.post("/ideas")
@limiter.limit("10/minute")
async def ideas(request: Request, prompt: Prompt):
    check_content_filter(prompt.prompt)
    res = gpt(IDEAS_PROMPT_FORMAT.format(
        prompt=prompt.prompt,
    ), temperature=0.9, max_tokens=500, presence_penalty=2, frequency_penalty=2)
    results = [remove_list_numeration(x) for x in res.split("\n")]
    results = remove_empty_items(results)
    return ListResponse(
        results=results,
    )


@app.post("/app-features")
@limiter.limit("10/minute")
async def app_features(request: Request, features: ListRequest):
    check_content_filter(features.prompt)
    res = gpt(APP_FEATURES_PROMPT_FORMAT.format(
        previous_features='\n'.join(features.prompts),
    ), temperature=0.9, max_tokens=500, presence_penalty=2, frequency_penalty=2)
    results = [remove_list_numeration(x.strip()) for x in res.split("\n")]
    results = remove_empty_items(results)
    return ListResponse(
        results=results,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
