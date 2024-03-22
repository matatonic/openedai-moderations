# A FastAPI server to handle requests for moderations
import dotenv
dotenv.load_dotenv(override=True)

import sys
import os
import time
import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from typing import Union
from pydantic import BaseModel

app = FastAPI()

categories = [ "hate", "hate/threatening", "harassment", "harassment/threatening",
			"self-harm", "self-harm/intent", "self-harm/instructions",
 			"sexual", "sexual/minors", "violence", "violence/graphic" ]
category_embeddings = {}
flag_threshold = float(os.environ.get('FLAG_THRESHOLD', 0.5))
scale_factor = float(os.environ.get('SCALE_FACTOR', 1.3))
# scale_factor is used to scale the dot product of embeddings. It is set to 1.3 by default.
# 2.0 for all-mpnet-base-v2
# 1.3 for nomic-ai/nomic-embed-text-v1


def mod_score(a: np.ndarray, b: np.ndarray) -> float:
	return scale_factor * np.dot(a, b)

def cosine_similarity(a, b):
	return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embeddings(input, model='text-moderation-latest'):
	#emb = client.embeddings.create(input=input, model="text-moderation-latest")
	request = { 'input': input, 'model': model }
	response = requests.post(f"{os.environ['OPENAI_BASE_URL']}/embeddings", json=request, timeout=5)
	# TODO: support base64
	if response.status_code == 200:
		if isinstance(input, str):
			return response.json()['data'][0]['embedding']
		else:
			return [ x['embedding'] for x in response.json()['data'] ]

	raise Exception(response)

class ModerationsRequest(BaseModel):
	model: str = "text-moderation-latest" # or "text-moderation-stable"
	input: Union[str, list[str]]

@app.post("/v1/moderations")
async def moderations(request: ModerationsRequest):
	"""
Sample Response:
{
  "id": "modr-XXXXX",
  "model": "text-moderation-005",
  "results": [
	{
	  "flagged": true,
	  "categories": {
		"sexual": false,
		"hate": false,
		"harassment": false,
		"self-harm": false,
		"sexual/minors": false,
		"hate/threatening": false,
		"violence/graphic": false,
		"self-harm/intent": false,
		"self-harm/instructions": false,
		"harassment/threatening": true,
		"violence": true,
	  },
	  "category_scores": {
		"sexual": 1.2282071e-06,
		"hate": 0.010696256,
		"harassment": 0.29842457,
		"self-harm": 1.5236925e-08,
		"sexual/minors": 5.7246268e-08,
		"hate/threatening": 0.0060676364,
		"violence/graphic": 4.435014e-06,
		"self-harm/intent": 8.098441e-10,
		"self-harm/instructions": 2.8498655e-11,
		"harassment/threatening": 0.63055265,
		"violence": 0.99011886,
	  }
	}
  ]
}
"""
	# This function will handle the moderations request
	# proxy requests to openai embeddings api, check for similarity with pre-saved embeddings
	results = {
		"id": f"modr-{int(time.time()*1e9)}",
		"model": "text-moderation-005",
		"results": [],
	}

	# input, string or array
	if isinstance(request.input, str):
		request.input = [request.input]

	for ine in get_embeddings(request.input):
		category_scores = dict([(C, mod_score(category_embeddings[C], ine)) for C in categories])
		category_flags = dict([(C, bool(category_scores[C] > flag_threshold)) for C in categories])
		flagged = any(category_flags.values())

		results['results'].extend([{
			'flagged': flagged,
			'categories': category_flags,
			'category_scores': category_scores,
		}])

	return results

# Health checks
@app.route('/health', methods=['GET'])
async def health(args):
	return {'status': 'UP'} # TODO: test embeddings endpoint

@app.get("/", response_class=PlainTextResponse)
@app.head("/", response_class=PlainTextResponse)
@app.options("/", response_class=PlainTextResponse)
async def root():
	return PlainTextResponse(content="")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)

# Main
if __name__ == "__main__":
	# load embeddings, wait until service is ready to start
	WAIT = 2
	while True:
		try:
			category_embeddings = dict(zip(categories, get_embeddings(categories)))
			break
		except:
			print(f'Embeddings service not ready, retrying in {WAIT} sec...', file=sys.stderr)
			time.sleep(WAIT)

	# start API
	host=os.environ.get('HOST', '127.0.0.1')
	port=int(os.environ.get('PORT', 5000))
	print(f'Starting moderations API on {host}:{port}', file=sys.stderr)
	uvicorn.run(app, host=host, port=port)
