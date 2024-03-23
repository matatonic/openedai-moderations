# A FastAPI server to handle requests for moderations
import dotenv
dotenv.load_dotenv(override=True)

import sys
import time
import argparse
import uvicorn
from typing import Union
from pydantic import BaseModel

import torch
import openedai

app = openedai.OpenAIStub()
moderation = None

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

	# minor name adjustments
	mod_cat_map =  {
		"harassment": "harassment",
		"harassment-threatening": "harassment/threatening",
		"hate": "hate",
		"hate-threatening": "hate/threatening",
		"self-harm": "self-harm",
		"self-harm-instructions": "self-harm/instructions",
		"self-harm-intent": "self-harm/intent",
		"sexual": "sexual",
		"sexual-minors": "sexual/minors",
		"violence": "violence",
		"violence-graphic": "violence/graphic",
	}

	for embeddings_for_prediction in mod.getEmbeddings(request.input).tolist():
		prediction = mod.predict(moderation, embeddings_for_prediction)
		category_scores = dict([ (mod_cat_map[C], score) for C, score in prediction['category_scores'].items() ])
		category_flags = dict([ (mod_cat_map[C], flagged) for C, flagged in prediction['detect'].items() ])
		flagged = prediction['detected']

		results['results'].extend([{
			'flagged': flagged,
			'categories': category_flags,
			'category_scores': category_scores,
		}])

	return results

def parse_args(argv):
	parser = argparse.ArgumentParser(description='Moderation API')
	parser.add_argument('--host', type=str, default='0.0.0.0')
	parser.add_argument('--port', type=int, default=5002)
	parser.add_argument('--test-load', action='store_true')
	return parser.parse_args(argv)

# Main
if __name__ == "__main__":

	args = parse_args(sys.argv[1:])

	device = "cuda" if torch.cuda.is_available() else "cpu"
	# start API
	print(f'Starting moderations[{device}] API on {args.host}:{args.port}', file=sys.stderr)

	import repos.moderation_by_embeddings.moderation as mod
	# Load model
	moderation = mod.ModerationModel()
	moderation.load_state_dict(torch.load('repos/moderation_by_embeddings/moderation_model.pth', map_location=torch.device(device)))

	app.register_model('text-moderations-latest', 'text-moderations-stable')
	app.register_model('text-moderations-005', 'text-moderations-ifmain')

	if not args.test_load:
		uvicorn.run(app, host=args.host, port=args.port)
