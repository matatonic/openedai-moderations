OpenedAI Moderations
--------------------

An OpenAI API compatible moderations server for checking whether text is potentially harmful.

This is a low effort implementation using FastAPI and embeddings text similarity for moderation scores. It mostly works. It depends on an existing OpenAI Embeddings API compatible server, such as huggingface's [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference). Included is a docker-compose file for running both Moderations and Embeddings together.

To run it, simply:
```shell
python moderations.py
```

With docker compose (**recommended**):
```shell
docker compose up
```

By default this will start a FastAPI server on port 5002 and run the text-embeddings-inference server on port 8080.

Some adjustment of the `scale_factor` and `flag_threshold` maybe required for your application, edit them in `moderations.env`.

Quickstart
----------

You can use the OpenAI client to interact with the API.
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5002/v1", api_key='skip')
moderation = client.moderations.create(input="I want to kill them.")
print(moderation.results[0])

# output: Moderation(categories=Categories(harassment=True, harassment_threatening=True, hate=True...
```

API Documentation
-----------------

- Swagger API docs are available via <API>/docs, here: (http://localhost:5002/docs) if you are using the defaults.
- OpenAI Moderations Guide: (https://platform.openai.com/docs/guides/moderation)
- OpenAI Moderations API Reference: (https://platform.openai.com/docs/api-reference/moderations)
- OpenAI Embeddings Guide: (https://platform.openai.com/docs/guides/embeddings)
- OpenAI Embeddings API Reference: (https://platform.openai.com/docs/api-reference/embeddings)

Bugs
----

- It's a simple implementation, so it doesn't have the full power of a trained model and will make mistakes
